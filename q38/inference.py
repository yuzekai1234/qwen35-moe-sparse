"""Inference-only fast paths for the repository model, applied by q38.evaluate.

The model source (shared with training and carried by HF exports) is unchanged; these
replacements are forward-only and keep its numerics up to BF16 rounding order:

- Routed experts: transformers' grouped_mm falls back on non-Hopper GPUs with torch < 2.9 to
  a per-expert loop with two host syncs per layer. Two Triton grouped GEMMs replace it,
  routing on the device and fusing the token gather, SiLU gating and routing weights.
  Tile sizes are benchmarked once per GPU model and expert shape on the first call (warmup).
- Gated DeltaNet prefill: FLA's Triton causal conv1d on the [B, T, D] layout and its native
  grouped-value heads replace the reference conv (two transposes) and repeat_interleave.
- RMSNorm / gated RMSNorm: torch.compile fuses the FP32 elementwise chains.
- mHC: one Triton pass reads the FP32 residual branches once for both the RMS and the
  low-rank projection (3xTF32, no less accurate than the FP32 SGEMM it replaces), and the
  4x4 branch mix is a fused broadcast-sum instead of a cuBLAS batched GEMM per token.
  Opt-in reduced precision (changes numerics; off by default): Q38_MHC_BF16_PROJ=1 runs the
  projection as one BF16 tensor-core pass, Q38_MHC_BF16_STREAM=1 stores the residual branches
  in BF16 between sublayers (the mix itself still computes in FP32).

Keep _gdn_prefill_forward in sync with Qwen3_8MoeGatedDeltaNet.forward and _mhc_pre_forward /
_mhc_post_forward in sync with MHC when editing the model.
"""
import os
import types

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.testing import do_bench

from .models.qwen3_8_moe.modeling_qwen3_8_moe import (
    MHC, Qwen3_8MoeGatedDeltaNet, Qwen3_8MoeRMSNorm, Qwen3_8MoeRMSNormGated, _agg, _sinkhorn,
)

EXPERTS_IMPLEMENTATION = "q38_triton"


@triton.jit
def _locate_block(pid_m, blk_cum_ptr, counts_ptr, starts_ptr, E: tl.constexpr, E_PAD: tl.constexpr,
                  BLOCK_M: tl.constexpr):
    """Map a global M-block to (expert, first sorted row, valid rows); surplus blocks get 0 rows."""
    offs_e = tl.arange(0, E_PAD)
    blk_cum = tl.load(blk_cum_ptr + offs_e, mask=offs_e < E, other=2**30)
    e = tl.sum((blk_cum <= pid_m).to(tl.int32), axis=0)
    e_safe = tl.minimum(e, E - 1)
    count = tl.load(counts_ptr + e_safe)
    local = (pid_m - tl.load(blk_cum_ptr + e_safe) + tl.cdiv(count, BLOCK_M)) * BLOCK_M
    row0 = tl.load(starts_ptr + e_safe) + local
    return e_safe, row0, tl.where(e < E, count - local, 0)


@triton.jit
def _dot(a, b, acc, IEEE: tl.constexpr):
    if IEEE:
        return tl.dot(a, b, acc, input_precision="ieee")
    return tl.dot(a, b, acc)


@triton.jit
def _gate_up_kernel(
    blk_cum_ptr, counts_ptr, starts_ptr, x_ptr, w_ptr, h_ptr, perm_ptr,
    K, N, stride_x, stride_we, stride_wn, stride_h,
    E: tl.constexpr, E_PAD: tl.constexpr, TOPK: tl.constexpr, IEEE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """h[sorted row] = silu(x[token] @ W_gate[e]^T) * (x[token] @ W_up[e]^T)."""
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    e, row0, valid = _locate_block(pid_m, blk_cum_ptr, counts_ptr, starts_ptr, E, E_PAD, BLOCK_M)
    if valid <= 0:
        return
    offs_m = tl.arange(0, BLOCK_M)
    mask_m = offs_m < valid
    token = (tl.load(perm_ptr + row0 + offs_m, mask=mask_m, other=0) // TOPK).to(tl.int64)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    offs_k = tl.arange(0, BLOCK_K)
    acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    w_base = w_ptr + e.to(tl.int64) * stride_we
    for k0 in range(0, K, BLOCK_K):
        mask_k = k0 + offs_k < K
        a = tl.load(x_ptr + token[:, None] * stride_x + (k0 + offs_k)[None, :],
                    mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w_mask = mask_n[None, :] & mask_k[:, None]
        w_gate = tl.load(w_base + offs_n[None, :] * stride_wn + (k0 + offs_k)[:, None], mask=w_mask, other=0.0)
        w_up = tl.load(w_base + (offs_n + N)[None, :] * stride_wn + (k0 + offs_k)[:, None], mask=w_mask, other=0.0)
        acc_g = _dot(a, w_gate, acc_g, IEEE)
        acc_u = _dot(a, w_up, acc_u, IEEE)
    # Round like the eager path: gate_up output, silu output, then their product.
    dtype = h_ptr.dtype.element_ty
    g = acc_g.to(dtype).to(tl.float32)
    s = (g * tl.sigmoid(g)).to(dtype).to(tl.float32)
    out = s * acc_u.to(dtype).to(tl.float32)
    rows = (row0 + offs_m).to(tl.int64)
    tl.store(h_ptr + rows[:, None] * stride_h + offs_n[None, :], out.to(dtype), mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def _down_kernel(
    blk_cum_ptr, counts_ptr, starts_ptr, h_ptr, w_ptr, y_ptr, perm_ptr, rw_ptr,
    K, N, stride_h, stride_we, stride_wn, stride_y,
    E: tl.constexpr, E_PAD: tl.constexpr, IEEE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """y[pair] = (h[sorted row] @ W_down[e]^T) * routing_weight[pair], written back in pair order."""
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    e, row0, valid = _locate_block(pid_m, blk_cum_ptr, counts_ptr, starts_ptr, E, E_PAD, BLOCK_M)
    if valid <= 0:
        return
    offs_m = tl.arange(0, BLOCK_M)
    mask_m = offs_m < valid
    rows = (row0 + offs_m).to(tl.int64)
    pair = tl.load(perm_ptr + rows, mask=mask_m, other=0).to(tl.int64)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    w_base = w_ptr + e.to(tl.int64) * stride_we
    for k0 in range(0, K, BLOCK_K):
        mask_k = k0 + offs_k < K
        a = tl.load(h_ptr + rows[:, None] * stride_h + (k0 + offs_k)[None, :],
                    mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w = tl.load(w_base + offs_n[None, :] * stride_wn + (k0 + offs_k)[:, None],
                    mask=mask_n[None, :] & mask_k[:, None], other=0.0)
        acc = _dot(a, w, acc, IEEE)
    dtype = y_ptr.dtype.element_ty
    weight = tl.load(rw_ptr + pair, mask=mask_m, other=0.0).to(tl.float32)
    out = acc.to(dtype).to(tl.float32) * weight[:, None]
    tl.store(y_ptr + pair[:, None] * stride_y + offs_n[None, :], out.to(dtype), mask=mask_m[:, None] & mask_n[None, :])


def _expert_configs(block_n):
    return [dict(BLOCK_M=m, BLOCK_N=n, BLOCK_K=k, num_warps=w, num_stages=s)
            for m in (64, 128) for n in block_n for k in (32, 64) for w in (4, 8) for s in (2, 3)]


# Candidates for the first-call benchmark; the best depends on the GPU and the expert shape.
GATE_UP_CONFIGS = _expert_configs((64, 128))
DOWN_CONFIGS = _expert_configs((64, 128, 256))
_TUNED = {}


def _tuned_config(key, candidates, run):
    """Fastest candidate for key, benchmarked once per process; candidates that fail to compile are skipped."""
    if key not in _TUNED:
        timings = []
        for config in candidates:
            try:
                timings.append((do_bench(lambda: run(config), warmup=5, rep=20), config))
            except triton.runtime.errors.OutOfResources:
                continue
        _TUNED[key] = min(timings, key=lambda t: t[0])[1]
        print(f"Tuned {key}: {_TUNED[key]}", flush=True)
    return _TUNED[key]


def triton_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    """Forward-only grouped experts with the math of transformers' grouped_mm path and no host sync."""
    if torch.is_grad_enabled() and (hidden_states.requires_grad or self.gate_up_proj.requires_grad):
        raise RuntimeError("The Triton experts path has no backward; run it under torch.inference_mode()")
    num_tokens, hidden = hidden_states.shape
    top_k = top_k_index.shape[-1]
    num_experts, intermediate = self.num_experts, self.intermediate_dim
    expert_ids = top_k_index.reshape(-1)
    num_pairs = expert_ids.numel()
    sorted_ids, perm = torch.sort(expert_ids, stable=True)
    # Expert boundaries in sorted order; searchsorted avoids contended atomics and host syncs.
    bounds = torch.searchsorted(sorted_ids, torch.arange(num_experts + 1, device=sorted_ids.device, dtype=sorted_ids.dtype))
    starts = bounds[:-1].to(torch.int32)
    counts = (bounds[1:] - bounds[:-1]).to(torch.int32)
    common = dict(E=num_experts, E_PAD=triton.next_power_of_2(num_experts), IEEE=hidden_states.dtype == torch.float32)

    def launch(kernel, candidates, k, n, *args, **meta):
        def run(config):
            block_m = config["BLOCK_M"]
            blk_cum = torch.cumsum((counts + block_m - 1) // block_m, 0, dtype=torch.int32)
            # Upper bound on non-empty M-blocks; surplus programs exit immediately.
            grid = (triton.cdiv(num_pairs, block_m) + num_experts, triton.cdiv(n, config["BLOCK_N"]))
            kernel[grid](blk_cum, counts, starts, *args, **meta, **config)
        key = (kernel.__name__, torch.cuda.get_device_name(hidden_states.device), hidden_states.dtype,
               k, n, num_experts, top_k)
        run(_tuned_config(key, candidates, run))

    h = hidden_states.new_empty(num_pairs, intermediate)
    gate_up = self.gate_up_proj
    launch(_gate_up_kernel, GATE_UP_CONFIGS, hidden, intermediate,
           hidden_states, gate_up, h, perm,
           hidden, intermediate, hidden_states.stride(0), gate_up.stride(0), gate_up.stride(1), h.stride(0),
           TOPK=top_k, **common)
    y = hidden_states.new_empty(num_pairs, hidden)
    down = self.down_proj
    weights = top_k_weights.reshape(-1)
    launch(_down_kernel, DOWN_CONFIGS, intermediate, hidden,
           h, down, y, perm, weights,
           intermediate, hidden, h.stride(0), down.stride(0), down.stride(1), y.stride(0),
           **common)
    # Same deterministic top-k reduction as grouped_mm (no atomics).
    return y.view(num_tokens, top_k, hidden).sum(dim=1)


def _gdn_prefill_forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
    """Uncached full-sequence forward; falls back to the model code for caches, padding masks or varlen."""
    if cache_params is not None or attention_mask is not None or kwargs.get("cu_seq_lens_q") is not None:
        return self._reference_forward(hidden_states, cache_params=cache_params, attention_mask=attention_mask, **kwargs)
    from fla.modules.conv.causal_conv1d import causal_conv1d
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    batch_size, seq_len, _ = hidden_states.shape
    mixed_qkv, _ = causal_conv1d(
        self.in_proj_qkv(hidden_states), weight=self.conv1d.weight.squeeze(1), bias=self.conv1d.bias,
        activation=self.activation,
    )
    query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
    query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)
    z = self.in_proj_z(hidden_states)
    beta = self.in_proj_b(hidden_states).sigmoid()
    g = -self.A_log.float().exp() * F.softplus(self.in_proj_a(hidden_states).float() + self.dt_bias)
    # Value heads > key heads use FLA's grouped-value attention instead of repeating q/k.
    core_attn_out, _ = chunk_gated_delta_rule(query, key, value, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
    core_attn_out = self.norm(core_attn_out.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
    return self.out_proj(core_attn_out.reshape(batch_size, seq_len, -1))


def _rms_lowrank_configs():
    return [triton.Config(dict(BLOCK_M=m, BLOCK_K=k), num_warps=w, num_stages=s)
            for m in (32, 64, 128) for k in (16, 32) for w in (4, 8) for s in (3, 4)]


@triton.autotune(_rms_lowrank_configs(), key=["K", "R", "BF16_DOT"])
@triton.jit
def _rms_lowrank_kernel(x_ptr, a_ptr, z_ptr, M, K, R, stride_x, stride_a, stride_z, eps,
                        BF16_DOT: tl.constexpr, R_PAD: tl.constexpr, BLOCK_M: tl.constexpr,
                        BLOCK_K: tl.constexpr):
    """z = rmsnorm(x) @ A^T as rsqrt(mean(x^2) + eps) * (x @ A^T), reading each row of x once."""
    offs_m = (tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    mask_m = offs_m < M
    offs_r = tl.arange(0, R_PAD)
    mask_r = offs_r < R
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, R_PAD), dtype=tl.float32)
    sum_sq = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        mask_k = k0 + offs_k < K
        x = tl.load(x_ptr + offs_m[:, None] * stride_x + (k0 + offs_k)[None, :],
                    mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
        a = tl.load(a_ptr + offs_r[None, :] * stride_a + (k0 + offs_k)[:, None],
                    mask=mask_r[None, :] & mask_k[:, None], other=0.0)
        if BF16_DOT:
            acc = tl.dot(x.to(tl.bfloat16), a.to(tl.bfloat16), acc)
        else:
            acc = tl.dot(x, a, acc, input_precision="tf32x3")
        sum_sq += tl.sum(x * x, axis=1)
    inv = 1.0 / tl.sqrt(sum_sq / K + eps)
    tl.store(z_ptr + offs_m[:, None] * stride_z + offs_r[None, :], acc * inv[:, None],
             mask=mask_m[:, None] & mask_r[None, :])


def rms_lowrank(x, a, eps=1e-6, bf16_dot=False):
    """[M, K] x (FP32 or BF16), FP32 [R, K] a -> FP32 [M, R]; matches F.linear(x * rsqrt(mean(x^2) + eps), a)."""
    M, K = x.shape
    R = a.shape[0]
    z = x.new_empty(M, R, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
    _rms_lowrank_kernel[grid](x, a, z, M, K, R, x.stride(0), a.stride(0), z.stride(0), eps,
                              BF16_DOT=bf16_dot, R_PAD=max(16, triton.next_power_of_2(R)))
    return z


def _mhc_coeffs_agg(X, lowrank, B, b_pre, b_post, b_res, alphas, n, iters):
    # Same as the model's _coeffs after its first projection, followed by _agg.
    z = torch.tanh(F.linear(lowrank, B))
    zp, zq, zr = z[..., :n], z[..., n:2 * n], z[..., 2 * n:].reshape(*z.shape[:-1], n, n)
    pre = torch.sigmoid(b_pre + alphas[0] * zp)
    post = 2 * torch.sigmoid(b_post + alphas[1] * zq)
    res = _sinkhorn(b_res + alphas[2] * zr, iters)
    return _agg(X, pre), pre, post, res


def _mix_fused(X, res, post, y):
    # einsum("bsmn,bsnh->bsmh") as a broadcast-sum so Inductor fuses it with the update.
    return (res.unsqueeze(-1) * X.unsqueeze(2)).sum(3) + post.unsqueeze(-1) * y.float().unsqueeze(2)


def _mix_fused_bf16(X, res, post, y):
    # Rounds to BF16 inside the fused kernel, so the FP32 result is never written.
    return _mix_fused(X, res, post, y).to(torch.bfloat16)


_mhc_coeffs_agg_c = torch.compile(_mhc_coeffs_agg)
_mix_fused_c = torch.compile(_mix_fused)
_mix_fused_bf16_c = torch.compile(_mix_fused_bf16)


def _mhc_pre_forward(self, X):
    if not X.is_cuda:
        return self._reference_pre(X)
    with torch.autocast("cuda", enabled=False):
        if not self._bf16_stream:
            X = X.float()
        lowrank = rms_lowrank(X.reshape(-1, X.shape[2] * X.shape[3]), self.A.weight.float(),
                              bf16_dot=self._bf16_proj)
        xin, pre, post, res = _mhc_coeffs_agg_c(
            X, lowrank.view(*X.shape[:2], -1), self.B.weight.float(), self.b_pre.float(), self.b_post.float(),
            self.b_res.float(), self.alphas.float(), self.n, self.iters,
        )
        return xin, (X, pre, post, res)


def _mhc_post_forward(self, X, y, c):
    if not X.is_cuda:
        return self._reference_post(X, y, c)
    return (_mix_fused_bf16_c if self._bf16_stream else _mix_fused_c)(c[0], c[3], c[2], y)


def compile_inference_norms(model):
    """Fuse normalization without tracing dynamic expert dispatch or the FLA kernels."""
    count = 0
    for module in model.modules():
        if isinstance(module, (Qwen3_8MoeRMSNorm, Qwen3_8MoeRMSNormGated)):
            module.compile(
                fullgraph=True, dynamic=True,
                # Preserve intermediate BF16/FP16 rounding, including the gated norm's
                # casts before weighting/gating; the default compiler elides these.
                options={"emulate_precision_casts": True},
            )
            count += 1
    return count


def optimize_for_inference(model):
    """Install all fast paths on a loaded model in place; returns counts for logging."""
    from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS

    ALL_EXPERTS_FUNCTIONS.register(EXPERTS_IMPLEMENTATION, triton_experts_forward)
    model.set_experts_implementation(EXPERTS_IMPLEMENTATION)
    gdn = 0
    for module in model.modules():
        if isinstance(module, Qwen3_8MoeGatedDeltaNet):
            module._reference_forward = module.forward
            module.forward = types.MethodType(_gdn_prefill_forward, module)
            gdn += 1
    mhc = 0
    bf16_proj, bf16_stream = (os.environ.get(k) == "1" for k in ("Q38_MHC_BF16_PROJ", "Q38_MHC_BF16_STREAM"))
    for module in model.modules():
        if isinstance(module, MHC):
            module._bf16_proj, module._bf16_stream = bf16_proj, bf16_stream
            module._reference_pre, module._reference_post = module.pre, module.post
            module.pre = types.MethodType(_mhc_pre_forward, module)
            module.post = types.MethodType(_mhc_post_forward, module)
            mhc += 1
    return dict(experts=model.config._experts_implementation, gated_deltanet=gdn, mhc=mhc,
                mhc_bf16_proj=bf16_proj, mhc_bf16_stream=bf16_stream, norms=compile_inference_norms(model))


def padded_batched_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    """Differentiable tensor-core expert GEMMs on SM80 with older Torch.

    Torch <=2.8's grouped_mm is Hopper-only and otherwise launches one GEMM per
    expert. Pack exact routed rows into a padded batch instead. Capacity is the
    actual maximum expert load, rounded to 16; no token is dropped or rerouted.
    The one scalar read per layer replaces the grouped fallback's forward and
    backward host offset reads and hundreds of tiny GEMM launches.
    """
    tokens, hidden = hidden_states.shape
    top_k = top_k_index.shape[1]
    ids = top_k_index.reshape(-1)
    sorted_ids, permutation = torch.sort(ids, stable=True)
    counts = torch.bincount(ids, minlength=self.num_experts)
    capacity = ((int(counts.max()) + 15) // 16) * 16
    starts = counts.cumsum(0) - counts
    local = torch.arange(ids.numel(), device=ids.device) - starts[sorted_ids]
    padded_rows = sorted_ids * capacity + local
    rows = hidden_states[permutation // top_k]
    padded = hidden_states.new_zeros(self.num_experts * capacity, hidden)
    padded = padded.index_copy(0, padded_rows, rows).view(self.num_experts, capacity, hidden)
    gate, up = torch.bmm(padded, self.gate_up_proj.transpose(1, 2)).chunk(2, dim=-1)
    activated = self.act_fn(gate) * up
    raw = torch.bmm(activated, self.down_proj.transpose(1, 2))
    routed = raw.reshape(-1, hidden)[padded_rows] * top_k_weights.reshape(-1)[permutation, None]
    output = routed.new_zeros(ids.numel(), hidden).index_copy(0, permutation, routed)
    return output.view(tokens, top_k, hidden).sum(1).to(hidden_states.dtype)
