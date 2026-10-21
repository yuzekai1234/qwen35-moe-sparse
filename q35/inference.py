"""Inference-only fast paths for the repository model, applied by q35.evaluate.

The model source (shared with training and carried by HF exports) is unchanged; these
replacements are forward-only and keep its numerics up to BF16 rounding order:

- Routed experts: transformers' grouped_mm falls back on non-Hopper GPUs with torch < 2.9 to
  a per-expert loop with two host syncs per layer. Two Triton grouped GEMMs replace it,
  routing on the device and fusing the token gather, SiLU gating and routing weights.
- Gated DeltaNet prefill: FLA's Triton causal conv1d on the [B, T, D] layout and its native
  grouped-value heads replace the reference conv (two transposes) and repeat_interleave.
- RMSNorm / gated RMSNorm: torch.compile fuses the FP32 elementwise chains.

Keep _gdn_prefill_forward in sync with Qwen3_5MoeGatedDeltaNet.forward when editing the model.
"""
import types

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from .models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeGatedDeltaNet, Qwen3_5MoeRMSNorm, Qwen3_5MoeRMSNormGated,
)

EXPERTS_IMPLEMENTATION = "q35_triton"


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


# Tuned on an RTX 4090 for hidden = intermediate = 192, 262144 tokens, top-4 of 64 experts.
GATE_UP_CONFIG = dict(BLOCK_M=128, BLOCK_N=64, BLOCK_K=64, num_warps=4, num_stages=2)
DOWN_CONFIG = dict(BLOCK_M=64, BLOCK_N=256, BLOCK_K=32, num_warps=8, num_stages=3)


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

    def launch(kernel, config, n, *args, **meta):
        block_m = config["BLOCK_M"]
        blk_cum = torch.cumsum((counts + block_m - 1) // block_m, 0, dtype=torch.int32)
        # Upper bound on non-empty M-blocks; surplus programs exit immediately.
        grid = (triton.cdiv(num_pairs, block_m) + num_experts, triton.cdiv(n, config["BLOCK_N"]))
        kernel[grid](blk_cum, counts, starts, *args, **meta, **config)

    h = hidden_states.new_empty(num_pairs, intermediate)
    gate_up = self.gate_up_proj
    launch(_gate_up_kernel, GATE_UP_CONFIG, intermediate,
           hidden_states, gate_up, h, perm,
           hidden, intermediate, hidden_states.stride(0), gate_up.stride(0), gate_up.stride(1), h.stride(0),
           TOPK=top_k, **common)
    y = hidden_states.new_empty(num_pairs, hidden)
    down = self.down_proj
    weights = top_k_weights.reshape(-1)
    launch(_down_kernel, DOWN_CONFIG, hidden,
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


def compile_inference_norms(model):
    """Fuse normalization without tracing dynamic expert dispatch or the FLA kernels."""
    count = 0
    for module in model.modules():
        if isinstance(module, (Qwen3_5MoeRMSNorm, Qwen3_5MoeRMSNormGated)):
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
        if isinstance(module, Qwen3_5MoeGatedDeltaNet):
            module._reference_forward = module.forward
            module.forward = types.MethodType(_gdn_prefill_forward, module)
            gdn += 1
    return dict(experts=model.config._experts_implementation, gated_deltanet=gdn, norms=compile_inference_norms(model))
