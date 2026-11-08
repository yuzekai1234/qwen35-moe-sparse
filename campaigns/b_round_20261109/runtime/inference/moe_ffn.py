"""Inference-only FP8 MoE FFN for the prequantized trunk (routed experts plus a gated shared expert).

The block receives the FP8 rows produced by the residual/RMSNorm kernel. The router and the
shared-expert gate run as one FP8 GEMM; routing follows the checkpoint's router (softmax in
FP32, top-k, renormalized, BF16 weights), computed by one Triton kernel. The shared expert uses
the packed FP8 dense SwiGLU of dense_ffn.py.

Routed experts gather their FP8 rows in expert order, run gate_up, the dense SwiGLU quantization
kernel and down, and one Triton kernel then forms each token's weighted sum of its k expert rows
plus the gated shared expert. By default gate_up and down are one cuBLAS torch._scaled_mm per
expert, inside a custom op that torch.compile does not trace: the host reads the expert row
offsets once per block, which only drains the queue because the evaluation uses no CUDA graphs.
Q38_MOE_ROUTED=grouped uses torch._scaled_grouped_mm (CUTLASS) instead, without the host read
(usable under CUDA graphs, about 7% slower at the C4 widths).
"""
import os
import types

import torch
import triton
import triton.language as tl

import dense_ffn
from low_precision import quant_rows as plain_quant_rows
from prequant_norm import quant_rows


def _shared(self, q, scale, rows):
    gu = torch._scaled_mm(q, self._shared_q_gate_up.t(), scale_a=scale[:, None],
                          scale_b=self._shared_s_gate_up[None, :], out_dtype=torch.bfloat16, use_fast_accum=True)
    width = self._shared_width
    hidden = torch.empty((rows, width), device=q.device, dtype=torch.float8_e4m3fn)
    hidden_scale = torch.empty(rows, device=q.device, dtype=torch.float32)
    dense_ffn._swiglu_quant[(rows,)](gu, hidden, hidden_scale, width, triton.next_power_of_2(width), num_warps=4)
    return torch._scaled_mm(hidden, self._shared_q_down.t(), scale_a=hidden_scale[:, None],
                            scale_b=self._shared_s_down[None, :], out_dtype=torch.bfloat16, use_fast_accum=True)


@triton.jit
def _route(LOGITS, W, IDX, T, STRIDE, E: tl.constexpr, E_PAD: tl.constexpr, K: tl.constexpr, BLOCK_T: tl.constexpr):
    """Top-k of softmax(logits[:, :E]) renormalized over the k choices, i.e. softmax over the k largest logits."""
    rows = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    cols = tl.arange(0, E_PAD)
    live = rows < T
    x = tl.load(LOGITS + rows[:, None].to(tl.int64) * STRIDE + cols[None, :], live[:, None] & (cols < E)[None, :],
                other=float('-inf')).to(tl.float32)
    p = tl.exp(x - tl.max(x, 1)[:, None])
    left = p
    total = tl.zeros((BLOCK_T,), tl.float32)
    for _ in tl.static_range(K):
        best = tl.max(left, 1)
        total += best
        index = tl.min(tl.where(left == best[:, None], cols[None, :], E_PAD), 1)
        left = tl.where(cols[None, :] == index[:, None], -1.0, left)
    left = p
    for j in tl.static_range(K):
        best = tl.max(left, 1)
        index = tl.min(tl.where(left == best[:, None], cols[None, :], E_PAD), 1)
        left = tl.where(cols[None, :] == index[:, None], -1.0, left)
        tl.store(W + rows * K + j, (best / total).to(tl.bfloat16), live)
        tl.store(IDX + rows * K + j, index.to(tl.int64), live)


@triton.jit
def _combine(Y, POS, W, SHARED, LOGITS, OUT, LOGIT_STRIDE, GATE_COLUMN, H: tl.constexpr, K: tl.constexpr,
             BLOCK: tl.constexpr):
    """out[t] = sum_j w[t, j] * y[pos[t, j]] + sigmoid(gate logit[t]) * shared[t], accumulated in FP32."""
    t = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = col < H
    gate = tl.sigmoid(tl.load(LOGITS + t * LOGIT_STRIDE + GATE_COLUMN).to(tl.float32))
    acc = gate * tl.load(SHARED + t * H + col, mask, other=0).to(tl.float32)
    for j in tl.static_range(K):
        row = tl.load(POS + t * K + j).to(tl.int64)
        acc += tl.load(W + t * K + j).to(tl.float32) * tl.load(Y + row * H + col, mask, other=0).to(tl.float32)
    tl.store(OUT + t * H + col, acc.to(tl.bfloat16), mask)


def _swiglu(gu, width):
    rows = gu.shape[0]
    hidden_q = torch.empty((rows, width), device=gu.device, dtype=torch.float8_e4m3fn)
    hidden_scale = torch.empty(rows, device=gu.device, dtype=torch.float32)
    dense_ffn._swiglu_quant[(rows,)](gu, hidden_q, hidden_scale, width, triton.next_power_of_2(width), num_warps=4)
    return hidden_q, hidden_scale


def _per_expert_mm(a, scale_a, weight, scale_b, bounds, columns):
    """Rows bounds[e]:bounds[e+1] of a times weight[e]^T, one cuBLAS FP8 GEMM per expert."""
    out = torch.empty((a.shape[0], columns), device=a.device, dtype=torch.bfloat16)
    for e in range(len(bounds) - 1):
        begin, end = bounds[e], bounds[e + 1]
        if begin < end:
            torch._scaled_mm(a[begin:end], weight[e].t(), scale_a=scale_a[begin:end, None], scale_b=scale_b[e][None, :],
                             out_dtype=torch.bfloat16, use_fast_accum=True, out=out[begin:end])
    return out


@torch.library.custom_op('q38_moe::routed_per_expert', mutates_args=())
def routed_per_expert(a: torch.Tensor, scale_a: torch.Tensor, gate_up: torch.Tensor, gate_up_scale: torch.Tensor,
                      down: torch.Tensor, down_scale: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Expert-ordered FP8 rows through their experts' SwiGLU FFNs with one cuBLAS GEMM per expert and projection.

    Opaque to torch.compile, because the host reads the expert row offsets (one synchronization per block).
    """
    bounds = [0] + offsets.tolist()
    gu = _per_expert_mm(a, scale_a, gate_up, gate_up_scale, bounds, 2 * down.shape[-1])
    hidden_q, hidden_scale = _swiglu(gu, down.shape[-1])
    del gu
    return _per_expert_mm(hidden_q, hidden_scale, down, down_scale, bounds, down.shape[1])


@routed_per_expert.register_fake
def _(a, scale_a, gate_up, gate_up_scale, down, down_scale, offsets):
    return a.new_empty((a.shape[0], down.shape[1]), dtype=torch.bfloat16)


def _routed_grouped(self, a, scale_a, offsets):
    gu = torch._scaled_grouped_mm(a, self._grouped_gate_up, scale_a, self._grouped_gate_up_scale,
                                  offs=offsets, out_dtype=torch.bfloat16, use_fast_accum=True)
    hidden_q, hidden_scale = _swiglu(gu, self._moe_width)
    del gu
    return torch._scaled_grouped_mm(hidden_q, self._grouped_down, hidden_scale, self._grouped_down_scale,
                                    offs=offsets, out_dtype=torch.bfloat16, use_fast_accum=True)


def _forward(self, x):
    if self.training or torch.is_grad_enabled():
        raise RuntimeError('FP8 MoE FFN is inference-only')
    shape = x.shape
    hidden = shape[-1]
    q, scale = quant_rows(x.reshape(-1, hidden))
    tokens = q.shape[0]
    experts, topk = self._moe_experts, self._moe_topk
    logits = torch._scaled_mm(q, self._router_q.t(), scale_a=scale[:, None], scale_b=self._router_s[None, :],
                              out_dtype=torch.bfloat16, use_fast_accum=True)
    weights = torch.empty((tokens, topk), device=q.device, dtype=torch.bfloat16)
    index = torch.empty((tokens, topk), device=q.device, dtype=torch.int64)
    if self._paper_loss_free:
        # Preserve the trained router: bias affects selection only, while
        # original sigmoid scores determine the normalized expert weights.
        probabilities = logits[:, :experts].float().sigmoid()
        index = (probabilities + self._paper_routing_bias).topk(topk, dim=-1).indices
        weights = probabilities.gather(-1, index)
        weights = (weights / weights.sum(-1, keepdim=True).clamp_min(1e-20)).to(torch.bfloat16)
    else:
        _route[(triton.cdiv(tokens, 64),)](logits, weights, index, tokens, logits.stride(0), experts,
                                       triton.next_power_of_2(experts), topk, 64, num_warps=4)
    shared = _shared(self, q, scale, tokens)
    flat = index.reshape(-1)
    order = torch.argsort(flat, stable=True)
    counts = torch.zeros(experts, device=q.device, dtype=torch.int32).scatter_add_(
        0, flat, torch.ones_like(flat, dtype=torch.int32))
    offsets = torch.cumsum(counts, 0, dtype=torch.int32)
    position = torch.empty_like(order)
    position[order] = torch.arange(order.numel(), device=q.device)
    source = order // topk
    if self._moe_routed == 'grouped':
        y = _routed_grouped(self, q[source], scale[source], offsets)
    else:
        y = routed_per_expert(q[source], scale[source], self.experts._q8_gate, self._grouped_gate_up_scale,
                              self.experts._q8_down, self._grouped_down_scale, offsets)
    out = torch.empty((tokens, hidden), device=q.device, dtype=torch.bfloat16)
    block = 512
    _combine[(tokens, triton.cdiv(hidden, block))](y, position, weights, shared, logits, out, logits.stride(0),
                                                    experts, hidden, topk, block, num_warps=4)
    return out.reshape(*shape[:-1], hidden)


def install(model):
    routed = os.environ.get('Q38_MOE_ROUTED', 'per_expert')
    if routed not in ('per_expert', 'grouped'):
        raise ValueError(f'Q38_MOE_ROUTED must be per_expert or grouped, not {routed!r}')
    count = 0
    with torch.no_grad():
        for module in model.modules():
            if type(module).__name__ != 'Qwen3_8MoeSparseMoeBlock':
                continue
            experts = module.experts
            for source, name in (('gate_up_proj', 'gate'), ('down_proj', 'down')):
                q, scale = plain_quant_rows(getattr(experts, source).detach())
                experts.register_buffer('_q8_' + name, q, persistent=False)
                experts.register_buffer('_q8_' + name + '_scale', scale, persistent=False)
                parameter = getattr(experts, source)
                parameter.data = parameter.detach().cpu()
            router = torch.cat([module.gate.weight.detach(), module.shared_expert_gate.weight.detach()], 0)
            router = torch.nn.functional.pad(router, (0, 0, 0, (-router.shape[0]) % 16))
            q, scale = plain_quant_rows(router.to(torch.bfloat16))
            module.register_buffer('_router_q', q, persistent=False)
            module.register_buffer('_router_s', scale, persistent=False)
            shared = module.shared_expert
            g, sg = dense_ffn._weight_cache(shared.gate_proj)
            u, su = dense_ffn._weight_cache(shared.up_proj)
            d, sd = dense_ffn._weight_cache(shared.down_proj)
            module.register_buffer('_shared_q_gate_up', torch.cat((g, u), 0), persistent=False)
            module.register_buffer('_shared_s_gate_up', torch.cat((sg, su)), persistent=False)
            module.register_buffer('_shared_q_down', d, persistent=False)
            module.register_buffer('_shared_s_down', sd, persistent=False)
            module._shared_width = shared.gate_proj.out_features
            module._moe_experts, module._moe_topk = experts.num_experts, module.gate.top_k
            module._paper_loss_free = hasattr(module.gate, 'routing_bias')
            if module._paper_loss_free:
                module.register_buffer('_paper_routing_bias', module.gate.routing_bias.detach().float(), persistent=False)
            module._moe_width = experts.intermediate_dim
            module._moe_routed = routed
            # Row-quantized [E, N, K] weights; the grouped GEMM takes their column-major [E, K, N] views.
            module._grouped_gate_up = experts._q8_gate.transpose(-2, -1)
            module._grouped_gate_up_scale = experts._q8_gate_scale.view(experts.num_experts, -1)
            module._grouped_down = experts._q8_down.transpose(-2, -1)
            module._grouped_down_scale = experts._q8_down_scale.view(experts.num_experts, -1)
            module.forward = types.MethodType(_forward, module)
            count += 1
    if not count:
        raise ValueError('No MoE blocks found')
    return dict(modules=count, routed={'per_expert': 'per_expert_scaled_mm_with_gather_and_combine',
                                       'grouped': 'scaled_grouped_mm_with_gather_and_combine'}[routed],
                shared='packed_fp8_dense_swiglu', router_and_shared_gate='one_fp8_gemm', routing='triton_topk')
