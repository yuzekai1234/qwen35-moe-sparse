"""Fused Gated DeltaNet prefill for the dense C4 path (inference only).

Three changes against dev_optimizations._piece, each behind its own switch:

- Q38_GDN_MERGED_PROJ=1: in_proj_qkv, in_proj_z, in_proj_b and in_proj_a share one
  FP8 GEMM. Their row-wise FP8 weights and scales are concatenated, so every output
  column keeps its own weight row and scale.
- Q38_GDN_FUSED_CONV=1: causal conv1d + SiLU + Q/K L2 normalization + Q/K/V layout in
  one kernel, replacing FLA's conv kernel followed by gather_norm_qkv.
- Q38_GDN_FUSED_NORM=1: gated RMSNorm + row-wise FP8 quantization of the out_proj
  input in one kernel, replacing two Inductor kernels and quant_rows.

The kernels repeat the rounding steps of the replaced code: FP32 conv accumulation
with fused multiply-add per tap, a BF16 conv output, FP32 normalization with BF16
intermediates, and dynamic E4M3 row scaling.
"""
import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


def _enabled(name):
    return os.environ.get(name, '0') == '1'


@triton.jit
def _conv_silu_l2norm(X, W, OQ, OK, OV, T, stride_xb, stride_xt,
                      HK: tl.constexpr, HV: tl.constexpr, KW: tl.constexpr,
                      BT: tl.constexpr, EPS: tl.constexpr):
    """One program per (64-channel head, BT tokens, sequence)."""
    i_h = tl.program_id(0)
    i_t = tl.program_id(1)
    i_b = tl.program_id(2).to(tl.int64)
    d = tl.arange(0, 64)
    channel = i_h * 64 + d
    t = i_t * BT + tl.arange(0, BT)
    acc = tl.zeros((BT, 64), tl.float32)
    for i_w in tl.static_range(KW):
        source = t - (KW - 1) + i_w
        valid = (source >= 0) & (source < T)
        x = tl.load(X + i_b * stride_xb + source[:, None].to(tl.int64) * stride_xt + channel[None, :],
                    mask=valid[:, None], other=0.).to(tl.float32)
        w = tl.load(W + channel * KW + i_w).to(tl.float32)
        acc = tl.fma(x, w[None, :], acc)
    y = (acc * tl.sigmoid(acc)).to(tl.bfloat16, fp_downcast_rounding='rtne')
    rows = (i_b * T + t).to(tl.int64)
    store_mask = (t < T)[:, None]
    if i_h < 2 * HK:
        yf = y.to(tl.float32)
        inverse = 1. / tl.sqrt(tl.sum(yf * yf, 1) + EPS)
        normalized = yf * inverse[:, None]
        if i_h < HK:
            tl.store(OQ + (rows[:, None] * HK + i_h) * 64 + d[None, :], normalized, store_mask)
        else:
            tl.store(OK + (rows[:, None] * HK + (i_h - HK)) * 64 + d[None, :], normalized, store_mask)
    else:
        tl.store(OV + (rows[:, None] * HV + (i_h - 2 * HK)) * 64 + d[None, :], y, store_mask)


def conv_silu_l2norm(projected, weight, batch, length, key_heads, value_heads):
    """projected: [batch*length, 2*key_dim+value_dim] BF16 rows (row stride may exceed width)."""
    assert projected.dtype == torch.bfloat16 and projected.stride(-1) == 1
    channels = (2 * key_heads + value_heads) * 64
    assert projected.shape == (batch * length, channels) and weight.shape == (channels, 4)
    weight = weight.contiguous()
    oq = torch.empty((batch, length, key_heads, 64), device=projected.device, dtype=torch.bfloat16)
    ok = torch.empty_like(oq)
    ov = torch.empty((batch, length, value_heads, 64), device=projected.device, dtype=torch.bfloat16)
    row_stride = projected.stride(0)
    block = int(os.environ.get('Q38_GDN_CONV_BT', '64'))
    _conv_silu_l2norm[(2 * key_heads + value_heads, triton.cdiv(length, block), batch)](
        projected, weight, oq, ok, ov, length, length * row_stride, row_stride,
        key_heads, value_heads, 4, block, 1.e-6, num_warps=int(os.environ.get('Q38_GDN_CONV_WARPS', '4')),
        enable_fp_fusion=False)
    return oq, ok, ov


@triton.jit
def _gated_norm_quant(O, Z, W, Q, S, stride_z, H: tl.constexpr, HP: tl.constexpr, EPS: tl.constexpr):
    """One program per token row of H heads x 64."""
    row = tl.program_id(0).to(tl.int64)
    h = tl.arange(0, HP)
    d = tl.arange(0, 64)
    mask = (h < H)[:, None]
    offset = h[:, None] * 64 + d[None, :]
    o = tl.load(O + row * (H * 64) + offset, mask=mask, other=0.).to(tl.float32)
    z = tl.load(Z + row * stride_z + offset, mask=mask, other=0.).to(tl.float32)
    w = tl.load(W + d).to(tl.float32)
    rstd = libdevice.rsqrt(tl.sum(o * o, 1) / 64.0 + EPS)
    normalized = (o * rstd[:, None]).to(tl.bfloat16).to(tl.float32)
    weighted = (w[None, :] * normalized).to(tl.bfloat16).to(tl.float32)
    value = (weighted * (z * tl.sigmoid(z))).to(tl.bfloat16).to(tl.float32)
    value = tl.where(mask, value, 0.)
    scale = tl.maximum(tl.max(tl.max(tl.abs(value), 1), 0) / 448., 1.e-12)
    tl.store(Q + row * (H * 64) + offset, tl.minimum(tl.maximum(value / scale, -448.), 448.), mask=mask)
    tl.store(S + row, scale)


def gated_norm_quant(output, gate, weight, eps):
    """output: [rows, H*64] contiguous BF16; gate: [rows, H*64] BF16 with unit inner stride."""
    rows, width = output.shape
    assert output.is_contiguous() and gate.shape == (rows, width) and gate.stride(-1) == 1
    heads = width // 64
    q = torch.empty((rows, width), device=output.device, dtype=torch.float8_e4m3fn)
    scale = torch.empty(rows, device=output.device, dtype=torch.float32)
    _gated_norm_quant[(rows,)](output, gate, weight, q, scale, gate.stride(0), heads,
                               triton.next_power_of_2(heads), eps,
                               num_warps=int(os.environ.get('Q38_GDN_NORM_WARPS', '8')))
    return q, scale


def merge_projections(module):
    """Concatenate the cached row-wise FP8 projections of one GDN module."""
    parts = [module.in_proj_qkv, module.in_proj_z, module.in_proj_b, module.in_proj_a]
    assert all(p._q8_bias is None for p in parts)
    weight = torch.cat([p._q8_weight for p in parts], 0)
    scale = torch.cat([p._q8_weight_scale for p in parts], 0)
    widths = [p._q8_weight.shape[0] for p in parts]
    starts = [sum(widths[:i]) for i in range(len(widths))]
    module.register_buffer('_merged_q8_weight', weight, persistent=False)
    module.register_buffer('_merged_q8_scale', scale, persistent=False)
    module._merged_slices = [(start, start + p.out_features) for start, p in zip(starts, parts)]
    # Keep each projection usable through views of the merged cache, without a second copy.
    for start, rows, p in zip(starts, widths, parts):
        p._q8_weight = weight[start:start + rows]
        p._q8_weight_scale = scale[start:start + rows]
    return weight.shape[0]


def _gemm(q, scale, weight, weight_scale):
    return torch._scaled_mm(q, weight.t(), scale_a=scale[:, None], scale_b=weight_scale[None, :],
                            out_dtype=torch.bfloat16, use_fast_accum=True)


def piece(self, x):
    """Gated DeltaNet forward for one batch piece; x is the prequantized FP8 input."""
    import gdn_flashqla_backend
    from prequant_norm import quant_rows
    batch, length, _ = x.shape
    q, scale = quant_rows(x.reshape(-1, x.shape[-1]))
    if _enabled('Q38_GDN_MERGED_PROJ'):
        merged = _gemm(q, scale, self._merged_q8_weight, self._merged_q8_scale)
        (qkv0, qkv1), (z0, z1), (b0, b1), (a0, a1) = self._merged_slices
        projected, z, beta_raw, a = merged[:, qkv0:qkv1], merged[:, z0:z1], merged[:, b0:b1], merged[:, a0:a1]
    else:
        projected = _gemm(q, scale, self.in_proj_qkv._q8_weight, self.in_proj_qkv._q8_weight_scale)[:, :self.in_proj_qkv.out_features]
        z = _gemm(q, scale, self.in_proj_z._q8_weight, self.in_proj_z._q8_weight_scale)[:, :self.in_proj_z.out_features]
        beta_raw = _gemm(q, scale, self.in_proj_b._q8_weight, self.in_proj_b._q8_weight_scale)[:, :self.in_proj_b.out_features]
        a = _gemm(q, scale, self.in_proj_a._q8_weight, self.in_proj_a._q8_weight_scale)[:, :self.in_proj_a.out_features]
    del q, scale
    beta = beta_raw.sigmoid().reshape(batch, length, -1)
    g = (-self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)).reshape(batch, length, -1)
    if _enabled('Q38_GDN_FUSED_CONV'):
        query, key, value = conv_silu_l2norm(projected, self.conv1d.weight.squeeze(1), batch, length,
                                             self.num_k_heads, self.num_v_heads)
    else:
        from full_compile import causal_conv1d
        from dev_optimizations import gather_norm_qkv
        mixed, _ = causal_conv1d(projected.reshape(batch, length, -1), weight=self.conv1d.weight.squeeze(1),
                                 bias=self.conv1d.bias, activation=self.activation)
        query, key, value = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], -1)
        query, key, value = gather_norm_qkv(query.reshape(batch, length, -1, self.head_k_dim),
                                            key.reshape(batch, length, -1, self.head_k_dim),
                                            value.reshape(batch, length, -1, self.head_v_dim))
    output, _ = gdn_flashqla_backend.run(query, key, value, g=g, beta=beta, use_qk_l2norm_in_kernel=False)
    del query, key, value, g, beta
    out_proj = self.out_proj
    if _enabled('Q38_GDN_FUSED_NORM'):
        hq, hs = gated_norm_quant(output.reshape(batch * length, -1), z, self.norm.weight, self.norm.variance_epsilon)
    else:
        normed = self.norm(output.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
        hq, hs = quant_rows(normed.reshape(batch * length, -1))
    y = _gemm(hq, hs, out_proj._q8_weight, out_proj._q8_weight_scale)[:, :out_proj.out_features]
    return y.reshape(batch, length, -1)


def install(model):
    count = 0
    for module in model.modules():
        if type(module).__name__.endswith('GatedDeltaNet'):
            if _enabled('Q38_GDN_MERGED_PROJ'):
                merge_projections(module)
            count += 1
    return dict(modules=count, merged_projection=_enabled('Q38_GDN_MERGED_PROJ'),
                fused_conv_silu_l2norm=_enabled('Q38_GDN_FUSED_CONV'),
                fused_gated_norm_quant=_enabled('Q38_GDN_FUSED_NORM'))
