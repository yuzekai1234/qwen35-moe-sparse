"""FP8 GEMM outputs whose row-wise scales are applied by the consuming kernel.

ATen's row-wise scaled GEMM is 25-45% slower than cuBLASLt's tensor-wise GEMM on the
C4 shapes with N=1152 (GDN out_proj, FFN down_proj, attention o_proj). With
Q38_DEFERRED_SCALE=1 these GEMMs run tensor-wise with unit scales, and the residual
producer that reads their output applies the activation row scale and weight row scale:

    value = bf16(bf16(A_fp8 @ W_fp8^T) * row_scale[:, None] * col_scale[None, :])

The only numerical difference from the row-wise GEMM is the extra BF16 rounding of the
unscaled product.
"""
import os

import torch


def enabled():
    return os.environ.get('Q38_DEFERRED_SCALE', '0') == '1'


class ScaledPending:
    """raw: [rows, N] BF16 unscaled product; row_scale: [rows]; col_scale: [N]."""

    def __init__(self, raw, row_scale, col_scale, shape):
        self.raw = raw
        self.row_scale = row_scale
        self.col_scale = col_scale
        self.shape = tuple(shape)

    def materialize(self):
        value = self.raw.float() * self.row_scale[:, None] * self.col_scale[None, :]
        return value.to(torch.bfloat16).reshape(self.shape)


def gemm(q, scale, weight, weight_scale, shape):
    """Tensor-wise FP8 GEMM of row-quantized q against row-quantized weight."""
    assert weight.shape[0] == shape[-1], 'deferred scaling requires an unpadded weight'
    one = torch.ones((), device=q.device, dtype=torch.float32)
    raw = torch._scaled_mm(q, weight.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16,
                           use_fast_accum=True)
    return ScaledPending(raw, scale, weight_scale, shape)


def materialize(value):
    return value.materialize() if isinstance(value, ScaledPending) else value
