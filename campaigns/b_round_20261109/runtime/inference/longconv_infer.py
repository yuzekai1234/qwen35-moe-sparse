"""Inference path of the FFT long-convolution blocks (modeling LongConv) for the prequantized trunk.

Each block's frequency response W = rfft(T, n) of its lag filter T [K, c, c] is computed once at installation in FP32
and kept in FP16 in the real form of complex multiplication on interleaved (re, im) pairs, [n/2+1, 2c, 2c]; the taps
then move to the CPU. n and the segmentation follow modeling fft_segments: with K < L dividing L, L/K overlap-save
segments with n = 2K, else one zero-padded FFT.

Per call, with rows = batch * segments, the transforms run along the contiguous last dimension (FP16 torch.fft is
three to four times faster there than along dim 0 on H100) and Triton kernels do the layout changes:
1. the u half of the in_proj output (its weight scales carry a factor 1/n, which keeps the FFT sums in FP16 range
   and cancels the unnormalized inverse) is transposed into the segment buffer x [rows, c, n] (time innermost);
2. torch.fft.rfft gives the spectrum [rows, c, n/2+1], transposed to [n/2+1, rows, c], whose interleaved real view
   is the operand of one batched FP16 GEMM per frequency with W;
3. the product is transposed back to [rows, c, n/2+1] and torch.fft.irfft (unnormalized, norm="forward") gives
   the convolution [rows, c, n];
4. one Triton kernel reads it in tiles of tokens x channels together with the gate, forms z * silu(g) and quantizes
   each token's row to FP8 for out_proj.
The convolution is a custom op so that the compiled trunk treats it as one extern kernel.
"""
import os

import torch
import triton
import triton.language as tl


def fft_segments(length, taps):
    if taps < length and length % taps == 0:
        return 2 * taps, length // taps
    return 1 << (length + taps - 2).bit_length(), 1


@triton.jit
def _segments(UG, X, L: tl.constexpr, N: tl.constexpr, SEG: tl.constexpr, OFFSET: tl.constexpr,
              SEGMENTS: tl.constexpr, C: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_C: tl.constexpr):
    """Transpose BLOCK_T tokens x BLOCK_C channels of the u half of the in_proj output [tokens, 2C] into the segment
    buffer X [rows, C, N]: token t of sequence b is position OFFSET + t % SEG of row b * SEGMENTS + t // SEG and,
    with overlap-save segments, also position t % SEG of the next segment's row."""
    token0 = tl.program_id(0) * BLOCK_T
    c0 = tl.program_id(1) * BLOCK_C
    b, t0 = token0 // L, token0 % L
    s, j0 = t0 // SEG, t0 % SEG
    j = j0 + tl.arange(0, BLOCK_T)
    k = c0 + tl.arange(0, BLOCK_C)
    mask = (j < SEG)[:, None] & (k < C)[None, :]
    u = tl.load(UG + (b * L + s * SEG + j).to(tl.int64)[:, None] * (2 * C) + k[None, :], mask, other=0).to(tl.float16)
    row = (b * SEGMENTS + s).to(tl.int64)
    tl.store(X + (row * C + k[None, :]) * N + OFFSET + j[:, None], u, mask)
    if SEGMENTS > 1:
        tl.store(X + ((row + 1) * C + k[None, :]) * N + j[:, None], u, mask & (s + 1 < SEGMENTS))


@triton.jit
def _transpose(X, Y, M, F, BLOCK_M: tl.constexpr, BLOCK_F: tl.constexpr):
    """Y [F, M] = X [M, F]^T for 4-byte elements."""
    m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    f = tl.program_id(1) * BLOCK_F + tl.arange(0, BLOCK_F)
    mask = (m < M)[:, None] & (f < F)[None, :]
    x = tl.load(X + m.to(tl.int64)[:, None] * F + f[None, :], mask)
    tl.store(Y + f.to(tl.int64)[None, :] * M + m[:, None], x, mask)


def _transposed(x, rows, cols):
    """[cols, rows] copy of a contiguous complex32 or interleaved FP16 tensor read as a [rows, cols] matrix of
    complex values."""
    source = x.view(torch.int32).view(rows, cols)
    target = torch.empty((cols, rows), device=x.device, dtype=torch.int32)
    _transpose[(triton.cdiv(rows, 64), triton.cdiv(cols, 64))](source, target, rows, cols, BLOCK_M=64, BLOCK_F=64,
                                                                num_warps=4)
    return target


@triton.jit
def _gate_quant(Y, G, Q, S, L: tl.constexpr, N: tl.constexpr, SEG: tl.constexpr, OFFSET: tl.constexpr,
                SEGMENTS: tl.constexpr, C: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_C: tl.constexpr):
    """Rows h = y * silu(g) of BLOCK_T consecutive tokens of one segment, quantized to FP8 with per-row scales (as
    low_precision._quant_rows). y is the convolution laid out [rows, C, N] (time innermost): token t of sequence b
    is position OFFSET + t % SEG of row b * SEGMENTS + t // SEG; g is the gate half of the in_proj output
    [tokens, 2C]. The first pass over the channels finds each row's maximum, the second writes the FP8 values."""
    row = tl.program_id(0).to(tl.int64)
    j = tl.program_id(1) * BLOCK_T + tl.arange(0, BLOCK_T)
    tokens = (row // SEGMENTS) * L + (row % SEGMENTS) * SEG + j
    rows_ok = j < SEG
    k = tl.arange(0, BLOCK_C)
    amax = tl.zeros((BLOCK_T,), tl.float32)
    for c0 in range(0, C, BLOCK_C):
        mask = rows_ok[:, None] & (c0 + k < C)[None, :]
        y = tl.load(Y + row * (C * N) + (c0 + k)[None, :] * N + (OFFSET + j)[:, None], mask, other=0).to(tl.float32)
        g = tl.load(G + tokens[:, None] * (2 * C) + C + (c0 + k)[None, :], mask, other=0).to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.abs(y * g * tl.sigmoid(g)), 1))
    scale = tl.maximum(amax / 448., 1.e-12)
    for c0 in range(0, C, BLOCK_C):
        mask = rows_ok[:, None] & (c0 + k < C)[None, :]
        y = tl.load(Y + row * (C * N) + (c0 + k)[None, :] * N + (OFFSET + j)[:, None], mask, other=0).to(tl.float32)
        g = tl.load(G + tokens[:, None] * (2 * C) + C + (c0 + k)[None, :], mask, other=0).to(tl.float32)
        h = y * g * tl.sigmoid(g) / scale[:, None]
        tl.store(Q + tokens[:, None] * C + (c0 + k)[None, :], tl.minimum(tl.maximum(h, -448.), 448.), mask)
    tl.store(S + tokens, scale, rows_ok)


# FP16 cuFFT calls fail with CUFFT_NOT_SUPPORTED above 2^31 input elements, so larger batches are processed in
# groups of sequences of balanced size.
FFT_ELEMENTS = 2 ** 31 - 1


@torch.library.custom_op('q38lc::fft_conv_gate', mutates_args=())
def fft_conv_gate(ug: torch.Tensor, weight: torch.Tensor, taps: int) -> tuple[torch.Tensor, torch.Tensor]:
    """ug [B, L, 2c] BF16 in_proj output (u half pre-scaled by 1/n) -> FP8 rows of (u * T) * silu(g) [B*L, c] and
    their scales [B*L]."""
    batch, length, width = ug.shape
    channels = width // 2
    n, segments = fft_segments(length, taps)
    assert ug.is_contiguous() and ug.dtype == torch.bfloat16
    assert weight.shape == (n // 2 + 1, 2 * channels, 2 * channels) and weight.dtype == torch.float16
    assert length % 64 == 0 and taps % 64 == 0, 'token tiles of 64 must not cross segments'
    q = torch.empty((batch * length, channels), device=ug.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty(batch * length, device=ug.device, dtype=torch.float32)
    largest = max(1, FFT_ELEMENTS // (segments * channels * n))
    group = -(-batch // -(-batch // largest))
    for start in range(0, batch, group):
        stop = min(batch, start + group)
        _conv_gate(ug[start:stop], weight, taps, q[start * length:stop * length], scales[start * length:stop * length])
    return q, scales


def _conv_gate(ug, weight, taps, q, scales):
    """fft_conv_gate on a group of sequences, writing into q and scales (row slices of the outputs)."""
    batch, length, width = ug.shape
    channels = width // 2
    n, segments = fft_segments(length, taps)
    f, rows = n // 2 + 1, batch * segments
    x = torch.empty((rows, channels, n), device=ug.device, dtype=torch.float16)
    if segments == 1:
        x[:, :, length:].zero_()
        offset, seg = 0, length
    else:
        x.view(batch, segments, channels, n)[:, 0, :, :taps].zero_()
        offset, seg = taps, taps
    _segments[(batch * length // 64, triton.cdiv(channels, 64))](ug, x, length, n, seg, offset, segments, channels,
                                                                 BLOCK_T=64, BLOCK_C=64, num_warps=4)
    spectrum = torch.fft.rfft(x, dim=-1)                                                  # [rows, c, f]
    del x
    by_frequency = _transposed(torch.view_as_real(spectrum), rows * channels, f)          # [f, rows * c]
    del spectrum
    products = torch.bmm(by_frequency.view(torch.float16).view(f, rows, 2 * channels), weight)   # [f, rows, 2c]
    del by_frequency
    by_row = _transposed(products, f, rows * channels)                                    # [rows * c, f]
    del products
    y = torch.fft.irfft(torch.view_as_complex(by_row.view(torch.float16).view(rows, channels, f, 2)), n=n, dim=-1,
                        norm='forward')                                                   # [rows, c, n]
    del by_row
    _gate_quant[(rows, triton.cdiv(seg, 64))](y, ug, q, scales, length, n, seg, offset, segments, channels,
                                             BLOCK_T=64, BLOCK_C=64, num_warps=4)


@fft_conv_gate.register_fake
def _(ug, weight, taps):
    batch, length, width = ug.shape
    return (ug.new_empty((batch * length, width // 2), dtype=torch.float8_e4m3fn),
            ug.new_empty(batch * length, dtype=torch.float32))


def frequency_weight(module, length, column_chunk=64):
    """[n/2+1, 2c, 2c] FP16 real form of the block's frequency response for sequences of `length` tokens:
    rows (2i, 2i+1) and columns (2o, 2o+1) hold [[Re W_io, Im W_io], [-Im W_io, Re W_io]]."""
    taps = module.taps.detach()
    K, c, _ = taps.shape
    n, _ = fft_segments(length, K)
    inverse = 1 / torch.arange(1, K + 1, device=taps.device, dtype=torch.float32)
    scale = (inverse / inverse.sum()).sqrt().view(-1, 1, 1)
    weight = torch.empty((n // 2 + 1, c, 2, c, 2), device=taps.device, dtype=torch.float16)
    for start in range(0, c, column_chunk):
        stop = min(start + column_chunk, c)
        W = torch.fft.rfft(taps[:, :, start:stop].float() * scale, n=n, dim=0)
        weight[:, :, 0, start:stop, 0] = W.real
        weight[:, :, 0, start:stop, 1] = W.imag
        weight[:, :, 1, start:stop, 0] = -W.imag
        weight[:, :, 1, start:stop, 1] = W.real
        del W
    return weight.view(n // 2 + 1, 2 * c, 2 * c)


def block_forward(module, residual):
    if residual.shape[1] != module._fft_length:
        raise ValueError('LongConv frequency response was precomputed for another sequence length')
    x = module.norm(residual.to(torch.bfloat16))
    q, scales = torch.ops.q38lc.fft_conv_gate(module.in_proj(x), module._fft_weight, module._fft_taps)
    out = module.out_proj
    y = torch._scaled_mm(q, out._q8_weight.t(), scale_a=scales.view(-1, 1), scale_b=out._q8_weight_scale.view(1, -1),
                         bias=out._q8_bias, out_dtype=torch.bfloat16,
                         use_fast_accum=os.environ.get('Q38_FP8_FAST_ACCUM') == '1')
    return y[:, :out.out_features].view(*residual.shape[:-1], out.out_features)


def install(model, length):
    blocks = getattr(model.model, 'longconv', None)
    if not blocks:
        return dict(blocks=0)
    if os.environ.get('Q38_PREQUANT_NORM') != '1':
        raise ValueError('LongConv inference is implemented only in the prequantized trunk (Q38_PREQUANT_NORM=1)')
    with torch.no_grad():
        for module in blocks.values():
            if not (hasattr(module.in_proj, '_q8_weight_scale') and hasattr(module.out_proj, '_q8_weight')):
                raise ValueError('LongConv inference expects FP8 in_proj and out_proj (fp8_linear, general scope)')
            K, c = module.taps.shape[0], module.taps.shape[1]
            n, _ = fft_segments(length, K)
            module.register_buffer('_fft_weight', frequency_weight(module, length), persistent=False)
            module.in_proj._q8_weight_scale[:c] *= 1.0 / n
            module._fft_taps, module._fft_length = K, length
            module.taps.data = module.taps.detach().cpu()
    torch.cuda.empty_cache()
    first = next(iter(blocks.values()))
    n, segments = fft_segments(length, first._fft_taps)
    return dict(blocks=len(blocks), layers=sorted(int(i) for i in blocks), channels=first._fft_weight.shape[1] // 2,
                taps=first._fft_taps, fft_length=n, segments=segments, sequence_length=length,
                fft='FP16 torch.fft along the contiguous time dimension, unnormalized inverse; Triton layout transposes',
                products='FP16 interleaved real-form batched GEMM per frequency',
                epilogue='Triton z * silu(g) with FP8 row quantization for out_proj, read from the inverse FFT layout',
                projections='FP8 linear; 1/n folded into the in_proj u-half weight scales')
