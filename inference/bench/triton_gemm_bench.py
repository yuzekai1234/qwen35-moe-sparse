"""Persistent Triton FP8 GEMM with row-wise scales, timed against ATen's CUTLASS path.

Decides whether a Triton GEMM is fast enough on the C4 shapes to host fused
epilogues (residual add, SwiGLU). Output: BF16 = (A_fp8 @ B_fp8^T) * sa[:, None] * sb[None, :].
"""
import itertools
import json
import sys

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _tile(tile_id, num_pid_m, num_pid_n, GROUP_M: tl.constexpr):
    group = GROUP_M * num_pid_n
    first_m = (tile_id // group) * GROUP_M
    size_m = tl.minimum(num_pid_m - first_m, GROUP_M)
    return first_m + (tile_id % group) % size_m, (tile_id % group) // size_m


@triton.jit
def _fp8_gemm(a_desc, b_desc, c_desc, sa_ptr, sb_ptr, M, N, K,
              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr,
              NUM_SMS: tl.constexpr):
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    k_tiles = tl.cdiv(K, BK)
    for tile_id in tl.range(tl.program_id(0), num_pid_m * num_pid_n, NUM_SMS, flatten=True):
        pid_m, pid_n = _tile(tile_id, num_pid_m, num_pid_n, GROUP_M)
        off_m = pid_m * BM
        off_n = pid_n * BN
        acc = tl.zeros((BM, BN), tl.float32)
        for ki in range(k_tiles):
            a = a_desc.load([off_m, ki * BK])
            b = b_desc.load([off_n, ki * BK])
            acc = tl.dot(a, b.T, acc, max_num_imprecise_acc=BK)
        rows = off_m + tl.arange(0, BM)
        cols = off_n + tl.arange(0, BN)
        sa = tl.load(sa_ptr + rows, mask=rows < M, other=0.)
        sb = tl.load(sb_ptr + cols, mask=cols < N, other=0.)
        c_desc.store([off_m, off_n], (acc * sa[:, None] * sb[None, :]).to(tl.bfloat16))


def fp8_gemm(a, b, sa, sb, BM=128, BN=256, BK=128, GROUP_M=8, num_stages=3, num_warps=8):
    m, k = a.shape
    n = b.shape[0]
    c = torch.empty((m, n), device=a.device, dtype=torch.bfloat16)
    sms = torch.cuda.get_device_properties(a.device).multi_processor_count
    a_desc = TensorDescriptor.from_tensor(a, [BM, BK])
    b_desc = TensorDescriptor.from_tensor(b, [BN, BK])
    c_desc = TensorDescriptor.from_tensor(c, [BM, BN])
    tiles = triton.cdiv(m, BM) * triton.cdiv(n, BN)
    _fp8_gemm[(min(sms, tiles),)](a_desc, b_desc, c_desc, sa, sb, m, n, k, BM, BN, BK, GROUP_M, sms,
                                  num_stages=num_stages, num_warps=num_warps)
    return c


def timed(fn, iters=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters / 1e3


SHAPES = {'gdn_in_merged': (1152, 7264), 'gdn_out': (2688, 1152), 'ffn_gate_up': (1152, 3456),
          'ffn_down': (1728, 1152), 'attn_q': (1152, 5632), 'attn_o': (2816, 1152), 'lm_head': (1152, 10112)}
CONFIGS = [dict(BM=bm, BN=bn, BK=bk, GROUP_M=8, num_stages=s, num_warps=w)
           for bm, bn, bk, s, w in [(128, 256, 128, 3, 8), (128, 128, 128, 4, 8), (128, 256, 64, 4, 8),
                                    (256, 128, 128, 3, 8), (128, 128, 128, 3, 4), (64, 256, 128, 4, 4)]]


def main():
    m = int(sys.argv[1]) if len(sys.argv) > 1 else 1048576
    torch.manual_seed(0)
    summary = {}
    for name, (k, n) in SHAPES.items():
        a = torch.randn(m, k, device='cuda').to(torch.float8_e4m3fn)
        b = torch.randn(n, k, device='cuda').to(torch.float8_e4m3fn)
        sa = torch.rand(m, device='cuda') + 0.5
        sb = torch.rand(n, device='cuda') + 0.5
        flops = 2 * m * n * k
        cutlass = timed(lambda: torch._scaled_mm(a, b.t(), scale_a=sa[:, None], scale_b=sb[None, :],
                                                 out_dtype=torch.bfloat16, use_fast_accum=True))
        reference = torch._scaled_mm(a[:4096], b.t(), scale_a=sa[:4096, None], scale_b=sb[None, :],
                                     out_dtype=torch.bfloat16, use_fast_accum=True).float()
        best = None
        for config in CONFIGS:
            if n % config['BN'] and config['BN'] > n:
                continue
            try:
                seconds = timed(lambda: fp8_gemm(a, b, sa, sb, **config))
                error = (fp8_gemm(a[:4096], b, sa[:4096], sb, **config).float() - reference).abs().max().item()
            except Exception as exc:  # resource limits for some tile shapes
                print(name, config, 'failed', type(exc).__name__, flush=True)
                continue
            if best is None or seconds < best[0]:
                best = (seconds, config, error)
        summary[name] = dict(K=k, N=n, cutlass_tflops=flops / cutlass / 1e12,
                             triton_tflops=flops / best[0] / 1e12, triton_config=best[1], max_abs_diff=best[2])
        print(name, json.dumps(summary[name]), flush=True)
        del a, b


if __name__ == '__main__':
    main()
