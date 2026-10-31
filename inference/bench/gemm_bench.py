"""Throughput of the C4 FP8 GEMM shapes under row-wise and per-tensor scaling.

M is the token count of one 256-sequence trunk chunk. Row-wise scaling is what the
evaluation uses (ATen's CUTLASS path); per-tensor scaling uses cuBLASLt and shows
how much a different kernel could gain on the same shape.
"""
import json
import sys

import torch

SHAPES = {  # name: (K, N)
    'gdn_in_merged': (1152, 7264), 'gdn_in_qkv': (1152, 4480), 'gdn_in_z': (1152, 2688),
    'gdn_out': (2688, 1152), 'ffn_gate_up': (1152, 3456), 'ffn_down': (1728, 1152),
    'attn_q': (1152, 5632), 'attn_o': (2816, 1152), 'lm_head': (1152, 10112),
}


def bench(fn, iters=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters / 1000.0


def main():
    m = int(sys.argv[1]) if len(sys.argv) > 1 else 1048576
    torch.manual_seed(0)
    results = {}
    for name, (k, n) in SHAPES.items():
        a = torch.randn(m, k, device='cuda').to(torch.float8_e4m3fn)
        b = torch.randn(n, k, device='cuda').to(torch.float8_e4m3fn)
        sa = torch.rand(m, 1, device='cuda') + 0.5
        sb = torch.rand(1, n, device='cuda') + 0.5
        one = torch.ones((), device='cuda')
        flops = 2 * m * k * n
        row = bench(lambda: torch._scaled_mm(a, b.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16, use_fast_accum=True))
        tensor = bench(lambda: torch._scaled_mm(a, b.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16, use_fast_accum=True))
        results[name] = dict(K=k, N=n, rowwise_ms=row * 1e3, rowwise_tflops=flops / row / 1e12,
                             tensorwise_ms=tensor * 1e3, tensorwise_tflops=flops / tensor / 1e12)
        print(name, json.dumps({key: round(value, 1) if isinstance(value, float) else value
                                for key, value in results[name].items()}), flush=True)
        del a, b
    print(json.dumps(dict(M=m, results=results)))


if __name__ == '__main__':
    main()
