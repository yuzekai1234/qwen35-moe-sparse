"""Time the fused GDN kernels on one 256-sequence C4 chunk for several launch configs."""
import itertools
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(sys.argv[1]).resolve()))
from gdn_fused import conv_silu_l2norm, gated_norm_quant  # noqa: E402


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
    return start.elapsed_time(end) / iters


batch, length, hk, hv = 256, 4096, 14, 42
rows, width = batch * length, (2 * hk + hv) * 64
merged = torch.randn(rows, 7264, device='cuda').to(torch.bfloat16)
conv_weight = torch.randn(width, 4, device='cuda').to(torch.bfloat16)
for block, warps in itertools.product([32, 64, 128], [4, 8]):
    os.environ['Q38_GDN_CONV_BT'], os.environ['Q38_GDN_CONV_WARPS'] = str(block), str(warps)
    ms = timed(lambda: conv_silu_l2norm(merged[:, :width], conv_weight, batch, length, hk, hv))
    print(f'conv BT={block} warps={warps}: {ms:.3f} ms, {rows * width * 4 / ms / 1e9:.2f} TB/s(r+w)', flush=True)
output = torch.randn(rows, hv * 64, device='cuda').to(torch.bfloat16)
norm_weight = torch.randn(64, device='cuda').to(torch.bfloat16)
for warps in [2, 4, 8, 16]:
    os.environ['Q38_GDN_NORM_WARPS'] = str(warps)
    ms = timed(lambda: gated_norm_quant(output, merged[:, 4480:7168], norm_weight, 1e-6))
    print(f'norm warps={warps}: {ms:.3f} ms, {rows * hv * 64 * 5 / ms / 1e9:.2f} TB/s(r+w)', flush=True)
