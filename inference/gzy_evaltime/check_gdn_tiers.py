"""FlashQLA's GDN forward on the head counts of all four tiers.

FlashQLA's TileLang chunk_local_cumsum holds a (heads, 64) FP32 fragment on 128 threads
and finds no register layout when the value-head count is odd (C2: 27, C3: 33);
gdn_flashqla_backend therefore routes odd head counts to FLA's Triton cumsum. For the
key/value head counts of every tier this script checks, on random inputs shaped like the
model's (L2-normalized BF16 q/k, FP32 gate g, BF16 beta):

1. the TileLang cumsum and FLA's Triton cumsum against an FP64 reference;
2. FlashQLA's kkt_solve against an FP64 inverse;
3. gdn_flashqla_backend.run against FLA's chunk_gated_delta_rule, and on even head counts
   its bitwise equality with the unpatched FlashQLA forward;

then times both cumsums at the C4 shape.
"""
import argparse
import sys
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

TIERS = {'c1': (6, 18), 'c2': (9, 27), 'c3': (11, 33), 'c4': (14, 42)}


def timed(fn, iters=20):
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


def inputs(batch, tokens, key_heads, value_heads, generator):
    def randn(*shape):
        return torch.randn(*shape, device='cuda', generator=generator)
    q = F.normalize(randn(batch, tokens, key_heads, 64), dim=-1).to(torch.bfloat16)
    k = F.normalize(randn(batch, tokens, key_heads, 64), dim=-1).to(torch.bfloat16)
    v = randn(batch, tokens, value_heads, 64).to(torch.bfloat16)
    g = -randn(value_heads).exp()[None, None, :] * 0.1 * F.softplus(randn(batch, tokens, value_heads))
    beta = torch.sigmoid(randn(batch, tokens, value_heads) * 2).to(torch.bfloat16)
    return q, k, v, g, beta


def cumsum_reference(g):
    batch, tokens, heads = g.shape
    return g.double().view(batch, tokens // 64, 64, heads).cumsum(2).view(batch, tokens, heads)


def kkt_reference(k, beta):
    """(I + StrictLower(diag(beta) K K^T))^-1 per 64-token chunk, as a[b, t, h, j]."""
    batch, tokens, key_heads, _ = k.shape
    heads = beta.shape[2]
    chunks = tokens // 64
    keys = k.double().repeat_interleave(heads // key_heads, dim=2)
    keys = keys.view(batch, chunks, 64, heads, 64).permute(0, 1, 3, 2, 4)
    scaled = (keys @ keys.transpose(-1, -2)) * beta.double().view(batch, chunks, 64, heads).permute(0, 1, 3, 2)[..., None]
    eye = torch.eye(64, device=k.device, dtype=torch.float64)
    inverse = torch.linalg.inv(eye + torch.tril(scaled, -1))
    return inverse.permute(0, 1, 3, 2, 4).reshape(batch, tokens, heads, 64)


def error(out, ref):
    diff = (out.double() - ref).abs()
    return f'max_abs={float(diff.max()):.2e} mean_abs={float(diff.mean()):.2e} ref_max={float(ref.abs().max()):.2e}'


def attempt(label, fn):
    try:
        return fn()
    except Exception as exc:
        lines = traceback.format_exception_only(type(exc), exc)
        print(f'  {label}: FAILED {lines[-1].strip()[:160]}', flush=True)
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eval-dir', type=Path, required=True)
    parser.add_argument('--tokens', type=int, default=4096)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--time-batch', type=int, default=256)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    sys.path.insert(0, str(args.eval_dir.resolve()))
    import flash_qla.ops.gated_delta_rule.chunk as qla_chunk
    from flash_qla.ops.gated_delta_rule.chunk.hopper import kkt_solve as qla_kkt
    from flash_qla.ops.utils import chunk_local_cumsum as qla_cumsum
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_forward
    from fla.ops.utils.cumsum import chunk_local_cumsum as fla_cumsum_raw
    import gdn_flashqla_backend
    patched_cumsum = qla_chunk.chunk_local_cumsum
    assert patched_cumsum is not qla_cumsum, 'gdn_flashqla_backend did not patch the cumsum'

    def fla_cumsum(g):
        return fla_cumsum_raw(g, 64, output_dtype=g.dtype)

    generator = torch.Generator(device='cuda').manual_seed(0)
    for tier, (key_heads, value_heads) in TIERS.items():
        print(f'{tier}: key_heads={key_heads} value_heads={value_heads} batch={args.batch} tokens={args.tokens}', flush=True)
        q, k, v, g, beta = inputs(args.batch, args.tokens, key_heads, value_heads, generator)
        ref = cumsum_reference(g)
        tilelang_ok = attempt('tilelang cumsum', lambda: qla_cumsum(g=g, chunk_size=64)) is not None
        if tilelang_ok:
            out = qla_cumsum(g=g, chunk_size=64)
            print(f'  tilelang cumsum: {error(out, ref)} equal_to_fla={bool(torch.equal(out, fla_cumsum(g)))}', flush=True)
        print(f'  fla cumsum: {error(fla_cumsum(g), ref)}', flush=True)
        ref = kkt_reference(k, beta)
        out = attempt('flashqla kkt_solve', lambda: qla_kkt(k=k, b=beta, chunk_size=64))
        if out is not None:
            print(f'  flashqla kkt_solve: {error(out, ref)}', flush=True)
        group = value_heads // key_heads
        ref, _ = fla_forward(q.repeat_interleave(group, 2), k.repeat_interleave(group, 2), v, g=g, beta=beta,
                             scale=64 ** -.5, use_qk_l2norm_in_kernel=False)
        ref = ref.double()
        out, _ = gdn_flashqla_backend.run(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=False)
        print(f'  backend run vs fla: {error(out, ref)}', flush=True)
        if tilelang_ok:
            qla_chunk.chunk_local_cumsum = qla_cumsum
            original, _ = gdn_flashqla_backend.run(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=False)
            qla_chunk.chunk_local_cumsum = patched_cumsum
            print(f'  backend run equal to unpatched flashqla: {bool(torch.equal(out, original))}', flush=True)
        del q, k, v, g, beta, ref
        torch.cuda.empty_cache()
    key_heads, value_heads = TIERS['c4']
    _, _, _, g, _ = inputs(args.time_batch, args.tokens, key_heads, value_heads, generator)
    print(f'timing c4 shape batch={args.time_batch} tokens={args.tokens}', flush=True)
    print(f'  tilelang cumsum: {timed(lambda: qla_cumsum(g=g, chunk_size=64)):.3f} ms', flush=True)
    print(f'  fla cumsum: {timed(lambda: fla_cumsum(g)):.3f} ms', flush=True)


if __name__ == '__main__':
    main()
