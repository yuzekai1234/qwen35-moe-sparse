"""Compare the fused GDN kernels with the kernels they replace, on one C4 GDN layer.

Inputs are synthetic BF16 activations at the C4 widths; weights come from the
checkpoint. Reports bitwise mismatches and the largest absolute difference.
"""
import argparse
import sys
from pathlib import Path

import torch


def compare(name, reference, candidate):
    reference, candidate = reference.float(), candidate.float()
    differ = (reference != candidate)
    print(f'{name}: elements={reference.numel()} mismatched={int(differ.sum())} '
          f'max_abs_diff={float((reference - candidate).abs().max()):.3e}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eval-dir', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--batch', type=int, default=4)
    args = parser.parse_args()
    sys.path.insert(0, str(args.eval_dir.resolve()))
    from entry_defaults import apply_defaults
    apply_defaults()
    from safetensors import safe_open
    from fla.modules.conv.causal_conv1d import causal_conv1d
    from dev_optimizations import gather_norm_qkv
    from low_precision import quant_rows
    from gdn_fused import conv_silu_l2norm, gated_norm_quant

    device = torch.device('cuda')
    weights = {}
    for path in sorted(args.model.glob('*.safetensors')):
        with safe_open(str(path), 'pt') as handle:
            for key in handle.keys():
                if key.startswith('model.layers.0.linear_attn.') and ('conv1d' in key or 'norm' in key):
                    weights[key] = handle.get_tensor(key).to(device)
    conv_weight = weights['model.layers.0.linear_attn.conv1d.weight'].squeeze(1).to(torch.bfloat16)
    norm_weight = weights['model.layers.0.linear_attn.norm.weight'].to(torch.bfloat16)
    torch.manual_seed(0)
    batch, length, hk, hv = args.batch, 4096, 14, 42
    width = (2 * hk + hv) * 64

    # Conv input as a strided view of a wider projection buffer, as in the merged GEMM.
    wide = (torch.randn(batch * length, width + 2784, device=device) * 2.0).to(torch.bfloat16)
    projected = wide[:, :width]
    mixed, _ = causal_conv1d(projected.reshape(batch, length, width).contiguous(), weight=conv_weight,
                             bias=None, activation='silu')
    q, k, v = torch.split(mixed, [hk * 64, hk * 64, hv * 64], -1)
    rq, rk, rv = gather_norm_qkv(q.reshape(batch, length, hk, 64), k.reshape(batch, length, hk, 64),
                                 v.reshape(batch, length, hv, 64))
    cq, ck, cv = conv_silu_l2norm(projected, conv_weight, batch, length, hk, hv)
    compare('conv_q', rq, cq); compare('conv_k', rk, ck); compare('conv_v', rv, cv)

    # Gated RMSNorm + quantization; reference is the module code under the same compile options.
    output = torch.randn(batch * length, hv * 64, device=device).to(torch.bfloat16)
    gate = (torch.randn(batch * length, hv * 64 + 2048, device=device) * 1.5).to(torch.bfloat16)[:, :hv * 64]

    def reference_norm(o, z, weight):
        h = o.reshape(-1, 64).float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)
        h = weight * h.to(torch.bfloat16)
        h = h * torch.nn.functional.silu(z.reshape(-1, 64).float())
        return h.to(torch.bfloat16).reshape(o.shape)

    compiled = torch.compile(reference_norm, fullgraph=True, dynamic=False, options={'emulate_precision_casts': True})
    normed = compiled(output, gate.contiguous(), norm_weight)
    rq8, rscale = quant_rows(normed)
    cq8, cscale = gated_norm_quant(output, gate, norm_weight, 1e-6)
    compare('norm_scale', rscale, cscale)
    compare('norm_q8', rq8, cq8)
    eager = reference_norm(output, gate.contiguous(), norm_weight)
    compare('norm_eager_vs_compiled', eager, normed)


if __name__ == '__main__':
    main()
