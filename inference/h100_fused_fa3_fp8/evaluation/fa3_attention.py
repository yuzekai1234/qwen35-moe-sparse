"""Qualified C4 FA3 FP8 callback; quantization remains inside the compiled trunk."""
import hashlib
import json
import os
from pathlib import Path
import sys

import torch


def _load_and_verify():
    manifest = json.loads(Path(__file__).with_name('fa3-dependency.json').read_text())
    default = Path(__file__).resolve().parents[1] / 'external/fa3/hopper'
    directory = Path(os.environ.get('Q38_FA3_HOPPER_DIR', str(default))).resolve()
    for name, expected in manifest['files'].items():
        path = directory / name
        if not path.is_file():
            raise RuntimeError('Missing qualified FA3 artifact: ' + str(path) +
                               '; see h100_fused_fa3_fp8/README.md')
        with path.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != expected:
            raise RuntimeError('Qualified FA3 artifact hash mismatch: ' + str(path))
    for name in ('flash_attn_interface', 'flash_attn_3'):
        loaded = sys.modules.get(name)
        if loaded is not None and not Path(loaded.__file__).resolve().is_relative_to(directory):
            raise RuntimeError('Another FA3 installation is already imported: ' + name)
    sys.path.insert(0, str(directory))
    from flash_attn_interface import flash_attn_func
    evidence = dict(source_commit=manifest['FA3_commit'],
                    cutlass_commit=manifest['CUTLASS_commit'],
                    artifact_directory=str(directory),
                    files_sha256=manifest['files'], native_dtype='E4M3',
                    output_dtype='BF16', causal=True, pack_gqa=True,
                    quantization='dynamic per batch / KV group in FX',
                    qualified_scope='C4 Q44/KV2/D64 on H100; other tiers not qualified')
    return flash_attn_func, evidence


flash_attn_func, ARTIFACT_EVIDENCE = _load_and_verify()


@torch.library.custom_op('q38fp8::fa3_fused_baseline_native', mutates_args=())
def native(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
           qs: torch.Tensor, ks: torch.Tensor, vs: torch.Tensor,
           scale: float) -> torch.Tensor:
    assert q.dtype == k.dtype == v.dtype == torch.float8_e4m3fn
    assert q.shape[-2:] == (44, 64) and k.shape[-2:] == v.shape[-2:] == (2, 64)
    out = flash_attn_func(q, k, v, q_descale=qs, k_descale=ks, v_descale=vs,
                         causal=True, softmax_scale=scale, num_splits=1, pack_gqa=True)
    assert out.dtype == torch.bfloat16
    return out


@native.register_fake
def _(q, k, v, qs, ks, vs, scale):
    return torch.empty(q.shape, device=q.device, dtype=torch.bfloat16)


def quant(x):
    b, s, h, d = x.shape
    g = x.reshape(b, s, 2, h // 2, d)
    descale = (g.float().abs().amax(dim=(1, 3, 4)) / 448).clamp_min(1e-12)
    return (g / descale[:, None, :, None, None]).to(torch.float8_e4m3fn).reshape(b, s, h, d), descale


def attention(q, k, v, scale):
    q8, qs = quant(q.transpose(1, 2).contiguous())
    k8, ks = quant(k.transpose(1, 2).contiguous())
    v8, vs = quant(v.transpose(1, 2).contiguous())
    return native(q8, k8, v8, qs, ks, vs, scale).transpose(1, 2)


def install():
    import full_compile
    full_compile.cudnn_attention = attention


def evidence():
    return ARTIFACT_EVIDENCE
