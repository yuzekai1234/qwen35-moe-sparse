"""Read data metadata, exercise the native parser, and reconcile meta-model counts on CPU."""
import argparse
import json
import os
from pathlib import Path
import sys
from common import HERE, DEFAULT_WORKSPACE, digest, read, source_hash, state_root, write


def check(workspace, run_id, level):
    root = state_root(workspace)/'runs'/run_id
    manifest = read(root/'manifest.json')
    snapshot = Path(manifest['source_directory'])
    if manifest['source_hash'] != source_hash(snapshot):
        raise RuntimeError('Source changed after preparation; prepare another run id before training')
    row = next(r for r in manifest['levels'] if r['level'] == level)
    path = Path(row['config'])
    if digest(path) != row['config_sha256']:
        raise RuntimeError('Config changed after preparation: '+str(path))
    sys.path.insert(0, str(snapshot/'runtime'))
    from q38 import train
    from q38.models.qwen3_8_moe import Qwen3_8MoeForCausalLM
    import torch
    args = train.parse_args(['--config', str(path), '--dry-run'])
    _, config, fc, _, _, _, _, plan = train.prepare_inputs(args, row['world'])
    expected = read(path)['accounting']
    for key in ('train_flops_per_token', 'total_params_all', 'active_params_excluding_embeddings'):
        if fc[key] != expected[key]:
            raise RuntimeError('Count differs from generated plan: '+key)
    with torch.device('meta'):
        model = Qwen3_8MoeForCausalLM(config)
    actual = sum(p.numel() for p in model.parameters())
    if actual != fc['total_params_all']:
        raise RuntimeError(f'Meta-model parameter mismatch: {actual} != {fc["total_params_all"]}')
    proof = dict(passed=True, scope='CPU only; GPU memory, kernel correctness and quality not yet tested',
                 source_hash=source_hash(snapshot), config_sha256=digest(path), actual_meta_parameters=actual,
                 native_plan=plan, cuda_initialized=torch.cuda.is_initialized())
    if proof['cuda_initialized']:
        raise RuntimeError('CPU preflight initialized CUDA unexpectedly')
    write(root/level/'preflight.json', proof)
    return proof


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, default=DEFAULT_WORKSPACE)
    p.add_argument('--run-id', default='combined-v2')
    p.add_argument('--level', choices=['c1', 'c2', 'c3', 'c4', 'all'], default='all')
    a = p.parse_args()
    levels = ['c1', 'c2', 'c3', 'c4'] if a.level == 'all' else [a.level]
    print(json.dumps({level: check(a.workspace, a.run_id, level) for level in levels}, indent=2))


if __name__ == '__main__':
    main()
