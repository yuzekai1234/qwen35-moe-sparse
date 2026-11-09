"""Prepare C2/C3 matching the running LC3/E512/B1.5 C1/C4 family."""
import argparse
import math
import shutil
from pathlib import Path

from common import HERE, digest, read, source_hash, state_root, write
from prepare import BUDGETS, build, counter, hf_config


def make_config(level, workspace, run_id, world=8, micro_bs=None):
    c, _ = build(level, workspace, run_id, experts=512, taps=2048, world=world)
    m, t = c['model'], c['train']
    layers = m['layers']
    batch = 98304 if level == 'c2' else 196608
    m.update(longconv_layers=f'{layers//3},{2*layers//3},{layers-2}',
             longconv_filter_seqs=batch//4096)
    micro_bs = micro_bs or (3 if level == 'c2' and world == 8 else 2)
    t.update(batch_tokens=batch, micro_bs=micro_bs, longconv_offload=0)
    if batch % (world * micro_bs * t['seq']):
        raise ValueError('Global batch must divide world * micro_bs * seq')
    fc = counter()(hf_config(m, t['seq']), t['seq'])
    tokens = (BUDGETS[level] * 995 // 1000) // (fc['train_flops_per_token'] * batch) * batch
    steps = tokens // batch
    data = Path(t['data'])
    available = data.stat().st_size // 4 if data.exists() else 5617222311
    t.update(tokens=tokens, warmup=max(10, round(steps*.01)),
             data_epochs=math.ceil((tokens//4096)/((available-1)//4096)))
    fc.update(steps=steps, train_tokens=tokens,
              actual_train_flops=tokens*fc['train_flops_per_token'], compute_reserve_fraction=.005)
    c['accounting'] = fc
    c['variant']['common_schedule']['start_step'] = round(steps*14719/16096)
    c['variant'].update(scaling_rule='LC3 after token mixer, before MoE: floor(L/3), floor(2L/3), L-2; E512/top4; global batch x1.5',
                        reference_run='combined-lc3-e512-b15-c4-w8-20261109',
                        reference_config_sha256=digest(HERE/'provenance/c4-lc3-e512-b15.json'),
                        status='prepared_unmeasured')
    ref = read(HERE/'provenance/c4-lc3-e512-b15.json')
    for key in ('experts', 'topk', 'longconv_taps', 'paper_engram', 'paper_loss_free',
                'paper_dwa', 'mtp', 'mhc_enabled', 'qsa_mode', 'aux_coef'):
        assert m[key] == ref['model'][key], key
    return c


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--run-id', default='matched-lc3-e512-b15-c23-20261109')
    p.add_argument('--world', type=int, default=8)
    p.add_argument('--micro-bs', type=int, help='Default C2/world8=3; otherwise 2')
    a = p.parse_args()
    import re
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', a.run_id):
        p.error('Invalid run id')
    root = state_root(a.workspace)/'runs'/a.run_id
    snapshot = root/'source'
    if snapshot.exists():
        if source_hash(snapshot) != source_hash():
            raise RuntimeError('Existing immutable snapshot differs; use a new run ID')
    else:
        shutil.copytree(HERE, snapshot, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '._*'))
    rows = []
    for level in ('c2', 'c3'):
        c = make_config(level, a.workspace, a.run_id, a.world, a.micro_bs)
        f = root/level/'config.json'
        if f.exists() and read(f) != c:
            raise RuntimeError('Existing configuration differs: '+str(f))
        write(f, c)
        rows.append(dict(level=level, world=a.world, config=str(f), config_sha256=digest(f),
                         tokens=c['train']['tokens'], steps=c['accounting']['steps'],
                         total_params=c['accounting']['total_params_all'],
                         train_flops=c['accounting']['actual_train_flops']))
    manifest = dict(run_id=a.run_id, source_directory=str(snapshot), source_hash=source_hash(),
                    levels=rows, gpu_jobs_started=False,
                    reference_run='combined-lc3-e512-b15-c4-w8-20261109',
                    evaluation_profile=dict(mode='raw', sequence_length=10240,
                        precision='FP8 trunk/BF16 logits', single_gpu=True, batch_size='auto'))
    write(root/'manifest.json', manifest)
    import json
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
