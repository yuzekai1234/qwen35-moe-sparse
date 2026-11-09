"""Generate four portable, budgeted candidate configs without importing torch."""
import argparse
import copy
import importlib.util
import json
import math
import re
import shutil
from pathlib import Path
from common import HERE, DEFAULT_WORKSPACE, digest, read, source_hash, state_root, write

BUDGETS = {'c1': 3*10**17, 'c2': 10**18, 'c3': 3*10**18, 'c4': 10**19}
# Memory-constrained starting grid; GPU fit and quality are explicitly unmeasured.
SHAPES = {'c1': (512, 10), 'c2': (640, 12), 'c3': (768, 14), 'c4': (768, 16)}
WORLDS = {'c1': 2, 'c2': 2, 'c3': 4, 'c4': 4}


def counter():
    spec = importlib.util.spec_from_file_location('campaign_flops', HERE/'runtime/q38/flops.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.count


def hf_config(model, seq):
    h = copy.deepcopy(read(HERE/'provenance/c1-best-hf.json'))
    mapping = dict(layers='num_hidden_layers', hidden='hidden_size', heads='num_attention_heads',
                   kv_heads='num_key_value_heads', lin_k_heads='linear_num_key_heads',
                   lin_v_heads='linear_num_value_heads', experts='num_experts', topk='num_experts_per_tok',
                   moe_ffn='moe_intermediate_size', shared_ffn='shared_expert_intermediate_size')
    for k, v in mapping.items():
        h[v] = model[k]
    for k in ('paper_engram', 'paper_loss_free', 'paper_dwa', 'paper_bias_rate', 'paper_hash_heads',
              'paper_hash_dim', 'ngram_rows', 'mtp', 'mhc_enabled', 'qsa_mode', 'longconv_layers',
              'longconv_channels', 'longconv_taps', 'longconv_filter_seqs'):
        h[k] = model[k]
    h['router_aux_loss_coef'] = model['aux_coef']
    h['layer_types'] = ['full_attention' if i%4 == 3 else 'linear_attention' for i in range(model['layers'])]
    h['max_position_embeddings'] = seq
    return h


def build(level, workspace, run_id, experts=1024, taps=2048, world=None):
    c = copy.deepcopy(read(HERE/'provenance/c1-best-training.json'))
    h, layers = SHAPES[level]
    m, t = c['model'], c['train']
    batch = 65536 if level in ('c1', 'c2') else 131072
    key_heads = max(1, round(3*h/512))
    m.update(hidden=h, layers=layers, heads=2*math.ceil(h/128), kv_heads=2,
             lin_k_heads=key_heads, lin_v_heads=3*key_heads, experts=experts,
             moe_ffn=round(h/64)*16, shared_ffn=round(h*480/512/16)*16,
             paper_hash_dim=h//4, longconv_layers=f'{layers//3},{2*layers//3}' if taps else '',
             longconv_channels=h if taps else 0, longconv_taps=taps, longconv_filter_seqs=batch//4096)
    fc = counter()(hf_config(m, 4096), 4096)
    budget = BUDGETS[level]
    # Half a percent retained for conversion, calibration and documented auxiliary work.
    tokens = (budget*995//1000)//(fc['train_flops_per_token']*batch)*batch
    steps = tokens//batch
    data = Path(workspace)/'c1-combined-papers-data-20261108/c1-pruned10k-v1'
    available_tokens = (data/'train.bin').stat().st_size//4 if (data/'train.bin').exists() else 5617222311
    available_blocks = (available_tokens-1)//4096
    epochs = math.ceil((tokens//4096)/available_blocks)
    world = world or WORLDS[level]
    if world < 1 or batch % (4096*world) or (taps and (taps%world or h%world)):
        raise ValueError('World size does not divide the global batch/LongConv shards')
    t.update(run_dir=str(state_root(workspace)/'runs'/run_id/level/'train'),
             data=str(data/'train.bin'), val_data=str(data/'valid.bin'), tokenizer=str(data/'tokenizer'),
             tokens=tokens, batch_tokens=batch, micro_bs=2 if level == 'c1' else 1,
             auto_micro_bs=0, warmup=max(10, round(steps*.01)), lr=.0014266*(h/512)**-.5,
             eval_every=2000, ckpt_every=2000, longconv_offload=1, checkpoint_ffn=1,
             optimizer_shard=1, tensorboard=0, resume=1, data_epochs=epochs)
    fc.update(steps=steps, train_tokens=tokens, actual_train_flops=tokens*fc['train_flops_per_token'],
              compute_reserve_fraction=.005)
    c.update(compute=budget, accounting=fc)
    c['variant'] = dict(paper_method='combined-papers', initialization='fresh_seed0',
        source=str(state_root(workspace)/'runs'/run_id/'source/runtime'), common_schedule=dict(start_step=round(steps*14719/16096),
            mode='linear_zero', auxiliary_hard_zero=True, longconv_tail_lr_mult=.25),
        scaling_rule='SHAPES in prepare.py; 2 LC blocks at floor(L/3), floor(2L/3); fixed E/topk/taps; Engram dim=h/4',
        status='prepared_unmeasured', run_id=run_id, level=level)
    taps_params = 2*taps*h*h if taps else 0
    resident = fc['total_params_all']-taps_params
    report = dict(level=level, width=h, layers=layers, experts=experts, taps=taps, world=world,
        tokens=tokens, steps=steps, epochs=epochs, train_flops=fc['actual_train_flops'],
        budget=budget, reserved_flops=budget-fc['actual_train_flops'],
        parameter_count=fc['total_params_all'],
        activation_ratio=fc['active_params_excluding_embeddings']/fc['total_params_excluding_embeddings'],
        resident_parameter_gradient_optimizer_lower_bound_gib=(8*resident+4*resident/world+2*taps_params/world)/2**30,
        memory_estimate_scope='Lower bound, not a fit proof: excludes activations, FFT workspaces, collectives and extra Adam slots',
        gpu_fit='not_tested', quality='not_tested')
    if report['activation_ratio'] < .1:
        report['submission_parameter_ratio'] = 'below_10_percent; research_only_until_resolved'
    return c, report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, default=DEFAULT_WORKSPACE)
    p.add_argument('--run-id', default='combined-v5')
    p.add_argument('--experts', type=int, default=1024)
    p.add_argument('--taps', type=int, default=2048)
    p.add_argument('--world', type=int, help='Override all four native worlds before the first run')
    a = p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', a.run_id) or a.experts < 4 or a.taps < 0:
        p.error('Invalid run id/experts/taps')
    root = state_root(a.workspace)/'runs'/a.run_id
    snapshot = root/'source'
    if snapshot.exists():
        if source_hash(snapshot) != source_hash():
            raise RuntimeError('An immutable source snapshot already exists; use another run id')
    else:
        root.mkdir(parents=True, exist_ok=True)
        shutil.copytree(HERE, snapshot, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '._*'))
    rows = []
    for level in BUDGETS:
        c, report = build(level, a.workspace.resolve(), a.run_id, a.experts, a.taps, a.world)
        path = root/level/'config.json'
        if path.exists() and read(path) != c:
            raise RuntimeError('Existing configuration differs; use another --run-id: '+str(path))
        write(path, c)
        rows.append(report | {'config': str(path), 'config_sha256': digest(path)})
    manifest = dict(run_id=a.run_id, source_hash=source_hash(), source_directory=str(snapshot), levels=rows,
                    evaluation_profile=dict(mode='raw', sequence_length=10240, precision='FP8 trunk/BF16 logits',
                                            single_gpu=True, batch_size='auto'),
                    priority=['c4', 'c2', 'c3', 'c1'], gpu_jobs_started=False)
    existing = root/'manifest.json'
    if existing.exists() and read(existing) != manifest:
        raise RuntimeError('Run manifest differs; use a new run id')
    write(existing, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
