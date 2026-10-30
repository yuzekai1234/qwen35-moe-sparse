#!/usr/bin/env python3
"""Generate the current C1 dense baseline; historical dense presets stay frozen."""
import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from q38.flops import count
from q38.presets import ROOT, budget_tokens, model_config


def make_config():
    config = copy.deepcopy(json.loads((ROOT/'configs/dense/q38_c1/config.json').read_text()))
    config['model'].update(layers=10, hidden=512, heads=20, dense_ffn=768,
                           ngram_rows=61440, mhc_enabled=0, qsa_mode='full')
    pool = '/home/shared/llm_pt_ablation/cache/ling-no-mtp-c1-global-pruned10k-scale-isoflops-v1'
    config['train'].update(run_dir='runs/q38_baseline/c1', data=pool+'/train.bin',
                           val_data=pool+'/valid.bin',
                           tokenizer='/home/shared/llm_pt/cache/tokenizers/ling-no-mtp-c1-top10k-v1',
                           require_ready_pool=0, micro_bs=8, auto_micro_bs=1,
                           resume=1, kl_coef=0.0, dense_warmup=0)
    cfg = model_config(config['model'], 10112, config['train']['seq'], 1, 53, 0)
    fc = count(cfg.to_dict(), config['train']['seq'])
    batch = config['train']['batch_tokens']
    tokens = budget_tokens(fc['train_flops_per_token'], config['compute']*995//1000, batch)
    steps = tokens//batch
    config['train'].update(tokens=tokens, warmup=max(10, round(steps*.01)))
    config['accounting'] = dict(fc, vocab_size=10112, compute_reserve_fraction=.005,
                                train_tokens=tokens, steps=steps,
                                actual_train_flops=tokens*fc['train_flops_per_token'])
    config.pop('comparison', None)
    config['baseline'] = dict(name='dense_no_mhc_full_attention_c1',
                              shape='h512/L10/F768; eight GDN and two causal full-attention layers',
                              status='configuration qualified; combined full-budget quality not measured',
                              preserve='N-gram, attention output gates, GQA, Q/K norms and training-only MTP')
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'configs/baseline.json')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    config = make_config()
    if args.check:
        if json.loads(args.output.read_text()) != config:
            raise ValueError(f'Generated baseline differs: {args.output}')
    else:
        if args.output.exists() and json.loads(args.output.read_text()) != config:
            raise FileExistsError(f'Refusing a different existing config: {args.output}')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(config, ensure_ascii=False, indent=2)+'\n')
    print(args.output)


if __name__ == '__main__':
    main()
