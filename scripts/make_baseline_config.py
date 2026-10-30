#!/usr/bin/env python3
"""Generate the current C1 dense baseline; historical dense presets stay frozen."""
import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from q38.flops import count
from q38.presets import ROOT, budget_tokens, dense_shape, model_config
from scripts.make_q38_configs import SEQ, VOCAB, batch_for, shape_for

FAMILY = 'dense_no_mhc_full_attention_bf16'
POOL = '/home/shared/llm_pt_ablation/cache/ling-no-mtp-c1-global-pruned10k-scale-isoflops-v1'
TOKENIZER = '/home/shared/llm_pt/cache/tokenizers/ling-no-mtp-c1-top10k-v1'


def make_training_config(hidden, compute, run_dir):
    """Build the combined architecture; N-gram rows come from the legacy MoE."""
    if hidden not in range(64, 1601, 64) or not isinstance(compute, int) or compute <= 0:
        raise ValueError('Require h64:64:1600 and a positive integer compute budget')
    config = copy.deepcopy(json.loads((ROOT/'configs/dense/q38_c1/config.json').read_text()))
    # dense_shape materializes rows before either graph-changing switch is set.
    config['model'] = dict(dense_shape(shape_for(hidden), VOCAB, SEQ),
                           mhc_enabled=0, qsa_mode='full')
    config['compute'] = compute
    config['train'].update(run_dir=str(run_dir), data=POOL+'/train.bin',
                           val_data=POOL+'/valid.bin', tokenizer=TOKENIZER,
                           seq=SEQ, batch_tokens=batch_for(compute),
                           lr=round(.3118*compute**-.125, 6),
                           require_ready_pool=0, micro_bs=4, auto_micro_bs=0,
                           resume=1, kl_coef=0.0, dense_warmup=0,
                           ffn_precision='bf16', projection_precision='bf16',
                           ffn_calibration='', precision_transition=0)
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
    return config


def make_config():
    config = make_training_config(512, 3*10**17, 'runs/q38_baseline/c1')
    config['baseline'] = dict(name='dense_no_mhc_full_attention_c1',
                              shape='h512/L10/F768; eight GDN and two causal full-attention layers',
                              results='docs/experiments/q38_baseline.md',
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
