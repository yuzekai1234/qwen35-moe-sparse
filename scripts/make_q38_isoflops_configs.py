#!/usr/bin/env python3
"""Generate the complete combined-model 17-point sweep, without data access."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from q38.presets import ROOT
from scripts.make_baseline_config import POOL, TOKENIZER, make_training_config
from scripts.make_q38_configs import SEQ
from scripts.q38_isoflops_contract import (
    BUDGETS, FAMILY, GRID, validate_config_data_paths, validate_data_identity,
)


def serialized(value):
    return (json.dumps(value, indent=2, ensure_ascii=False)+'\n').encode()


def build_plan(output_root, run_root, source_commit=None, *, data_identity,
               data_root=POOL, tokenizer=TOKENIZER):
    output_root, run_root = Path(output_root).absolute(), Path(run_root).absolute()
    identity = validate_data_identity(data_identity)
    validation = json.loads((ROOT/'configs/q35_isoflops/validation.json').read_text())
    configs, jobs = {}, []
    for label, compute in BUDGETS.items():
        for hidden in GRID[label]:
            config = make_training_config(hidden, compute, '/unused', data_root, tokenizer)
            validate_config_data_paths(config, identity)
            name = f"h{hidden}_L{config['model']['layers']}"
            run_dir = run_root/label/name
            config['train']['run_dir'] = str(run_dir)
            config['experiment'] = dict(family=FAMILY, budget=label, model=name)
            path = output_root/label/name/'config.json'
            configs[path] = config
            fc, train = config['accounting'], config['train']
            jobs.append(dict(id=f'{FAMILY}/{label}/{name}', budget=label, hidden=hidden,
                config=str(path), run_dir=str(run_dir), compute=compute,
                expected_tokens=train['tokens'], expected_steps=fc['steps'],
                train_flops_per_token=fc['train_flops_per_token'],
                Na=fc['active_params_excluding_embeddings'],
                expected_config_sha256=hashlib.sha256(serialized(config)).hexdigest(),
                final_eval_offset=train['final_eval_offset'], final_eval_targets=train['final_eval_tokens'],
                tokenizer_sha256=identity['tokenizer_sha256']))
    manifest = dict(schema_version=1, family=FAMILY, fit_budgets=BUDGETS,
        held_out_compute=3*10**17, sequence_length=SEQ, seed=0,
        source_commit=source_commit, data_root=str(data_root), tokenizer=str(tokenizer),
        data_identity=identity,
        shape_rule='legacy MoE shape_for -> resolve N-gram rows -> dense -> no mHC/full attention',
        grid_rule='Fixed original four-budget 17-width grid; no current-graph size filtering',
        hidden_grid={label:list(widths) for label,widths in GRID.items()},
        metric='pure_cross_entropy', metric_units='nats/target including EOS',
        parameter_count='active_params_excluding_embeddings; main head without MTP',
        model_files_sha256={name:hashlib.sha256((ROOT/'q38/models/qwen3_8_moe'/name).read_bytes()).hexdigest()
            for name in ('modeling_qwen3_8_moe.py', 'configuration_qwen3_8_moe.py')},
        validation=dict(offset=validation['curve']['offset'], targets=validation['curve']['targets'],
            tokenizer_sha256=identity['tokenizer_sha256'], validation_sha256=identity['validation']['sha256']),
        jobs=jobs)
    return configs, manifest


def write_plan(output_root, run_root, source_commit, check=False, *, data_identity,
               data_root=POOL, tokenizer=TOKENIZER):
    configs, manifest = build_plan(output_root, run_root, source_commit,
        data_identity=data_identity, data_root=data_root, tokenizer=tokenizer)
    files = {**configs, Path(output_root).absolute()/'manifest.json':manifest}
    for path,value in files.items():
        if check and not path.is_file():
            raise FileNotFoundError(path)
        if path.exists() and path.read_bytes() != serialized(value):
            raise FileExistsError(f'Refusing different existing configuration: {path}')
    if not check:
        for path,value in files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(serialized(value))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--data-identity', type=Path, required=True, help='Independent whole-file hash/stat snapshot')
    parser.add_argument('--data-root', type=Path, required=True, help='Explicit directory containing train.bin and valid.bin')
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
    manifest = write_plan(args.output, args.run_root, commit, args.check,
        data_identity=json.loads(args.data_identity.read_text()), data_root=args.data_root, tokenizer=args.tokenizer)
    print(json.dumps(dict(manifest=str(args.output.absolute()/'manifest.json'), jobs=len(manifest['jobs']),
                          budgets=manifest['fit_budgets'], source_commit=commit), indent=2))


if __name__ == '__main__':
    main()
