#!/usr/bin/env python3
"""Export combined-model recipes from the recorded current-architecture fit."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from q38.presets import ROOT
from scripts.fit_q38_isoflops import training_recipe, validate_recorded_fit
from scripts.make_q38_configs import BUDGETS as TRAINING_BUDGETS
from scripts.make_q38_isoflops_configs import BUDGETS as FIT_BUDGETS, serialized
from scripts.q38_isoflops_contract import (
    FAMILY, VALIDATION_OFFSET, VALIDATION_TARGETS, validate_config_data_paths,
    validate_data_identity, validate_fit_grid,
)

DEFAULT_OUTPUT = ROOT/"configs/dense_isoflops"
DEFAULT_RUN_ROOT = Path("runs/q38_dense_isoflops")
DEFAULT_FIT_SOURCE = ROOT/"docs/experiments/q38_combined_isoflops_results.json"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load_current_fit(source):
    """Verify the recorded raw fit without replacing its bytes or values."""
    if source is None:
        raise ValueError('A recorded current-architecture fit source is required')
    source = Path(source)
    raw = source.read_bytes()
    fit = json.loads(raw)
    if fit.get('family') != FAMILY or fit.get('fit_budgets') != FIT_BUDGETS:
        raise ValueError('Fit source is not the current no-mHC/full-attention BF16 family')
    identity = validate_data_identity(fit.get('data_identity'))
    if fit.get('metric') != 'pure_cross_entropy':
        raise ValueError('Expected the recorded pure-CE fit')
    if (len(fit['budget_fits']) != len(FIT_BUDGETS) or
            {curve['budget'] for curve in fit['budget_fits']} != set(FIT_BUDGETS) or
            any(point.get('target_compute') != FIT_BUDGETS[curve['budget']]
                for curve in fit['budget_fits'] for point in curve['points'])):
        raise ValueError('Expected four complete current-architecture budget curves')
    if fit.get('validation') != dict(offset=VALIDATION_OFFSET, targets=VALIDATION_TARGETS,
            tokenizer_sha256=identity['tokenizer_sha256'], validation_sha256=identity['validation']['sha256']):
        raise ValueError('Changed current-fit validation partition or identity')
    points = [point for curve in fit['budget_fits'] for point in curve['points']]
    validate_fit_grid((p.get('target_compute'), p.get('hidden')) for p in points)
    if any(p.get('family') != FAMILY for p in points):
        raise ValueError('Legacy model results cannot enter the current fit')
    for key in ('Na_log10_coefficients', 'D_log10_coefficients'):
        if len(fit[key]) != 2 or not all(math.isfinite(value) for value in fit[key]):
            raise ValueError('Expected two finite power-law coefficients')
    validate_recorded_fit(fit)
    return fit, dict(path=str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else source.name,
        sha256=sha(raw), training_source_commit=fit['source_commit'],
        manifest_sha256=fit['manifest_sha256'], fitting_code_sha256=fit['fitting_code_sha256'],
        data_identity_sha256=sha(serialized(identity)))


def build_plan(output=DEFAULT_OUTPUT, run_root=DEFAULT_RUN_ROOT, *, fit_source=DEFAULT_FIT_SOURCE,
               data_root=None, tokenizer=None):
    fit, source = load_current_fit(fit_source)
    data_root = fit['data_root'] if data_root is None else data_root
    tokenizer = fit['tokenizer'] if tokenizer is None else tokenizer
    if str(tokenizer) != fit['tokenizer']:
        raise ValueError('Tokenizer location differs from the verified current fit')
    files, selections = {}, {}
    for tier in TRAINING_BUDGETS:
        config, prediction = training_recipe(fit, tier, Path(run_root)/tier, data_root, tokenizer)
        validate_config_data_paths(config, fit['data_identity'])
        path = Path(output)/f"{tier}.json"
        files[path] = config
        selections[tier] = dict(prediction, config=path.name,
            config_sha256=sha(serialized(config)),
            selection_rule=prediction['selection_rule'], quality_prediction=None,
            validation_status='combined_c1_validated' if tier == 'c1' else
                'untrained_fitted_extrapolation')
    selection = dict(schema_version=2, family=FAMILY, scale_prior=None,
        current_architecture_fit=source, planned_fit_budgets=FIT_BUDGETS,
        architecture="dense SwiGLU, no mHC, causal full attention, BF16; C1 5.617B-token pool",
        configurations=selections,
        scope=("The current graph was fitted on all 17 low-budget runs. C1 keeps its validated h512 shape; "
               "C2-C4 are untrained extrapolations constrained to the unique token pool. "
               "No quality prediction or proof of optimal size."),
        Na_log10_coefficients=fit['Na_log10_coefficients'],
        D_log10_coefficients=fit['D_log10_coefficients'])
    files[Path(output)/"selection.json"] = selection
    return files, selection


def write_plan(output=DEFAULT_OUTPUT, run_root=DEFAULT_RUN_ROOT, check=False, *, fit_source=DEFAULT_FIT_SOURCE,
               data_root=None, tokenizer=None):
    files, selection = build_plan(output, run_root, fit_source=fit_source, data_root=data_root, tokenizer=tokenizer)
    # Validate all destinations before writing any file.
    for path, value in files.items():
        if check and not path.is_file():
            raise FileNotFoundError(path)
        if path.exists() and path.read_bytes() != serialized(value):
            raise FileExistsError(f"Refusing different existing configuration: {path}")
    if not check:
        for path, value in files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(serialized(value))
    return selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--check", action="store_true")
    parser.add_argument('--fit-source', type=Path, default=DEFAULT_FIT_SOURCE,
                        help='Recorded raw fit from the current 17-point graph (default: tracked result)')
    parser.add_argument('--data-root', type=Path, help='Must match the verified current-fit snapshot')
    parser.add_argument('--tokenizer', type=Path, help='Must match the verified current-fit location')
    args = parser.parse_args()
    selection = write_plan(args.output, args.run_root, args.check, fit_source=args.fit_source,
                           data_root=args.data_root, tokenizer=args.tokenizer)
    print(json.dumps(selection["configurations"], indent=2))


if __name__ == "__main__":
    main()
