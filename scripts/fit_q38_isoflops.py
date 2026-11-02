#!/usr/bin/env python3
"""Fit all 17 combined-model runs; retain the already validated h512 C1."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from q38.presets import ROOT
from scripts.make_baseline_config import make_training_config
from scripts.make_q38_configs import BUDGETS as TRAINING_BUDGETS, SEQ
from scripts.make_q38_isoflops_configs import BUDGETS, serialized
from scripts.q38_isoflops_contract import (
    FAMILY, RUNTIME_OVERRIDES, runtime_identity_matches, validate_config_data_paths,
    validate_data_identity, validate_fit_grid,
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(manifest_path):
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("family") != FAMILY or manifest.get("fit_budgets") != BUDGETS:
        raise ValueError("Only the declared no-mHC/full-attention BF16 family can enter the fit")
    if not manifest.get('source_commit') or not manifest.get('model_files_sha256'):
        raise ValueError('Missing pinned training source identity')
    identity = validate_data_identity(manifest.get("data_identity"))
    validate_fit_grid((j.get("compute"), j.get("hidden")) for j in manifest["jobs"])
    partition = json.loads((ROOT/"configs/q35_isoflops/validation.json").read_text())
    expected_validation = dict(offset=partition["curve"]["offset"], targets=partition["curve"]["targets"],
        tokenizer_sha256=identity["tokenizer_sha256"], validation_sha256=identity["validation"]["sha256"])
    if manifest["validation"] != expected_validation:
        raise ValueError("Changed final validation partition or identity")
    rows, ids = [], set()
    for job in manifest["jobs"]:
        if job["id"] in ids or job["compute"] not in BUDGETS.values():
            raise ValueError("Duplicate run or held-out/unrecognized compute in fit input")
        ids.add(job["id"])
        path, run = Path(job["config"]), Path(job["run_dir"])
        if sha(path) != job["expected_config_sha256"]:
            raise ValueError(f"Changed configuration: {path}")
        cfg = json.loads(path.read_text())
        expected = make_training_config(job["hidden"], job["compute"], job["run_dir"],
            manifest["data_root"], manifest["tokenizer"])
        validate_config_data_paths(expected, identity)
        name = f"h{job['hidden']}_L{expected['model']['layers']}"
        expected["experiment"] = dict(family=FAMILY, budget=job["budget"], model=name)
        if cfg != expected or BUDGETS.get(job["budget"]) != job["compute"]:
            raise ValueError(f"Changed architecture or training recipe: {job['id']}")
        paths = {name: run/(name+".json") for name in ("summary", "final_eval", "run_config", "flops")}
        s, e, r, f = [json.loads(paths[k].read_text()) for k in paths]
        train = cfg["train"]
        if not (s.get("completed_budget") is True and e.get("completed_budget") is True and
                s["tokens_trained"] == e["tokens_trained"] == job["expected_tokens"] and
                s["train_flops"] == e["train_flops"] == job["expected_tokens"]*job["train_flops_per_token"] and
                r["compute"] == job["compute"] and r["steps"] == job["expected_steps"] and
                r["config_sha256"] == job["expected_config_sha256"] and
                r["seq"] == manifest["sequence_length"] and r["seed"] == manifest["seed"] and
                r["model_code"]["files"] == manifest["model_files_sha256"] and
                all(r.get(k) == v for k, v in cfg["model"].items()) and
                all(r.get(k) == v for k, v in train.items() if k not in RUNTIME_OVERRIDES) and
                e["metric"] == "pure_cross_entropy" and math.isfinite(e["loss"]) and
                e["offset"] == manifest["validation"]["offset"] and
                e["scored_tokens"] == manifest["validation"]["targets"] and
                e["tokenizer_sha256"] == manifest["validation"]["tokenizer_sha256"] and
                r.get("tokenizer_sha256") == identity["tokenizer_sha256"] and
                runtime_identity_matches(r.get("data_identity"), identity) and
                f["active_params_excluding_embeddings"] == job["Na"] and
                f["train_flops_per_token"] == job["train_flops_per_token"] and
                r["tokens"] == train["tokens"] and r["batch_tokens"] == train["batch_tokens"]):
            raise ValueError(f"Incomplete or incomparable fit input: {job['id']}")
        rows.append(dict(id=job["id"], family=FAMILY, budget=cfg["experiment"]["budget"],
            target_compute=job["compute"], compute=s["train_flops"], Na=job["Na"],
            D=s["tokens_trained"], loss=e["loss"], hidden=cfg["model"]["hidden"],
            run_dir=str(run), source_sha256={str(p):sha(p) for p in paths.values()}))
    return manifest, rows


def fit_curves(rows):
    if any(r.get("family") != FAMILY for r in rows):
        raise ValueError("Cannot mix legacy and current architecture fit inputs")
    validate_fit_grid((r.get("target_compute"), r.get("hidden")) for r in rows)
    fits = []
    for label, compute in BUDGETS.items():
        points = sorted((r for r in rows if r["target_compute"] == compute), key=lambda r:r["Na"])
        if len(points) < 4 or len({r["Na"] for r in points}) != len(points):
            raise ValueError(f"{label}: need >=4 different completed model sizes")
        x = np.log10([r["Na"] for r in points]); y = np.array([r["loss"] for r in points])
        if not np.isfinite(y).all():
            raise ValueError(f"{label}: nonfinite CE")
        if int(y.argmin()) in (0, len(points)-1):
            raise ValueError(f"{label}: best measured CE is on the grid boundary")
        coef = np.polyfit(x, y, 2)
        if coef[0] <= 0:
            raise ValueError(f"{label}: fitted curve is not convex")
        optimum = float(-coef[1]/(2*coef[0]))
        if not x.min() < optimum < x.max():
            raise ValueError(f"{label}: fitted optimum is outside the measured sizes")
        # Same log-D interpolation as the reference Q38 fitting script.
        dcoef = np.polyfit(x, np.log10([r["D"] for r in points]), 2)
        fits.append(dict(budget=label, compute=float(np.mean([r["compute"] for r in points])),
            compute_min=min(r["compute"] for r in points), compute_max=max(r["compute"] for r in points),
            quadratic_coefficients=coef.tolist(), Na_opt=10**optimum,
            D_opt=float(10**np.polyval(dcoef, optimum)),
            fitted_minimum_ce=float(np.polyval(coef, optimum)),
            residual_rmse=float(np.sqrt(np.mean((np.polyval(coef,x)-y)**2))),
            measured_best=min(points, key=lambda r:r["loss"]), points=points))
    log_c = np.log10([f["compute"] for f in fits])
    pn = np.polyfit(log_c, np.log10([f["Na_opt"] for f in fits]), 1)
    pd = np.polyfit(log_c, np.log10([f["D_opt"] for f in fits]), 1)
    loo = []
    for i, omitted in enumerate(fits):
        keep = [j for j in range(len(fits)) if j != i]
        n = np.polyfit(log_c[keep], np.log10([fits[j]["Na_opt"] for j in keep]), 1)
        d = np.polyfit(log_c[keep], np.log10([fits[j]["D_opt"] for j in keep]), 1)
        loo.append(dict(omitted_budget=omitted["budget"],
            Na_prediction=float(10**np.polyval(n, log_c[i])), Na_observed=omitted["Na_opt"],
            D_prediction=float(10**np.polyval(d, log_c[i])), D_observed=omitted["D_opt"],
            c1_Na_prediction=float(10**np.polyval(n, math.log10(3e17))),
            c1_D_prediction=float(10**np.polyval(d, math.log10(3e17)))))
    return dict(family=FAMILY, fit_budgets=BUDGETS, budget_fits=fits, Na_log10_coefficients=pn.tolist(),
                D_log10_coefficients=pd.tolist(), leave_one_budget_out=loo)


FIT_VALIDATION_RTOL = 1e-10
FIT_VALIDATION_ATOL = 1e-12


def validate_recorded_fit(fit):
    """Recompute for validation only; retain the original raw values and SHA."""
    if fit.get('family') != FAMILY or fit.get('fit_budgets') != BUDGETS:
        raise ValueError('Recorded fit is not the current architecture')
    identity = validate_data_identity(fit.get('data_identity'))
    if not isinstance(fit.get('data_root'), str) or not isinstance(fit.get('tokenizer'), str):
        raise ValueError('Recorded fit must identify its dataset and tokenizer locations')
    points = [point for curve in fit['budget_fits'] for point in curve['points']]
    validate_fit_grid((p.get('target_compute'), p.get('hidden')) for p in points)
    for point in points:
        compute, hidden = point['target_compute'], point['hidden']
        cfg = make_training_config(hidden, compute, '/unused', fit['data_root'], fit['tokenizer'])
        validate_config_data_paths(cfg, identity)
        label = point.get('budget')
        name = f"h{hidden}_L{cfg['model']['layers']}"
        if (point.get('family') != FAMILY or BUDGETS.get(label) != compute or
                point.get('id') != f'{FAMILY}/{label}/{name}'):
            raise ValueError('Recorded fit point has different grid metadata')
        expected = dict(Na=cfg['accounting']['active_params_excluding_embeddings'],
                        D=cfg['train']['tokens'], compute=cfg['accounting']['actual_train_flops'])
        if any(type(point.get(key)) is not int or point[key] != value for key,value in expected.items()):
            raise ValueError('Recorded point Na/D/actual FLOPs differ from the current graph')
        if (type(point.get('loss')) not in (int, float) or not math.isfinite(point['loss'])):
            raise ValueError('Recorded fit contains nonfinite or invalid CE')
    recomputed = fit_curves(points)

    def compare(actual, expected, path):
        if isinstance(expected, dict):
            if not isinstance(actual, dict) or set(actual) != set(expected):
                raise ValueError(f'Recorded fit structure differs at {path}')
            for key,value in expected.items():
                # Copies of measured evidence are exact, not derived floats.
                if key in ('points', 'measured_best'):
                    if actual[key] != value:
                        raise ValueError(f'Recorded fit evidence differs at {path}.{key}')
                else:
                    compare(actual[key], value, f'{path}.{key}')
        elif isinstance(expected, list):
            if not isinstance(actual, list) or len(actual) != len(expected):
                raise ValueError(f'Recorded fit structure differs at {path}')
            for i,(a,e) in enumerate(zip(actual, expected)):
                compare(a,e,f'{path}[{i}]')
        elif type(expected) is float:
            if (type(actual) not in (int, float) or not math.isfinite(actual) or
                    not math.isclose(actual, expected, rel_tol=FIT_VALIDATION_RTOL, abs_tol=FIT_VALIDATION_ATOL)):
                raise ValueError(f'Recorded fit numerical result differs at {path}')
        elif type(actual) is not type(expected) or actual != expected:
            raise ValueError(f'Recorded fit metadata differs at {path}')

    for key,value in recomputed.items():
        if key not in fit:
            raise ValueError(f'Missing recorded fit field: {key}')
        compare(fit[key], value, key)


def training_recipe(fit, tier, run_dir, data_root=None, tokenizer=None):
    """Select the current graph; keep h512 C1 and forbid corpus repetition."""
    if tier not in TRAINING_BUDGETS:
        raise ValueError("Training tier must be c1, c2, c3 or c4")
    compute = TRAINING_BUDGETS[tier]
    if fit.get("family") != FAMILY or fit.get("fit_budgets") != BUDGETS:
        raise ValueError("Training recipe requires a current-architecture N/D fit")
    identity = validate_data_identity(fit.get("data_identity"))
    data_root = fit.get('data_root') if data_root is None else data_root
    tokenizer = fit.get('tokenizer') if tokenizer is None else tokenizer
    if data_root is None or tokenizer is None or str(tokenizer) != fit.get('tokenizer'):
        raise ValueError('Training recipe must use the verified fit dataset/tokenizer locations')
    validate_config_data_paths(dict(train=dict(data=str(Path(data_root)/'train.bin'),
        val_data=str(Path(data_root)/'valid.bin'))), identity)
    unique_tokens = identity["train"]["bytes"]//4//SEQ*SEQ
    target_n = float(10**np.polyval(fit["Na_log10_coefficients"], math.log10(compute)))
    target_d = float(10**np.polyval(fit["D_log10_coefficients"], math.log10(compute)))
    if not all(math.isfinite(v) and v > 0 for v in (target_n, target_d)):
        raise ValueError("N/D predictions must be finite and positive")
    candidates = []
    for hidden in range(64,1601,64):
        config = make_training_config(hidden, compute, run_dir, data_root, tokenizer)
        n, d = config["accounting"]["active_params_excluding_embeddings"], config["train"]["tokens"]
        score = math.log(n/target_n)**2+math.log(d/target_d)**2
        candidates.append((score, hidden, config))
    free_score, free_hidden, free_config = min(candidates, key=lambda c:c[0])
    feasible = [row for row in candidates if row[2]["train"]["tokens"] <= unique_tokens]
    if tier == "c1":
        feasible = [row for row in feasible if row[1] == 512]
    if not feasible:
        raise ValueError("No allowed shape fits the unique training-token pool")
    score, hidden, config = min(feasible, key=lambda c:c[0])
    prediction = dict(compute=compute, predicted_Na=target_n, predicted_D=target_d,
        selected_hidden=hidden, selected_layers=config["model"]["layers"],
        selected_Na=config["accounting"]["active_params_excluding_embeddings"],
        selected_D=config["train"]["tokens"], selection_score=score,
        selection_rule="keep validated h512" if tier == "c1" else
            "min log(Na/Na_pred)^2 + log(D/D_pred)^2 on h64:64:1600, D <= unique tokens",
        unique_training_tokens=unique_tokens,
        unconstrained_grid_optimum=dict(hidden=free_hidden, layers=free_config["model"]["layers"],
            Na=free_config["accounting"]["active_params_excluding_embeddings"],
            D=free_config["train"]["tokens"], selection_score=free_score,
            fits_unique_pool=free_config["train"]["tokens"] <= unique_tokens),
        quality_prediction=None,
        limitation="N/D power laws do not predict CE or prove that a selected size is optimal")
    if tier == "c1":
        prediction["c1_free_choice_matches_validated"] = free_hidden == 512
    return config, prediction


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="New prediction directory outside source tree")
    p.add_argument("--c1-run-dir", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError("Prediction output must be new; C1 is not retrained")
    manifest, rows = load_rows(a.manifest)
    fit = fit_curves(rows)
    result = dict(metric=manifest["metric"], metric_units=manifest["metric_units"],
        validation=manifest["validation"], source_commit=manifest["source_commit"],
        data_identity=manifest["data_identity"], data_root=manifest["data_root"], tokenizer=manifest["tokenizer"],
        model_files_sha256=manifest["model_files_sha256"],
        manifest_sha256=sha(a.manifest), fitting_code_sha256=sha(Path(__file__)),
        **fit)
    config, prediction = training_recipe(result, "c1", a.c1_run_dir.absolute(),
        manifest["data_root"], manifest["tokenizer"])
    result["c1_prediction"] = prediction
    a.output.mkdir(parents=True)
    (a.output/"fit.json").write_bytes(serialized(result))
    (a.output/"c1-config.json").write_bytes(serialized(config))
    print(json.dumps(prediction,indent=2))


if __name__ == "__main__":
    main()
