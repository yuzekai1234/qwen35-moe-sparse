#!/usr/bin/env python3
"""Fit completed Q38 dense IsoFLOPs curves and freeze a held-out C1 recipe."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from q38.presets import ROOT
from scripts.make_dense_configs import make_config
from scripts.make_q38_configs import shape_for
from scripts.make_q38_isoflops_configs import BUDGETS, serialized


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(manifest_path):
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("family") != "dense" or manifest.get("fit_budgets") != BUDGETS:
        raise ValueError("Only the four declared dense budgets can enter the fit")
    rows, ids = [], set()
    for job in manifest["jobs"]:
        if job["id"] in ids or job["compute"] not in BUDGETS.values():
            raise ValueError("Duplicate run or held-out/unrecognized compute in fit input")
        ids.add(job["id"])
        path, run = Path(job["config"]), Path(job["run_dir"])
        if sha(path) != job["expected_config_sha256"]:
            raise ValueError(f"Changed configuration: {path}")
        cfg = json.loads(path.read_text())
        paths = {name: run/(name+".json") for name in ("summary", "final_eval", "run_config", "flops")}
        s, e, r, f = [json.loads(paths[k].read_text()) for k in paths]
        train = cfg["train"]
        variable_runtime_fields = {"micro_bs", "auto_micro_bs", "resume", "tensorboard", "ckpt_every", "log_every"}
        if not (s.get("completed_budget") is True and e.get("completed_budget") is True and
                s["tokens_trained"] == e["tokens_trained"] == job["expected_tokens"] and
                s["train_flops"] == e["train_flops"] == job["expected_tokens"]*job["train_flops_per_token"] and
                r["compute"] == job["compute"] and r["steps"] == job["expected_steps"] and
                r["config_sha256"] == job["expected_config_sha256"] and
                r["seq"] == manifest["sequence_length"] and r["seed"] == manifest["seed"] and
                r["model_code"]["files"] == manifest["model_files_sha256"] and
                all(r.get(k) == v for k, v in cfg["model"].items()) and
                all(r.get(k) == v for k, v in train.items() if k not in variable_runtime_fields) and
                e["metric"] == "pure_cross_entropy" and math.isfinite(e["loss"]) and
                e["offset"] == manifest["validation"]["offset"] and
                e["scored_tokens"] == manifest["validation"]["targets"] and
                e["tokenizer_sha256"] == manifest["validation"]["tokenizer_sha256"] and
                r["data_identity"]["validation"]["declared_sha256"] == manifest["validation"]["validation_sha256"] and
                f["active_params_excluding_embeddings"] == job["Na"] and
                f["train_flops_per_token"] == job["train_flops_per_token"] and
                r["tokens"] == train["tokens"] and r["batch_tokens"] == train["batch_tokens"]):
            raise ValueError(f"Incomplete or incomparable fit input: {job['id']}")
        rows.append(dict(id=job["id"], budget=cfg["experiment"]["budget"],
            target_compute=job["compute"], compute=s["train_flops"], Na=job["Na"],
            D=s["tokens_trained"], loss=e["loss"], hidden=cfg["model"]["hidden"],
            run_dir=str(run), source_sha256={str(p):sha(p) for p in paths.values()}))
    return manifest, rows


def fit_curves(rows):
    if {r["target_compute"] for r in rows} != set(BUDGETS.values()):
        raise ValueError("Need all four low budgets, without C1")
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
    return dict(budget_fits=fits, Na_log10_coefficients=pn.tolist(),
                D_log10_coefficients=pd.tolist(), leave_one_budget_out=loo)


def c1_recipe(fit, run_dir):
    target_n = float(10**np.polyval(fit["Na_log10_coefficients"], math.log10(3e17)))
    target_d = float(10**np.polyval(fit["D_log10_coefficients"], math.log10(3e17)))
    template = json.loads((ROOT/"configs/q38_c1.json").read_text())
    candidates = []
    for hidden in range(64,1601,64):
        reference = copy.deepcopy(template)
        reference["model"] = shape_for(hidden)
        config = make_config(reference, run_dir)
        n, d = config["accounting"]["active_params_excluding_embeddings"], config["train"]["tokens"]
        score = math.log(n/target_n)**2+math.log(d/target_d)**2
        candidates.append((score, hidden, config))
    score, hidden, config = min(candidates, key=lambda c:c[0])
    return config, dict(compute=3*10**17, predicted_Na=target_n, predicted_D=target_d,
        selected_hidden=hidden, selected_layers=config["model"]["layers"],
        selected_Na=config["accounting"]["active_params_excluding_embeddings"],
        selected_D=config["train"]["tokens"], selection_score=score,
        selection_rule="min log(Na/Na_pred)^2 + log(D/D_pred)^2 on h64:64:1600",
        quality_prediction=None,
        limitation="N/D power laws do not predict CE or prove that one C1 size is optimal")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="New prediction directory outside source tree")
    p.add_argument("--c1-run-dir", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or a.c1_run_dir.exists():
        raise FileExistsError("Prediction and held-out training output must be new")
    manifest, rows = load_rows(a.manifest)
    fit = fit_curves(rows)
    config, prediction = c1_recipe(fit, a.c1_run_dir.absolute())
    result = dict(metric=manifest["metric"], metric_units=manifest["metric_units"],
        validation=manifest["validation"], source_commit=manifest["source_commit"],
        manifest_sha256=sha(a.manifest), fitting_code_sha256=sha(Path(__file__)),
        **fit, c1_prediction=prediction)
    a.output.mkdir(parents=True)
    (a.output/"fit.json").write_bytes(serialized(result))
    (a.output/"c1-config.json").write_bytes(serialized(config))
    print(json.dumps(prediction,indent=2))


if __name__ == "__main__":
    main()
