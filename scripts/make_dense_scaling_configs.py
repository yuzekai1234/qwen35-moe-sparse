#!/usr/bin/env python3
"""Export C1-C4 dense recipes from the recorded N/D fit, without refitting."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from q38.presets import ROOT
from scripts.fit_q38_isoflops import training_recipe
from scripts.make_q38_configs import BUDGETS as TRAINING_BUDGETS
from scripts.make_q38_isoflops_configs import BUDGETS as FIT_BUDGETS, serialized

SOURCE = ROOT/"docs/experiments/q38_dense_results.json"
DEFAULT_OUTPUT = ROOT/"configs/dense_isoflops"
DEFAULT_RUN_ROOT = Path("runs/q38_dense_isoflops")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load_fit(source=SOURCE):
    """Authenticate the original fit before using any of its coefficients."""
    raw = source.read_bytes()
    data = json.loads(raw)
    fit = data["isoflops"]
    digest = sha(serialized(fit))
    if digest != data["prediction_files_sha256"]["fit.json"]:
        raise ValueError("Recorded dense fit SHA256 differs from its frozen identity")
    if (fit["metric"] != "pure_cross_entropy" or len(fit["budget_fits"]) != len(FIT_BUDGETS) or
            {f["budget"] for f in fit["budget_fits"]} != set(FIT_BUDGETS) or
            any(p["target_compute"] != FIT_BUDGETS[f["budget"]]
                for f in fit["budget_fits"] for p in f["points"])):
        raise ValueError("Expected the four low-budget dense curves, without C1")
    return fit, dict(path=str(source.relative_to(ROOT)), sha256=sha(raw),
        fit_sha256=digest, training_source_commit=fit["source_commit"],
        manifest_sha256=fit["manifest_sha256"], fitting_code_sha256=fit["fitting_code_sha256"])


def build_plan(output=DEFAULT_OUTPUT, run_root=DEFAULT_RUN_ROOT):
    fit, source = load_fit()
    files, selections = {}, {}
    for tier in TRAINING_BUDGETS:
        config, prediction = training_recipe(fit, tier, Path(run_root)/tier)
        path = Path(output)/f"{tier}.json"
        files[path] = config
        selections[tier] = dict(prediction, config=path.name,
            config_sha256=sha(serialized(config)),
            validation_status="held_out_c1_validated" if tier == "c1" else "untrained_extrapolation")
    selection = dict(schema_version=1, family="dense_isoflops", fit_source=source,
        fit_budgets=FIT_BUDGETS, metric=fit["metric"], metric_units=fit["metric_units"],
        Na_log10_coefficients=fit["Na_log10_coefficients"],
        D_log10_coefficients=fit["D_log10_coefficients"],
        architecture="dense SwiGLU with mHC and learned QSA; original 16B-token training pool",
        configurations=selections,
        scope="C1 has the recorded held-out validation; C2-C4 are untrained extrapolations. "
              "No refitting, quality prediction or proof of optimal model size.")
    files[Path(output)/"selection.json"] = selection
    return files, selection


def write_plan(output=DEFAULT_OUTPUT, run_root=DEFAULT_RUN_ROOT, check=False):
    files, selection = build_plan(output, run_root)
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
    args = parser.parse_args()
    selection = write_plan(args.output, args.run_root, args.check)
    print(json.dumps(selection["configurations"], indent=2))


if __name__ == "__main__":
    main()
