#!/usr/bin/env python3
"""Export provisional combined-model recipes; the old fit is a size prior only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from q38.presets import ROOT
from scripts.make_baseline_config import FAMILY, make_training_config
from scripts.make_q38_configs import BUDGETS as TRAINING_BUDGETS
from scripts.make_q38_isoflops_configs import BUDGETS as FIT_BUDGETS, serialized

SOURCE = ROOT/"docs/experiments/q38_dense_results.json"
DEFAULT_OUTPUT = ROOT/"configs/dense_isoflops"
DEFAULT_RUN_ROOT = Path("runs/q38_dense_isoflops")
PROVISIONAL_HIDDEN = dict(c1=512, c2=640, c3=832, c4=1024)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load_scale_prior(source=SOURCE):
    """Authenticate the legacy fit, which did not measure the combined graph."""
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
    return dict(path=str(source.relative_to(ROOT)), sha256=sha(raw),
        fit_sha256=digest, training_source_commit=fit["source_commit"],
        manifest_sha256=fit["manifest_sha256"], fitting_code_sha256=fit["fitting_code_sha256"],
        architecture="dense with mHC and learned QSA, trained on the original 16B-token pool",
        role="Hidden-width reference only; not an N/D fit of the current architecture")


def build_plan(output=DEFAULT_OUTPUT, run_root=DEFAULT_RUN_ROOT):
    source = load_scale_prior()
    files, selections = {}, {}
    for tier, compute in TRAINING_BUDGETS.items():
        config = make_training_config(PROVISIONAL_HIDDEN[tier], compute, Path(run_root)/tier)
        path = Path(output)/f"{tier}.json"
        files[path] = config
        selections[tier] = dict(compute=compute, selected_hidden=config['model']['hidden'],
            selected_layers=config['model']['layers'],
            selected_Na=config['accounting']['active_params_excluding_embeddings'],
            selected_D=config['train']['tokens'], config=path.name,
            config_sha256=sha(serialized(config)),
            selection_rule='fixed provisional hidden width; current graph budget recomputed',
            quality_prediction=None,
            validation_status="combined_c1_validated" if tier == "c1" else "untrained_provisional")
    selection = dict(schema_version=2, family=FAMILY, scale_prior=source,
        current_architecture_fit=None, planned_fit_budgets=FIT_BUDGETS,
        architecture="dense SwiGLU, no mHC, causal full attention, BF16; C1 5.617B-token pool",
        configurations=selections,
        scope="C1 keeps its validated h512 shape. C2-C4 are untrained provisional sizes. "
              "The old N/D fit is only a width reference, not evidence for the current graph. "
              "No current-architecture fit, quality prediction or proof of optimal size.")
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
