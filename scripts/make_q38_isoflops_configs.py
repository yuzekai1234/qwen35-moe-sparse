#!/usr/bin/env python3
"""Generate the four-budget Q38 dense sweep; no training or C1 fit input."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from q38.flops import count
from q38.presets import ROOT, budget_tokens, model_config
from scripts.make_dense_configs import make_config
from scripts.make_q38_configs import shape_for, batch_for, COMPUTE_RESERVE, SEQ, VOCAB

BUDGETS = {"3e15": 3*10**15, "1e16": 10**16, "3e16": 3*10**16, "1e17": 10**17}


def serialized(value):
    return (json.dumps(value, indent=2, ensure_ascii=False)+"\n").encode()


def build_plan(output_root, run_root, source_commit=None):
    output_root, run_root = Path(output_root).absolute(), Path(run_root).absolute()
    template = json.loads((ROOT/"configs/q38_c1.json").read_text())
    validation_path = ROOT/"configs/q35_isoflops/validation.json"
    validation = json.loads(validation_path.read_text())
    configs, jobs = {}, []
    for label, compute in BUDGETS.items():
        batch = batch_for(compute)
        # Original Q38 rule uses a 64-wide grid starting at 128. Include its
        # h64 continuation for a fourth lowest-budget point, before seeing CE.
        for hidden in range(64, 1601, 64):
            shape = shape_for(hidden)
            moe = model_config(shape, VOCAB, SEQ)
            shape["ngram_rows"] = moe.ngram_rows
            reference_count = count(moe.to_dict(), SEQ)
            tokens = budget_tokens(reference_count["train_flops_per_token"],
                                   int(compute*(1-COMPUTE_RESERVE)), batch)
            ratio = tokens/reference_count["active_params_excluding_embeddings"]
            if not 4 <= ratio <= 400:
                continue
            name = f"h{hidden}_L{shape['layers']}"
            run_dir = run_root/label/name
            reference = copy.deepcopy(template)
            reference.update(compute=compute, model=shape)
            reference["train"].update(seq=SEQ, batch_tokens=batch,
                lr=round(.3118*compute**-.125, 6), max_steps=0,
                eval_offset=validation["dev"]["offset"], eval_tokens=validation["dev"]["targets"],
                final_eval_offset=validation["curve"]["offset"],
                final_eval_tokens=validation["curve"]["targets"])
            config = make_config(reference, run_dir)
            config["experiment"] = dict(family="dense", budget=label, model=name)
            path = output_root/label/name/"config.json"
            configs[path] = config
            fc, train = config["accounting"], config["train"]
            jobs.append(dict(id=f"dense/{label}/{name}", config=str(path), run_dir=str(run_dir),
                compute=compute, expected_tokens=train["tokens"], expected_steps=fc["steps"],
                train_flops_per_token=fc["train_flops_per_token"],
                Na=fc["active_params_excluding_embeddings"],
                expected_config_sha256=hashlib.sha256(serialized(config)).hexdigest(),
                final_eval_offset=train["final_eval_offset"],
                final_eval_targets=train["final_eval_tokens"],
                tokenizer_sha256=validation["tokenizer_sha256"]))
    manifest = dict(schema_version=1, family="dense", fit_budgets=BUDGETS,
        held_out_compute=3*10**17, sequence_length=SEQ, seed=0,
        source_commit=source_commit,
        shape_rule="scripts/make_q38_configs.py:shape_for",
        grid_rule="h64:64:1600; MoE reference 4 <= D/Na <= 400; dense N-gram rows fixed from MoE",
        metric="pure_cross_entropy", metric_units="nats/target including EOS",
        parameter_count="active_params_excluding_embeddings; main head without MTP",
        model_files_sha256={name: hashlib.sha256((ROOT/"q38/models/qwen3_8_moe"/name).read_bytes()).hexdigest()
            for name in ("modeling_qwen3_8_moe.py", "configuration_qwen3_8_moe.py")},
        validation=dict(offset=validation["curve"]["offset"], targets=validation["curve"]["targets"],
            tokenizer_sha256=validation["tokenizer_sha256"], validation_sha256=validation["validation_sha256"]),
        jobs=jobs)
    return configs, manifest


def write_plan(output_root, run_root, source_commit, check=False):
    configs, manifest = build_plan(output_root, run_root, source_commit)
    files = {**configs, Path(output_root).absolute()/"manifest.json": manifest}
    # Validate every destination before writing any file.
    for path, value in files.items():
        if check and not path.is_file():
            raise FileNotFoundError(path)
        if path.exists() and path.read_bytes() != serialized(value):
            raise FileExistsError(f"Refusing different existing configuration: {path}")
    if not check:
        for path, value in files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(serialized(value))
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True, help="Generated configurations outside the source tree")
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--check", action="store_true")
    a = p.parse_args()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    m = write_plan(a.output, a.run_root, commit, a.check)
    print(json.dumps(dict(manifest=str(a.output.absolute()/"manifest.json"), jobs=len(m["jobs"]),
                          budgets=m["fit_budgets"], source_commit=commit), indent=2))


if __name__ == "__main__":
    main()
