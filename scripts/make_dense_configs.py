#!/usr/bin/env python3
"""Generate FFN-only dense counterparts of the frozen Q38 presets; never train."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from q38.flops import count
from q38.presets import ROOT, budget_tokens, dense_shape, model_config


def make_config(reference, run_dir):
    p = copy.deepcopy(reference)
    seq = p["train"]["seq"]
    vocab = p["accounting"]["vocab_size"]
    old = model_config(p["model"], vocab, seq)
    before = count(old.to_dict(), seq)
    p["model"] = dense_shape(p["model"], vocab, seq)
    cfg = model_config(p["model"], vocab, seq)
    after = count(cfg.to_dict(), seq)
    reserve = p["accounting"]["compute_reserve_fraction"]
    batch = p["train"]["batch_tokens"]
    tokens = budget_tokens(after["train_flops_per_token"], int(p["compute"]*(1-reserve)), batch)
    steps = tokens // batch
    p["train"].update(run_dir=str(run_dir), tokens=tokens,
                      warmup=min(steps, max(p["train"].get("warmup_min", 10),
                                            round(steps*p["train"].get("warmup_ratio", .01)))))
    p["accounting"] = {**after, "vocab_size": vocab, "compute_reserve_fraction": reserve,
                       "train_tokens": tokens, "steps": steps,
                       "actual_train_flops": tokens*after["train_flops_per_token"]}
    # A transferred Q35 prior is not evidence about this new dense architecture.
    p.pop("scaling_prior", None)
    p.pop("scaling_rule", None)
    p["comparison"] = {
        "reference_commit": "6daa43d6fdb5cb929036028fd7b03df488df1cc3",
        "reference_config_canonical_json_sha256": hashlib.sha256(json.dumps(reference, sort_keys=True).encode()).hexdigest(),
        "hash_serialization": "UTF-8 json.dumps(reference, sort_keys=True), default separators and ensure_ascii=True",
        "change": "Replace every main/MTP routed/shared-expert FFN by one dense SwiGLU; other modules unchanged",
        "dense_width": cfg.intermediate_size,
        "reference_active_params_excluding_embeddings": before["active_params_excluding_embeddings"],
        "dense_active_params_excluding_embeddings": after["active_params_excluding_embeddings"],
        "removed_main_router_and_shared_gate_params": old.num_hidden_layers*old.hidden_size*(old.num_experts+1),
        "ngram_rows_fixed_from_reference": old.ngram_rows,
        "budget": "Same target compute and reserve, each architecture rounded down to a full global batch",
    }
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=ROOT/"configs/dense")
    ap.add_argument("--run-root", type=Path, default=Path("runs/q38_dense"))
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    for tier in ("c1", "c2", "c3", "c4"):
        reference = json.loads((ROOT/f"configs/q38_{tier}.json").read_text())
        config = make_config(reference, args.run_root/tier)
        path = args.output/f"q38_{tier}"/"config.json"
        if args.check:
            if json.loads(path.read_text()) != config:
                raise ValueError(f"Generated config differs: {path}")
        else:
            if path.exists() and json.loads(path.read_text()) != config:
                raise FileExistsError(f"Refusing to overwrite a different config: {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(config, ensure_ascii=False, indent=2)+"\n")
        print(path)


if __name__ == "__main__":
    main()
