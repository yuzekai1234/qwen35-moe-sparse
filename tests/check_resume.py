"""Verify exact checkpoint restore and sample progression; report kernel numeric drift."""
import csv
import json
from pathlib import Path
import sys
import torch

left, right = map(Path, sys.argv[1:])
a, b = [torch.load(p / "ckpt.pt", map_location="cpu", weights_only=False) for p in (left, right)]
assert a["tok_seen"] == b["tok_seen"] == 1024
assert a["step"] == b["step"] == 1
assert a["world"] == b["world"] == 2
max_difference = 0.0
for key, tensor in a["model"].items():
    assert torch.isfinite(tensor).all(), key
    max_difference = max(max_difference, (tensor - b["model"][key]).abs().max().item())
    assert torch.isfinite(b["model"][key]).all(), key
verified = json.loads((left / "resume_verification.json").read_text())
assert len(verified) == 2
assert all(v["model_exact"] and v["optimizer_exact"] and v["rng_exact"] for v in verified)
for x, y in zip(a["rank_rng"], b["rank_rng"]):
    assert x["numpy"] == y["numpy"]
    assert torch.equal(x["torch"], y["torch"])
    assert torch.equal(x["cuda"], y["cuda"])
for path in (left, right):
    summary = json.loads((path / "summary.json").read_text())
    assert summary["completed_budget"] is True
    rows = list(csv.DictReader((path / "training_metrics.csv").open()))
    assert [int(row["tokens"]) for row in rows] == [512, 1024]
print(json.dumps({"resume_test": "passed", "max_weight_difference": max_difference,
                  "checkpoint_restore_exact": True, "rank_rng_states": len(a["rank_rng"]), "tokens": a["tok_seen"]}))
