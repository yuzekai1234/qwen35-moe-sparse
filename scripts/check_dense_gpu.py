#!/usr/bin/env python3
"""Verify a saved dense short run and portable BF16 GPU export; not a quality eval."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from q38.flops import count
from q38.models.qwen3_8_moe import Qwen3_8MoeForCausalLM, source_info
from q38.models.qwen3_8_moe import modeling_qwen3_8_moe as M


def verify(run_dir, updates):
    start = time.monotonic()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU qualification requires CUDA")
    torch._dynamo.config.suppress_errors = False
    config = json.loads((run_dir/"run_config.json").read_text())
    summary = json.loads((run_dir/"summary.json").read_text())
    checkpoint = torch.load(run_dir/"ckpt.pt", map_location="cpu", weights_only=False)
    if checkpoint["step"]+1 != updates or checkpoint["tok_seen"] != updates*config["batch_tokens"]:
        raise AssertionError("Checkpoint did not reach the requested update/token count")
    if summary["completed_budget"] or summary["tokens_trained"] != checkpoint["tok_seen"]:
        raise AssertionError("A short run must not be mislabeled as a completed compute budget")
    rows = list(csv.DictReader((run_dir/"training_metrics.csv").open()))
    if [int(r["step"]) for r in rows] != list(range(updates)):
        raise AssertionError("Expected one metric row per step, without replayed/lost resume steps")
    for row in rows:
        for key in ("loss", "aux", "grad_norm", "mtp_loss", "indexer_kl", "tok_per_s"):
            if not math.isfinite(float(row[key])):
                raise AssertionError(f"Nonfinite {key}: {row}")
        if float(row["aux"]) != 0:
            raise AssertionError("Dense has no routing auxiliary loss")
    adam_steps = sorted({int(state["step"].item()) for state in checkpoint["opt"]["adam"]["state"].values()})
    if adam_steps != [updates] or not checkpoint["opt"]["muon"]["state"]:
        raise AssertionError("Adam/Muon state was not preserved across resume")
    model = Qwen3_8MoeForCausalLM.from_pretrained(run_dir/"hf", local_files_only=True,
             dtype=torch.bfloat16, attn_implementation="sdpa").eval()
    if model.config.num_experts or any(isinstance(m, M.Qwen3_8MoeSparseMoeBlock) for m in model.modules()):
        raise AssertionError("Routed FFN remains in the dense main/MTP model")
    verified_tensors = []
    for name, parameter in model.named_parameters():
        expected = checkpoint["model"][name].to(dtype=parameter.dtype)
        if not torch.equal(parameter.detach(), expected):
            raise AssertionError(f"HF tensor differs from saved FP32 checkpoint cast: {name}")
        verified_tensors.append(name)
    fc = count(model.config.to_dict(), config["seq"])
    if sum(p.numel() for p in model.parameters()) != fc["total_params_all"]:
        raise AssertionError("Saved model and parameter accounting disagree")
    if model.mtp is None or not any(n.startswith("mtp.") for n in verified_tensors):
        raise AssertionError("Qualification must cover MTP tensors, not just main-head logits")
    del checkpoint
    data = np.memmap(config["val_data"], dtype="<u4", mode="r")
    offset, seq = config["final_eval_offset"], config["seq"]
    ids = torch.from_numpy(np.array(data[offset:offset+seq], dtype=np.int64)).unsqueeze(0).cuda()
    model.cuda()
    if model.training:
        raise AssertionError("Sparse exported inference must be in eval mode")
    with torch.inference_mode():
        expected_logits = model(input_ids=ids, use_cache=False).logits
    portable = AutoModelForCausalLM.from_pretrained(run_dir/"hf", trust_remote_code=True,
               local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    with torch.inference_mode():
        actual_logits = portable(input_ids=ids, use_cache=False).logits
    torch.testing.assert_close(actual_logits, expected_logits, rtol=0, atol=0)
    if not torch.isfinite(actual_logits).all():
        raise AssertionError("Nonfinite exported sparse logits")
    tail_lengths = []
    if model.config.qsa_mode == "full":
        # Full attention no longer needs QSA's 64-token input padding. Exercise
        # both the recurrent single-token GDN path and a non-aligned prefill.
        with torch.inference_mode():
            for length in (1, 65):
                tail = ids[:, :length]
                expected_tail = model(input_ids=tail, use_cache=False).logits
                actual_tail = portable(input_ids=tail, use_cache=False).logits
                torch.testing.assert_close(actual_tail, expected_tail, rtol=0, atol=0)
                if not torch.isfinite(actual_tail).all():
                    raise AssertionError(f"Nonfinite full-attention tail logits at length {length}")
                tail_lengths.append(length)
    return dict(passed=True, verification_kind="GPU short-run qualification, not final quality",
        torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
        updates=updates, trained_tokens=summary["tokens_trained"], full_budget_completed=False,
        warmup_updates=config["warmup"],
        attention_mode=model.config.qsa_mode, mhc_enabled=model.config.mhc_enabled,
        sparse_training_updates=max(0, updates-config["warmup"]) if model.config.qsa_mode == "learned" else 0,
        sequence_length=seq, batch_tokens=config["batch_tokens"], micro_bs=config["micro_bs"],
        adam_steps=adam_steps, verified_hf_tensors=verified_tensors,
        mtp_tensors_verified=sum(n.startswith("mtp.") for n in verified_tensors),
        export_logits_bitwise_equal=True,
        export_tail_lengths_verified=tail_lengths,
        export_sparse_logits_bitwise_equal=True if model.config.qsa_mode == "learned" else None,
        suppress_compile_errors=False,
        total_params=fc["total_params_all"], active_params_excluding_embeddings=fc["active_params_excluding_embeddings"],
        source=source_info(), run_dir=str(run_dir), elapsed_seconds=time.monotonic()-start)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=3)
    args = parser.parse_args()
    output = args.run_dir/"qualification.json"
    if output.exists():
        raise FileExistsError(output)
    result = verify(args.run_dir, args.updates)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
