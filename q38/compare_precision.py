"""Compare BF16 / INT8 W8A8 / FP8 W8A8 dense FFN quality on identical text.

CUDA_VISIBLE_DEVICES=0 python -m q38.compare_precision \
  --model-path /path/to/dense/hf --dataset-path /path/to/eval.jsonl \
  --output-dir runs/precision-bpb --batch-size 16

Scores all text by default. --max-documents N selects a deterministic prefix
for a pilot. Forward timing is disabled; results are not EvalTime measurements.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from .data import tokenizer_sha256
from .evaluate import active_parameter_count, pack_jsonl
from .evaluate_single_gpu import evaluate_model, load_model
from .quantization import PRECISIONS, apply_precision


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--precisions", nargs="+", choices=PRECISIONS, default=list(PRECISIONS))
    parser.add_argument("--backend", choices=("native", "reference"), default="native",
                        help="reference = explicit Q/DQ + FP32 GEMM; no speed claim")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Keep fixed across precisions; FP8 tensor scales depend on batch")
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--max-documents", type=int, help="Pilot: first N nonblank JSONL records")
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or not 1 <= args.sequence_length <= 8192:
        parser.error("batch-size must be positive and sequence-length must be in [1, 8192]")
    if args.max_documents is not None and args.max_documents < 1:
        parser.error("max-documents must be positive")
    if args.progress_every < 0 or len(set(args.precisions)) != len(args.precisions):
        parser.error("progress-every must be nonnegative; precisions must be unique")
    if args.output_dir.exists():
        parser.error("output-dir already exists; choose a new directory")
    return args


def compare_results(results):
    """Compute deltas only if this experiment includes its own BF16 baseline."""
    baseline = next((r["bpb"] for r in results if r["quantization"]["precision"] == "bf16"), None)
    for result in results:
        result["delta_bpb_vs_bf16"] = None if baseline is None else result["bpb"] - baseline
        result["relative_bpb_increase_percent"] = (None if baseline is None else
                                                   100 * (result["bpb"] / baseline - 1))
    return results


def main():
    args = parse_args()
    from transformers import PreTrainedTokenizerFast, set_seed
    from .models.qwen3_8_moe import Qwen3_8MoeTextConfig

    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one GPU with CUDA_VISIBLE_DEVICES")
    config = Qwen3_8MoeTextConfig.from_pretrained(args.model_path, local_files_only=True)
    if config.num_experts:
        raise ValueError("Use a dense checkpoint (num_experts=0)")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    args.output_dir.mkdir(parents=True)
    weight_files = sorted(set(args.model_path.glob("*.safetensors")) |
                          set(args.model_path.glob("pytorch_model*.bin")))
    if not weight_files:
        raise ValueError("No local checkpoint weight files found")
    identity = dict(
        model_path=str(args.model_path.resolve()), dataset_path=str(args.dataset_path.resolve()),
        model_files_sha256={p.name: sha256(p) for p in weight_files},
        config_sha256=sha256(args.model_path/"config.json"),
        tokenizer_sha256=tokenizer_sha256(args.model_path), dataset_sha256=sha256(args.dataset_path),
        implementation_sha256={name: sha256(Path(__file__).parent/name) for name in
                               ("quantization.py", "compare_precision.py", "evaluate.py", "evaluate_single_gpu.py",
                                "inference.py", "models/qwen3_8_moe/modeling_qwen3_8_moe.py")},
        dataset_scope="full" if args.max_documents is None else "first_documents",
        max_documents=args.max_documents, requested_precisions=args.precisions,
        torch_version=torch.__version__, cuda_version=torch.version.cuda,
        gpu_name=torch.cuda.get_device_name(device),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        compute_capability=list(torch.cuda.get_device_capability(device)),
    )
    results = []
    with tempfile.TemporaryDirectory(prefix=".packed-", dir=args.output_dir) as cache:
        cache = Path(cache)
        tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model_path, local_files_only=True)
        metadata = pack_jsonl(args.dataset_path, tokenizer, cache, max_documents=args.max_documents)
        identity["packing"] = metadata
        (args.output_dir/"experiment.json").write_text(json.dumps(identity, indent=2)+"\n")
        tokens = np.memmap(cache/"tokens.bin", dtype=np.int32, mode="r")
        mask = np.memmap(cache/"is_text.bin", dtype=np.bool_, mode="r")
        print(f"Packed {metadata['total_tokens']:,} text tokens / {metadata['total_bytes']:,} bytes", flush=True)
        for precision in args.precisions:
            print(f"Evaluating {precision}, backend={args.backend}, scope=main_dense_ffn", flush=True)
            set_seed(0)
            model = load_model(str(args.model_path), device)
            active_params = active_parameter_count(model)
            total_weights = sum(p.numel() for p in model.parameters())
            quantization = apply_precision(model, precision, args.backend)
            quantization["original_total_weight_elements"] = total_weights
            result = {**identity, "quantization": quantization, **evaluate_model(
                model, tokens, mask, metadata, args.batch_size, args.sequence_length, device, active_params,
                measure_time=False, warmup_steps=0, progress_every=args.progress_every)}
            results.append(result)
            compare_results(results)
            for item in results:
                path = args.output_dir/f"{item['quantization']['precision']}.json"
                path.write_text(json.dumps(item, indent=2, allow_nan=False)+"\n")
            (args.output_dir/"comparison.json").write_text(json.dumps(results, indent=2, allow_nan=False)+"\n")
            print(f"{precision}: BPB={result['bpb']:.9f}, delta={result['delta_bpb_vs_bf16']}", flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        del tokens, mask


if __name__ == "__main__":
    main()
