"""Official single-GPU evaluation sample adapted to Q38 MoE and dense models.

Source: https://scicomp26.oss-cn-beijing.aliyuncs.com/dolma3_eval_example_1gpu.zip
The sample eval.py SHA256 is 2851c45a20ccdba18b68061685440a1ab4ee198ab11e13041a8bfa0262b0a5f3.
Packing, warmup, full-vocabulary forward timing and scoring follow that sample.
Q38 adaptations use the trained tokenizer, local model/fast paths and main-head
active parameters excluding every Embedding, the vocabulary head and unused MTP.
"""
import argparse
import json
import math
from pathlib import Path
import tempfile
import time

import numpy as np

from .evaluate import active_parameter_count, pack_jsonl, packed_batch


def load_model(model_path, device):
    import torch
    from .inference import optimize_for_inference
    from .models.qwen3_8_moe import Qwen3_8MoeTextConfig, Qwen3_8MoeForCausalLM

    config = Qwen3_8MoeTextConfig.from_pretrained(model_path, local_files_only=True).get_text_config()
    model, info = Qwen3_8MoeForCausalLM.from_pretrained(
        model_path, config=config, dtype=torch.bfloat16, attn_implementation="sdpa",
        local_files_only=True, output_loading_info=True,
    )
    if info.get("missing_keys") or info.get("mismatched_keys"):
        raise RuntimeError(f"Checkpoint did not fully initialize the language model: {info}")
    model.to(device).eval()
    print(f"Model implementation: {type(model).__module__}", flush=True)
    print(f"Inference fast paths: {optimize_for_inference(model)}", flush=True)
    return model


def evaluate_model(model, tokens, is_text, metadata, batch_size, sequence_length, device, active_params,
                   measure_time=True, warmup_steps=5, progress_every=0):
    """Score every packed target on one device; time only the complete forward."""
    import torch
    import torch.nn.functional as F
    from transformers import set_seed

    if batch_size < 1 or sequence_length < 1 or active_params <= 0:
        raise ValueError("Batch size, sequence length and active parameter count must be positive")
    if warmup_steps < 0 or progress_every < 0:
        raise ValueError("Warmup and progress intervals must be nonnegative")
    if len(tokens) < 2 or len(tokens) != len(is_text):
        raise ValueError("Packed token and scoring-mask lengths must agree and contain targets")
    num_sequences = math.ceil((len(tokens)-1)/sequence_length)
    num_steps = math.ceil(num_sequences/batch_size)

    def prepare_batch(step):
        start = step*batch_size
        stop = min(start+batch_size, num_sequences)
        inputs, labels = packed_batch(tokens, is_text, start, stop, sequence_length, metadata["pad_token_id"])
        return (torch.from_numpy(inputs).to(device), torch.from_numpy(labels).to(device),
                int(np.count_nonzero(labels != -100)))

    def forward_only(inputs):
        return model(input_ids=inputs, use_cache=False, return_dict=True).logits

    total_nll = torch.zeros((), device=device, dtype=torch.float64)
    scored_tokens = 0
    forward_seconds = 0.0
    with torch.inference_mode():
        warmed_shapes = set()
        for step in ((0, num_steps-1) if warmup_steps else ()):
            batch = prepare_batch(step)
            if tuple(batch[0].shape) in warmed_shapes:
                continue
            warmed_shapes.add(tuple(batch[0].shape))
            for _ in range(warmup_steps):
                logits = forward_only(batch[0])
                del logits
            if measure_time:
                torch.cuda.synchronize(device)

        set_seed(0)
        for step in range(num_steps):
            inputs, labels, count = prepare_batch(step)
            if measure_time:
                torch.cuda.synchronize(device)
                started = time.perf_counter()
            logits = forward_only(inputs)
            if measure_time:
                torch.cuda.synchronize(device)
                forward_seconds += time.perf_counter()-started

            flat_logits, flat_labels = logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)
            for offset in range(0, flat_labels.numel(), 512):
                loss = F.cross_entropy(flat_logits[offset:offset+512].float(), flat_labels[offset:offset+512],
                                       ignore_index=-100, reduction="none")
                total_nll += loss.double().sum()
            scored_tokens += count
            del flat_logits, flat_labels, loss, logits
            if progress_every and ((step+1) % progress_every == 0 or step+1 == num_steps):
                print(f"Scored {step+1}/{num_steps} batches, {scored_tokens:,} text tokens", flush=True)

    if scored_tokens != metadata["total_tokens"]:
        raise RuntimeError("Scored token count does not match the original text token count")
    bpb = total_nll.item()/(math.log(2)*metadata["total_bytes"])
    evaltime = forward_seconds/((active_params/1e9)*(scored_tokens/1e6)) if measure_time else None
    if not math.isfinite(bpb) or (measure_time and (not math.isfinite(evaltime) or evaltime <= 0)):
        raise RuntimeError("Evaluation produced a non-finite loss or invalid forward time")
    return dict(bpb=bpb, evaltime=evaltime, forward_seconds=forward_seconds if measure_time else None,
                timing_enabled=measure_time,
                active_params_excluding_embeddings=active_params, scored_tokens=scored_tokens,
                text_bytes=metadata["total_bytes"], total_nll_nats=total_nll.item(),
                sequence_length=sequence_length, batch_size=batch_size, num_sequences=num_sequences,
                num_steps=num_steps, gpu_count=1)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--batch-size", type=int, default=1, help="Largest power of two stable for this model")
    parser.add_argument("--sequence-length", type=int, default=4096, help="Value declared for this model")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or not 1 <= args.sequence_length <= 8192:
        parser.error("--batch-size must be positive and --sequence-length must be in [1, 8192]")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}; choose a new --output path")
    return args


def main():
    args = parse_args()
    import torch
    from transformers import PreTrainedTokenizerFast, set_seed

    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one GPU with CUDA_VISIBLE_DEVICES")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    set_seed(0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".eval-", dir=args.output.parent) as cache:
        cache = Path(cache)
        tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model_path, local_files_only=True)
        metadata = pack_jsonl(args.dataset_path, tokenizer, cache)
        print(f"Packed {metadata['total_tokens']} text tokens; {metadata['total_bytes']} original bytes", flush=True)
        tokens = np.memmap(cache/"tokens.bin", dtype=np.int32, mode="r")
        is_text = np.memmap(cache/"is_text.bin", dtype=np.bool_, mode="r")
        model = load_model(args.model_path, device)
        result = dict(model_path=args.model_path, **evaluate_model(
            model, tokens, is_text, metadata, args.batch_size, args.sequence_length, device,
            active_parameter_count(model)))
        print(f"Forward time: {result['forward_seconds']:.3f}s", flush=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
        print(json.dumps(result), flush=True)
        del tokens, is_text


if __name__ == "__main__":
    main()
