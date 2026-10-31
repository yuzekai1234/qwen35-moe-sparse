"""Contest evaluation: JSONL -> packing -> forward -> bpb / EvalTime.

Usage (wrapped by evaluation/eval.sh):
  bash evaluation/eval.sh <dataset.jsonl> <model_dir>

Writes result.json in the calling working directory with exactly three
fields: model_path, bpb, evaltime.

EvalTime is forward-only seconds divided by
(1e9 non-vocabulary active parameters * 1e6 scored tokens), matching the
accounting recorded in MODEL_CARD.md.
"""

import argparse
import gc
import hashlib
import json
import math
import os
import tempfile
import time
from itertools import islice
from pathlib import Path

from entry_defaults import apply_defaults,explicit_batch,ensure_python
if __name__=='__main__':ensure_python()
apply_defaults()
import numpy as np


def pack_jsonl(dataset_path, tokenizer, cache_dir):
    """Write a flat token stream and a mask distinguishing text from inserted EOS."""
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("The tokenizer must define eos_token_id for document separation")
    total_bytes = total_tokens = 0
    with (
        dataset_path.open(encoding="utf-8") as source,
        (cache_dir / "tokens.bin").open("wb") as token_file,
        (cache_dir / "is_text.bin").open("wb") as mask_file,
    ):
        records = (json.loads(line)["text"] for line in source if line.strip())
        while texts := list(islice(records, 128)):
            total_bytes += sum(len(text.encode("utf-8")) for text in texts)
            encoded = tokenizer(
                texts, add_special_tokens=False, truncation=False,
                return_attention_mask=False, return_token_type_ids=False,
            )["input_ids"]
            for text, ids in zip(texts, encoded):
                if text and not ids:
                    raise ValueError("Tokenizer returned no tokens for nonempty text")
                np.asarray([eos, *ids], dtype=np.int32).tofile(token_file)
                mask = np.ones(len(ids) + 1, dtype=np.bool_)
                mask[0] = False
                mask.tofile(mask_file)
                total_tokens += len(ids)
    if not total_bytes or not total_tokens:
        raise ValueError("The dataset must contain nonempty text")
    return dict(
        total_bytes=total_bytes, total_tokens=total_tokens,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos,
    )


def packed_batch(tokens, is_text, start, stop, sequence_length, pad_token_id):
    """Return input/target arrays; the first target is the first real text token."""
    inputs = np.full((stop - start, sequence_length), pad_token_id, dtype=np.int64)
    labels = np.full_like(inputs, -100)
    for row, sequence_index in enumerate(range(start, stop)):
        offset = sequence_index * sequence_length
        length = min(sequence_length, len(tokens) - 1 - offset)
        inputs[row, :length] = tokens[offset:offset + length]
        target_slice = slice(offset + 1, offset + length + 1)
        labels[row, :length] = np.where(is_text[target_slice], tokens[target_slice], -100)
    return inputs, labels


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--batch-size", type=int, default=None, help="Explicit batch, else EVAL_BATCH_SIZE, else fresh-process automatic maximum")
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--output", type=Path, default=Path("result.json"))
    args = parser.parse_args()
    if not 1 <= args.sequence_length <= 8192:
        parser.error("--sequence-length must be between 1 and 8192")
    try:
        args.batch_size=explicit_batch(args.batch_size)
    except ValueError as error:
        parser.error(str(error))
    return args


def main():
    args = parse_args()
    from dependency_guard import verify
    dependency_evidence=verify()
    if args.batch_size is None:
        from autobatch import run_automatic
        run_automatic(args)
        return
    run_started = time.time()
    import torch
    import torch.distributed as dist
    # Performance qualification must not silently benchmark an eager fallback.
    torch._dynamo.config.suppress_errors = False
    from hopper_compat import install_flex_compat
    print("Runtime compatibility:", install_flex_compat(), flush=True)
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, set_seed

    from flops import count

    rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_dist = world_size > 1
    if use_dist:
        dist.init_process_group("nccl", device_id=device)
    set_seed(0)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    temporary = None
    shared = [None]
    if rank == 0:
        temporary = tempfile.TemporaryDirectory(prefix=".eval-", dir=args.output.parent)
        cache_dir = Path(temporary.name).resolve()
        tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model_path, local_files_only=True)
        metadata = pack_jsonl(args.dataset_path, tokenizer, cache_dir)
        shared[0] = (str(cache_dir), metadata)
        print(f"Packed {metadata['total_tokens']} text tokens", flush=True)
    if use_dist:
        dist.broadcast_object_list(shared, src=0, device=device)
    cache_path, metadata = shared[0]
    tokens = np.memmap(Path(cache_path) / "tokens.bin", dtype=np.int32, mode="r")
    is_text = np.memmap(Path(cache_path) / "is_text.bin", dtype=np.bool_, mode="r")
    num_sequences = math.ceil((len(tokens) - 1) / args.sequence_length)
    global_batch_size = args.batch_size * world_size
    num_steps = math.ceil(num_sequences / global_batch_size)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True,
    )
    print(f"Model implementation: {type(model).__module__}", flush=True)
    model.to(device).eval()
    from runtime import build_forward
    inference_forward, paths = build_forward(model)
    print("Inference fast paths:", paths, flush=True)
    active_params = count(model.config.to_dict(), args.sequence_length)["active_params_excluding_embeddings"]
    if active_params <= 0:
        raise ValueError("Non-vocabulary active parameter count must be positive")

    def prepare_batch(step):
        start = step * global_batch_size + rank * args.batch_size
        stop = min(start + args.batch_size, num_sequences)
        if start >= stop:
            return None
        inputs, labels = packed_batch(
            tokens, is_text, start, stop, args.sequence_length, metadata["pad_token_id"]
        )
        return (torch.from_numpy(inputs).to(device), torch.from_numpy(labels).to(device),
                int(np.count_nonzero(labels != -100)))

    def forward_only(inputs):
        return inference_forward(inputs)

    total_nll = torch.zeros((), device=device, dtype=torch.float64)
    scored_tokens = 0
    forward_seconds = 0.0
    actual_batch_sizes = []
    materialized_logits_dtype = None
    with torch.inference_mode():
        last_step = (num_sequences - 1 - rank * args.batch_size) // global_batch_size
        warmed = set()
        allocated_before_warmup = torch.cuda.memory_allocated(device)
        for step in (0, last_step):
            if step < 0:
                continue
            batch = prepare_batch(step)
            if batch is None or tuple(batch[0].shape) in warmed:
                continue
            warmed.add(tuple(batch[0].shape))
            for _ in range(5):
                logits = forward_only(batch[0])
                del logits
            torch.cuda.synchronize(device)
        # Compilation can leave reference cycles that hold tensors of the first forwards;
        # collect them before timing and log what the warmup left allocated.
        gc.collect()
        print("WARMUP_MEMORY", json.dumps(dict(allocated_before=allocated_before_warmup,
              allocated_after=torch.cuda.memory_allocated(device))), flush=True)
        if use_dist:
            dist.barrier()

        for step in range(num_steps):
            batch = prepare_batch(step)
            if batch is None:
                continue
            inputs, labels, count_tokens = batch
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            logits = forward_only(inputs)
            end_event.record()
            torch.cuda.synchronize(device)
            forward_seconds += start_event.elapsed_time(end_event) / 1000.0
            assert logits.dtype == torch.bfloat16, 'Qualified release requires BF16 logits'
            assert tuple(logits.shape) == (*inputs.shape, model.config.vocab_size), 'All token/vocabulary logits must be materialized'
            materialized_logits_dtype = str(logits.dtype)
            actual_batch_sizes.append(int(inputs.shape[0]))
            # packed_batch already aligns labels with inputs (labels[i] is the token
            # after inputs[i]), so logits are used as-is: no extra shift.
            flat_logits = logits.reshape(-1, logits.size(-1))
            flat_labels = labels.reshape(-1)
            step_nll = torch.zeros((), device=device, dtype=torch.float64)
            for offset in range(0, flat_labels.numel(), 512):
                chunk = torch.nn.functional.cross_entropy(
                    flat_logits[offset:offset + 512].float(),
                    flat_labels[offset:offset + 512],
                    reduction="sum", ignore_index=-100,
                )
                step_nll += chunk.to(torch.float64)
            total_nll += step_nll
            scored_tokens += count_tokens
            del logits, flat_logits, flat_labels, step_nll, inputs, labels, batch
            if step % 5 == 0 or step == num_steps - 1:
                print("PROGRESS", json.dumps(dict(step=step + 1, total_steps=num_steps,
                      scored_tokens=scored_tokens, forward_seconds=forward_seconds)), flush=True)

    stats = torch.tensor([total_nll.item(), float(scored_tokens), forward_seconds],
                         dtype=torch.float64, device=device)
    if use_dist:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    if rank == 0:
        nll, scored, seconds = stats.tolist()
        if scored <= 0:
            raise ValueError("No scored tokens")
        if not math.isfinite(nll) or not math.isfinite(seconds) or seconds <= 0:
            raise RuntimeError("Non-finite loss or invalid forward timing")
        bpb = nll / metadata["total_bytes"] / math.log(2)
        evaltime = seconds / ((active_params / 1e9) * (scored / 1e6))
        result = {"model_path": args.model_path, "bpb": bpb, "evaltime": evaltime}
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
        run_metadata = dict(
            batch_size=args.batch_size, requested_batch_size=args.batch_size,
            dependency_guard=dependency_evidence,
            actual_batch_sizes=actual_batch_sizes,max_actual_batch_size=max(actual_batch_sizes),
            materialized_logits_dtype=materialized_logits_dtype,
            sequence_length=args.sequence_length,
            gpu_count=world_size, gpu=torch.cuda.get_device_name(device),
            gpu_total_bytes=torch.cuda.get_device_properties(device).total_memory,
            torch=torch.__version__, cuda_runtime=torch.version.cuda,
            dataset_path=str(args.dataset_path), dataset_file_bytes=args.dataset_path.stat().st_size,
            scored_tokens=int(scored), text_utf8_bytes=metadata["total_bytes"], total_nll=nll,
            active_nonvocab_params=active_params, active_params_excluding_embeddings=active_params,
            active_parameter_counting="exclude_all_embeddings_vocab_head_and_unused_MTP", forward_seconds=seconds,
            wall_seconds=time.time()-run_started, peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
            allocator=os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "default"),
            inference_paths=paths, started_unix=run_started, finished_unix=time.time())
        with args.dataset_path.open("rb") as dataset_source:
            run_metadata["dataset_sha256"] = hashlib.file_digest(dataset_source, "sha256").hexdigest()
        if paths.get("variant", "baseline") != "baseline":
            from low_precision import NATIVE_FP8_EVIDENCE
            run_metadata["native_fp8_evidence"] = dict(NATIVE_FP8_EVIDENCE)
        args.output.with_name("run-metadata.json").write_text(json.dumps(run_metadata, indent=2)+"\n")
    if use_dist:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
