"""Time, check and profile the contest forward on a fixed slice of the packed A set.

The forward is built exactly as eval.py builds it (runtime.build_forward from the
given evaluation directory), so a baseline and a candidate directory can be
compared on the same GPU, batch and token slice. Besides the forward time, the
summed NLL of the slice is reported to check numerical equivalence.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eval-dir', type=Path, required=True, help='evaluation source directory')
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--packed', type=Path, required=True, help='directory with tokens.bin and is_text.bin')
    parser.add_argument('--batch', type=int, default=512)
    parser.add_argument('--batches', type=int, default=1, help='consecutive batches timed per iteration')
    parser.add_argument('--start-seq', type=int, default=0)
    parser.add_argument('--seq', type=int, default=4096)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--iters', type=int, default=3)
    parser.add_argument('--profile', action='store_true', help='record one profiled forward of the first batch')
    parser.add_argument('--memory-history', action='store_true', help='record allocator history of the warmup forwards to memory-history.pickle')
    parser.add_argument('--output', type=Path, required=True, help='new output directory')
    return parser.parse_args()


def kernel_group(name):
    lowered = name.lower()
    rules = [('gemm_fp8', ('f8f8', 'e4m3', 'fp8_gemm', 'sm90_xmma_gemm_e4m3')),
             ('attention', ('cudnn', 'fmha', 'flash_fwd', 'flash_attn', 'sdpa')),
             ('gdn_core', ('flash_qla', 'flashqla', 'tilelang', 'gdr', 'kkt', 'chunk_', 'solve_tril', 'fused_recurrent')),
             ('causal_conv', ('causal_conv',)),
             ('gemm_other', ('gemm', 'cutlass', 'sm90_xmma', 'cublas', 'nvjet')),
             ('triton_inductor', ('triton_',)),
             ('copy_cat', ('copy', 'cat', 'elementwise', 'vectorized', 'fill'))]
    for group, needles in rules:
        if any(needle in lowered for needle in needles):
            return group
    return 'triton_custom' if not lowered.startswith('void') else 'other'


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.eval_dir.resolve()))
    os.environ.setdefault('Q38_COMPILE_EVIDENCE', str(args.output / 'compile-evidence'))
    from entry_defaults import apply_defaults
    apply_defaults()
    import numpy as np
    import torch
    torch._dynamo.config.suppress_errors = False
    from hopper_compat import install_flex_compat
    install_flex_compat()
    from transformers import AutoModelForCausalLM
    import eval as evaluation
    from runtime import build_forward

    device = torch.device('cuda', 0)
    torch.cuda.set_device(device)
    tokens = np.memmap(args.packed / 'tokens.bin', dtype=np.int32, mode='r')
    is_text = np.memmap(args.packed / 'is_text.bin', dtype=np.bool_, mode='r')
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model), trust_remote_code=True, dtype=torch.bfloat16,
        attn_implementation='sdpa', local_files_only=True)
    model.to(device).eval()
    pad = model.config.pad_token_id if model.config.pad_token_id is not None else model.config.eos_token_id
    started = time.time()
    forward, paths = build_forward(model)

    batches = []
    for index in range(args.batches):
        start = args.start_seq + index * args.batch
        inputs, labels = evaluation.packed_batch(tokens, is_text, start, start + args.batch, args.seq, pad)
        batches.append((torch.from_numpy(inputs).to(device), torch.from_numpy(labels).to(device)))
    scored = sum(int((labels != -100).sum()) for _, labels in batches)

    result = dict(eval_dir=str(args.eval_dir), model=str(args.model), batch=args.batch, batches=args.batches,
                  start_seq=args.start_seq, seq=args.seq, scored_tokens=scored,
                  gpu=torch.cuda.get_device_name(device), physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  torch=torch.__version__)
    with torch.inference_mode():
        compile_start = time.time()
        if args.memory_history:
            torch.cuda.memory._record_memory_history(max_entries=1000000, stacks='python')
        for _ in range(args.warmup):
            logits = forward(batches[0][0]); del logits
        torch.cuda.synchronize()
        if args.memory_history:
            torch.cuda.memory._dump_snapshot(str(args.output / 'memory-history.pickle'))
            torch.cuda.memory._record_memory_history(enabled=None)
        result['warmup_seconds'] = time.time() - compile_start
        result['warmup_peak_allocated_gib'] = torch.cuda.max_memory_allocated(device) / 2**30
        torch.cuda.reset_peak_memory_stats(device)

        nll = torch.zeros((), device=device, dtype=torch.float64)
        timings = []
        for iteration in range(args.iters):
            elapsed = 0.0
            for inputs, labels in batches:
                begin = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
                begin.record(); logits = forward(inputs); end.record(); torch.cuda.synchronize()
                elapsed += begin.elapsed_time(end) / 1000.0
                if iteration == 0:
                    flat = logits.reshape(-1, logits.size(-1)); target = labels.reshape(-1)
                    for offset in range(0, target.numel(), 8192):
                        nll += torch.nn.functional.cross_entropy(
                            flat[offset:offset + 8192].float(), target[offset:offset + 8192],
                            reduction='sum', ignore_index=-100).double()
                    del flat, target
                del logits
            timings.append(elapsed)
        result.update(forward_seconds=timings, forward_seconds_min=min(timings),
                      us_per_scored_token=min(timings) / scored * 1e6, slice_nll=float(nll),
                      slice_nats_per_token=float(nll) / scored,
                      timed_peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30)

        if args.profile:
            from torch.profiler import profile, ProfilerActivity
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                logits = forward(batches[0][0]); torch.cuda.synchronize(); del logits
            prof.export_chrome_trace(str(args.output / 'trace.json'))
            kernels = {}
            for event in prof.events():
                if event.device_type != torch.autograd.DeviceType.CUDA:
                    continue
                entry = kernels.setdefault(event.name, [0.0, 0])
                entry[0] += event.device_time / 1000.0; entry[1] += 1
            total = sum(v[0] for v in kernels.values())
            groups = {}
            for name, (ms, count) in kernels.items():
                group = groups.setdefault(kernel_group(name), [0.0, 0]); group[0] += ms; group[1] += count
            result['profile'] = dict(total_kernel_ms=total,
                groups={k: dict(ms=v[0], share=v[0] / total, calls=v[1]) for k, v in sorted(groups.items(), key=lambda x: -x[1][0])})
            rows = sorted(kernels.items(), key=lambda x: -x[1][0])
            with (args.output / 'kernels.tsv').open('w') as table:
                table.write('ms\tshare\tcalls\tgroup\tkernel\n')
                for name, (ms, count) in rows:
                    table.write(f'{ms:.3f}\t{ms / total:.4f}\t{count}\t{kernel_group(name)}\t{name[:300]}\n')
    result['paths'] = paths
    result['wall_seconds'] = time.time() - started
    (args.output / 'result.json').write_text(json.dumps(result, indent=2, default=str) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('paths',)}, indent=2, default=str), flush=True)


if __name__ == '__main__':
    main()
