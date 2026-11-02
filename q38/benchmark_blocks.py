"""Warmed complete-forward timing and CUDA-event component attribution.

Uses a real packed batch and checkpoint. Component timings are measured in a
separate instrumented pass; nested projection timings must not be added to their
parent GDN/FFN timings. No scoring or profiler trace serialization is timed.
"""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .evaluate_single_gpu import load_model


def benchmark(model, inputs, warmup=5, repeat=11):
    def forward():
        return model(input_ids=inputs, use_cache=False).logits

    with torch.inference_mode():
        for _ in range(warmup):
            del_result = forward()
            del del_result
        torch.cuda.synchronize()
        samples = []
        for _ in range(repeat):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            result = forward()
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end))
            del result

        records = defaultdict(list)
        restores = []

        def wrap(fn, name):
            def measured(*args, **kwargs):
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                result = fn(*args, **kwargs)
                b.record()
                records[name].append((a, b))
                return result
            return measured

        def method(obj, attr, name):
            if attr == 'forward' and getattr(obj, '_compiled_call_impl', None) is not None:
                attr = '_compiled_call_impl'
            original = getattr(obj, attr)
            restores.append((obj, attr, original))
            setattr(obj, attr, wrap(original, name))

        # trunk calls several forwards directly, bypassing Module hooks.
        from .models.qwen3_8_moe import modeling_qwen3_8_moe as modeling
        method(modeling, 'qsa_attention', 'full_attention')
        from . import fused_precision
        method(fused_precision, '_fp8_attention', 'full_attention')
        for name, module in model.named_modules():
            if name.endswith('.mlp') and name.startswith('model.layers.'):
                method(module, 'forward', 'ffn')
            elif name.endswith('.linear_attn'):
                method(module, 'forward', 'gdn')
            elif name == 'lm_head':
                method(module, 'forward', 'lm_head')
            elif name.endswith(('input_layernorm', 'post_attention_layernorm')) or name == 'model.norm':
                method(module, 'forward', 'trunk_norms')
            elif '.linear_attn.' in name and isinstance(module, torch.nn.Linear):
                method(module, 'forward', 'gdn_projections_nested')
        if model.model.ngram is not None:
            method(model.model.ngram, 'delta', 'ngram')
        instrumented = []
        try:
            for _ in range(3):
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                result = forward()
                b.record()
                torch.cuda.synchronize()
                instrumented.append(a.elapsed_time(b))
                del result
            components = {name: sum(a.elapsed_time(b) for a, b in events)/3
                          for name, events in records.items()}
        finally:
            for obj, attr, original in reversed(restores):
                setattr(obj, attr, original)
    return dict(median_ms=statistics.median(samples), samples_ms=samples,
                instrumented_ms=statistics.median(instrumented), components_ms=components,
                attribution_note='Separate instrumented pass; gdn_projections_nested is included in gdn.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-path', required=True)
    p.add_argument('--packed-cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--sequence-length', type=int, default=4096)
    p.add_argument('--repeat', type=int, default=11)
    p.add_argument('--backend', choices=('baseline', 'bf16_fused', 'fp8_fused'), default='baseline')
    p.add_argument('--calibration', type=Path)
    p.add_argument('--calibration-data', type=Path)
    p.add_argument('--fused-trunk', action='store_true')
    p.add_argument('--packed-gdn', action='store_true')
    p.add_argument('--gdn-chunk-size', type=int, choices=(16, 32, 64), default=64)
    p.add_argument('--projection-scope', choices=('none', 'gdn', 'all'), default='none')
    args = p.parse_args()
    if args.batch_size < 1 or args.sequence_length < 1 or args.repeat < 1:
        p.error('Batch, sequence and repeat counts must be positive')
    if torch.cuda.device_count() != 1:
        p.error('Expose exactly one CUDA GPU')
    if args.output.exists():
        p.error('output already exists')
    torch.cuda.set_device(0)
    model = load_model(args.model_path, torch.device('cuda'))
    calibration = None
    if args.backend == 'fp8_fused':
        from .fused_precision import calibrate_ffn
        if args.calibration and args.calibration.exists():
            calibration = json.loads(args.calibration.read_text())
        elif args.calibration_data:
            train = np.memmap(args.calibration_data, dtype=np.int32, mode='r')
            def batches():
                # Fixed training-only prefix; no evaluation data is used for scales.
                for i in range(4):
                    start = i*args.batch_size*args.sequence_length
                    yield torch.from_numpy(np.array(train[start:start+args.batch_size*args.sequence_length],
                        dtype=np.int64).reshape(args.batch_size, args.sequence_length)).cuda()
            calibration = calibrate_ffn(model, batches())
            calibration['data_path'] = str(args.calibration_data.resolve())
            if args.calibration:
                args.calibration.parent.mkdir(parents=True, exist_ok=True)
                args.calibration.write_text(json.dumps(calibration, indent=2)+'\n')
        else:
            p.error('FP8 requires --calibration or --calibration-data')
    if args.backend != 'baseline':
        from .fused_precision import install_fused_ffn
        print(install_fused_ffn(model, args.backend, calibration), flush=True)
    if args.fused_trunk:
        from .fused_precision import install_fused_trunk
        install_fused_trunk(model)
    if args.projection_scope != 'none':
        if args.backend != 'fp8_fused' or not args.fused_trunk:
            p.error('FP8 projections require --backend fp8_fused --fused-trunk')
        from .fused_precision import install_fp8_projections
        install_fp8_projections(model, calibration, args.projection_scope)
    if args.packed_gdn:
        from .fused_precision import install_packed_gdn
        install_packed_gdn(model, args.gdn_chunk_size)
    tokens = np.memmap(args.packed_cache/'tokens.bin', dtype=np.int32, mode='r')
    n = args.batch_size*args.sequence_length
    inputs = torch.from_numpy(np.array(tokens[:n], dtype=np.int64).reshape(
        args.batch_size, args.sequence_length)).cuda()
    result = benchmark(model, inputs, repeat=args.repeat)
    result.update(model_path=args.model_path, shape=list(inputs.shape),
                  device=torch.cuda.get_device_name(), torch=torch.__version__, backend=args.backend,
                  fused_trunk=args.fused_trunk, packed_gdn=args.packed_gdn,
                  gdn_chunk_size=args.gdn_chunk_size, projection_scope=args.projection_scope)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
