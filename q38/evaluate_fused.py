"""Calibrate on training data or evaluate complete fused forward on full A data."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from .compare_precision import sha256
from .data import tokenizer_sha256
from .evaluate import active_parameter_count
from .evaluate_single_gpu import evaluate_model, load_model
from .fused_precision import calibrate_ffn, install_fused_ffn, install_fused_trunk


def validate_packed_cache(cache, dataset_sha, tokenizer_sha, pad_token_id):
    """Validate full-corpus identity and both streams before scoring/timing."""
    metadata = json.loads((cache/'metadata.json').read_text())
    expected = dict(dataset_sha256=dataset_sha, tokenizer_sha256=tokenizer_sha,
                    pad_token_id=pad_token_id)
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f'Packed cache identity mismatch: {key}')
    for name, key in (('tokens.bin', 'tokens_sha256'), ('is_text.bin', 'is_text_sha256')):
        if sha256(cache/name) != metadata.get(key):
            raise ValueError(f'Packed cache checksum mismatch: {name}')
    length = (cache/'is_text.bin').stat().st_size
    if length < 2 or (cache/'tokens.bin').stat().st_size != length*4:
        raise ValueError('Packed cache stream lengths disagree')
    mask = np.memmap(cache/'is_text.bin', dtype=np.bool_, mode='r')
    if mask[0] or int(mask.sum()) != metadata['total_tokens'] or metadata['total_bytes'] <= 0:
        raise ValueError('Packed cache scoring metadata disagrees with the mask')
    return {key: metadata[key] for key in ('total_tokens', 'total_bytes', 'pad_token_id')}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-path', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--calibrate-only', action='store_true')
    p.add_argument('--calibration-data', type=Path)
    p.add_argument('--calibration-batches', type=int, default=4)
    p.add_argument('--calibration', type=Path)
    p.add_argument('--dataset-path', type=Path)
    p.add_argument('--packed-cache', type=Path)
    p.add_argument('--backend', choices=('baseline', 'bf16_fused', 'fp8_fused'), default='baseline')
    p.add_argument('--projection-scope', choices=('none', 'gdn', 'all'), default='none')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--sequence-length', type=int, default=4096)
    p.add_argument('--progress-every', type=int, default=100)
    a = p.parse_args()
    if a.output.exists():
        p.error('Output exists; choose a fresh path')
    if a.batch_size < 1 or a.sequence_length < 1 or a.calibration_batches < 1:
        p.error('Batch, sequence and calibration counts must be positive')
    if torch.cuda.device_count() != 1:
        p.error('Expose exactly one CUDA GPU')
    torch.cuda.set_device(0)
    model = load_model(str(a.model_path), torch.device('cuda'))
    identity = dict(model_path=str(a.model_path.resolve()),
        model_files_sha256={f.name: sha256(f) for f in a.model_path.glob('*.safetensors')},
        config_sha256=sha256(a.model_path/'config.json'),
        tokenizer_sha256=tokenizer_sha256(a.model_path),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        gpu=torch.cuda.get_device_name(), torch=torch.__version__,
        implementation_sha256={name: sha256(Path(__file__).parent/name) for name in (
            'fused_precision.py', 'inference.py', 'evaluate_fused.py', 'evaluate_single_gpu.py',
            'models/qwen3_8_moe/modeling_qwen3_8_moe.py')})
    if a.calibrate_only:
        if not a.calibration_data:
            p.error('--calibrate-only requires --calibration-data')
        train = np.memmap(a.calibration_data, dtype=np.int32, mode='r')
        n = a.batch_size*a.sequence_length
        if len(train) < n*a.calibration_batches:
            p.error('Calibration stream is too short')
        def batches():
            for i in range(a.calibration_batches):
                yield torch.from_numpy(np.array(train[i*n:(i+1)*n], dtype=np.int64).reshape(
                    a.batch_size, a.sequence_length)).cuda()
        result = calibrate_ffn(model, batches())
        result.update(identity, calibration_data=str(a.calibration_data.resolve()),
                      calibration_data_bytes=a.calibration_data.stat().st_size,
                      calibration_tokens=n*a.calibration_batches,
                      calibration_source='training_only')
    else:
        if not a.dataset_path or not a.packed_cache:
            p.error('Evaluation requires --dataset-path and --packed-cache')
        data_sha = sha256(a.dataset_path)
        metadata = validate_packed_cache(a.packed_cache, data_sha, identity['tokenizer_sha256'],
                                        model.config.pad_token_id)
        calibration = None
        if a.backend == 'fp8_fused':
            calibration = (json.loads(a.calibration.read_text()) if a.calibration else
                           getattr(model.config, 'fp8_ffn_calibration', None))
            if calibration is None:
                p.error('FP8 needs a training calibration file or trained FP8 config')
        if a.backend != 'baseline':
            install_fused_ffn(model, a.backend, calibration)
            install_fused_trunk(model)
        if a.projection_scope != 'none':
            if a.backend != 'fp8_fused':
                p.error('Projection quantization requires --backend fp8_fused')
            from .fused_precision import install_fp8_projections
            install_fp8_projections(model, calibration, a.projection_scope)
        tokens = np.memmap(a.packed_cache/'tokens.bin', dtype=np.int32, mode='r')
        mask = np.memmap(a.packed_cache/'is_text.bin', dtype=np.bool_, mode='r')
        result = dict(identity, backend=a.backend, projection_scope=a.projection_scope, dataset_sha256=data_sha,
                      packed_cache=str(a.packed_cache.resolve()),
                      calibration=calibration, evaluation_scope='complete_A',
                      timing_scope='complete_forward_including_full_vocabulary_logits',
                      timing_repeats=1,
                      training_precision=getattr(model.config, 'ffn_training_precision', 'bf16'))
        result.update(evaluate_model(model, tokens, mask, metadata, a.batch_size,
            a.sequence_length, torch.device('cuda'), active_parameter_count(model),
            measure_time=True, warmup_steps=5, progress_every=a.progress_every))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k not in (
        'calibration', 'layers', 'implementation_sha256')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
