#!/usr/bin/env python3
"""Frozen C1 BF16 control + FP8 adaptation, followed by full A evaluation.

Run under nohup/setsid. A short --qualification run checks real training,
checkpoint transition/resume and inference; it does not claim full-budget BPB.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time


def write_json(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--train-gpus', required=True, help='Exactly four comma-separated GPU UUIDs')
    p.add_argument('--eval-gpu', required=True)
    p.add_argument('--qualification', action='store_true')
    a = p.parse_args()
    gpu = a.train_gpus.split(',')
    if len(gpu) != 4 or len(set(gpu+[a.eval_gpu])) != 5:
        p.error('Four distinct training GPUs and one separate evaluation GPU are required')
    source = Path(__file__).resolve().parents[1]
    out = a.output.resolve()
    if out.exists():
        p.error('Output exists; use a new directory')
    out.mkdir(parents=True)
    snapshot = out/'source'
    snapshot.mkdir()
    for name in ('q38', 'scripts', 'configs', 'tests'):
        shutil.copytree(source/name, snapshot/name, ignore=shutil.ignore_patterns('__pycache__'))
    python = source/'.venv_q35/bin/python'
    if not python.exists():
        python = Path(sys.executable)
    config = json.loads((snapshot/'configs/baseline.json').read_text())
    prefix_steps, final_steps = (2, 3) if a.qualification else (5500, 6206)
    manifest = {str(f.relative_to(snapshot)): hashlib.sha256(f.read_bytes()).hexdigest()
                for f in snapshot.rglob('*') if f.is_file()}
    write_json(out/'protocol.json', dict(
        qualification=a.qualification, training_gpus=gpu, evaluation_gpu=a.eval_gpu,
        source_root=str(source), snapshot=str(snapshot), source_sha256=manifest,
        shared_prefix_steps=prefix_steps, final_steps=final_steps,
        logical_train_budget_per_completed_branch=config['train']['batch_tokens']*final_steps*
            config['accounting']['train_flops_per_token'],
        fp8_scope='main FFN + GDN qkv/z/out + full-attention Q/K/V/O; E4M3 forward, BF16 backward, FP32 masters',
        comparison='identical BF16 prefix, then BF16 and FP8 tails at identical global batches/LR',
        inference='baseline, fused BF16, FFN FP8 PTQ, all-projection FP8 PTQ, all-projection FP8 adapted',
        selection='speed first; report BPB tradeoff without a fixed 0.001 cutoff',
        scaling='fixed training-only calibration; QAT inherits prefix calibration',
        timing='same evaluation GPU; complete forward including full vocabulary logits',
        model=config['model'], train=config['train']))

    def status(stage, **extra):
        write_json(out/'status.json', dict(stage=stage, updated_utc=time.strftime(
            '%Y-%m-%dT%H:%M:%SZ', time.gmtime()), controller_pid=os.getpid(), **extra))
        print(stage, extra, flush=True)

    def launch(name, module, args, devices):
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=','.join(devices), PYTHONPATH=str(snapshot),
            OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false',
            TORCHINDUCTOR_COMPILE_THREADS='2', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
            TRITON_LIBCUDA_PATH=str(source/'libcuda'), LIBRARY_PATH=str(source/'libcuda'))
        command = [str(python)]
        if len(devices) > 1:
            command += ['-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={len(devices)}']
        command += ['-m', module, *map(str, args)]
        log = open(out/(name+'.log'), 'a')
        process = subprocess.Popen(command, cwd=snapshot, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        log.close()
        write_json(out/(name+'.process.json'), dict(pid=process.pid, command=command, devices=devices))
        return process

    def wait(process, name):
        code = process.wait()
        if code:
            raise RuntimeError(f'{name} exited {code}; see {out/(name+".log")}')

    baseline, adapted = out/'baseline', out/'fp8_adapted'
    common = ['--preset', 'baseline', '--tensorboard', '0', '--log_every', '50']
    if a.qualification:
        common += ['--eval_tokens', '8193', '--final_eval_tokens', '8193', '--log_every', '1']
    try:
        status('training_shared_prefix')
        wait(launch('prefix', 'q38.train', [*common, '--run_dir', baseline,
            '--max_steps', prefix_steps], gpu), 'prefix')
        status('calibrating_training_prefix')
        calibration = out/'prefix_calibration.json'
        wait(launch('prefix_calibration', 'q38.evaluate_fused', ['--model-path', baseline/'hf',
            '--calibrate-only', '--calibration-data', config['train']['data'],
            '--output', calibration], [a.eval_gpu]), 'prefix_calibration')
        adapted.mkdir()
        for name in ('ckpt.pt', 'run_config.json'):
            shutil.copy2(baseline/name, adapted/name)
        status('training_matched_tails')
        base_process = launch('baseline_tail', 'q38.train', [*common, '--run_dir', baseline,
            '--max_steps', final_steps], gpu[:2])
        fp8_process = launch('fp8_tail', 'q38.train', [*common, '--run_dir', adapted,
            '--max_steps', final_steps, '--ffn_precision', 'fp8', '--ffn_calibration', calibration,
            '--projection_precision', 'fp8', '--precision_transition', '1'], gpu[2:])
        # Wait for both even if one fails, preserving the useful completed branch.
        codes = [base_process.wait(), fp8_process.wait()]
        if any(codes):
            raise RuntimeError(f'Tail exit codes {codes}; inspect baseline_tail.log / fp8_tail.log')
        status('calibrating_completed_bf16_control')
        final_calibration = out/'final_calibration.json'
        wait(launch('final_calibration', 'q38.evaluate_fused', ['--model-path', baseline/'hf',
            '--calibrate-only', '--calibration-data', config['train']['data'],
            '--output', final_calibration], [a.eval_gpu]), 'final_calibration')
        packed = '/home/shared/qwen35_moe_sparse/experiments/q38-4090-evaluation-20261028/packed'
        variants = [('baseline', baseline, 'baseline', None, 'none'),
                    ('bf16_fused', baseline, 'bf16_fused', None, 'none'),
                    ('fp8_ffn_ptq', baseline, 'fp8_fused', final_calibration, 'none'),
                    ('fp8_all_ptq', baseline, 'fp8_fused', final_calibration, 'all'),
                    ('fp8_adapted', adapted, 'fp8_fused', calibration, 'all')]
        results = []
        for name, run, backend, cal, scope in variants:
            status('benchmarking_'+name)
            args = ['--model-path', run/'hf', '--packed-cache', packed,
                    '--backend', backend, '--projection-scope', scope,
                    '--output', out/(name+'.benchmark.json')]
            if backend != 'baseline':
                args += ['--fused-trunk']
            if cal:
                args += ['--calibration', cal]
            wait(launch(name+'_benchmark', 'q38.benchmark_blocks', args, [a.eval_gpu]), name+'_benchmark')
            if a.qualification:
                continue
            status('evaluating_full_A_'+name)
            args = ['--model-path', run/'hf', '--packed-cache', packed,
                    '--dataset-path', '/home/shared/data/dolma3_mix_100M_a.jsonl',
                    '--backend', backend, '--projection-scope', scope,
                    '--output', out/(name+'.evaluation.json')]
            if cal:
                args += ['--calibration', cal]
            wait(launch(name+'_evaluation', 'q38.evaluate_fused', args, [a.eval_gpu]), name+'_evaluation')
            results.append(json.loads((out/(name+'.evaluation.json')).read_text()))
        if not a.qualification:
            # Reverse the ordering between rounds to expose thermal/order bias.
            # These short complete forwards supplement, not replace, full-A time.
            repeated = {name: [] for name, *_ in variants}
            for trial in range(3):
                order = variants if trial % 2 == 0 else list(reversed(variants))
                for name, run, backend, cal, scope in order:
                    status(f'interleaved_timing_{trial}_{name}')
                    target = out/f'{name}.timing{trial}.json'
                    args = ['--model-path', run/'hf', '--packed-cache', packed,
                        '--backend', backend, '--projection-scope', scope, '--output', target]
                    if backend != 'baseline':
                        args += ['--fused-trunk']
                    if cal:
                        args += ['--calibration', cal]
                    wait(launch(f'{name}_timing{trial}', 'q38.benchmark_blocks', args,
                        [a.eval_gpu]), f'{name}_timing{trial}')
                    repeated[name].append(json.loads(target.read_text())['median_ms'])
            write_json(out/'interleaved_timing.json', {name: dict(medians_ms=values,
                median_ms=statistics.median(values)) for name, values in repeated.items()})
        if results:
            reference = results[0]
            for result in results:
                result['delta_bpb_vs_baseline'] = result['bpb']-reference['bpb']
                result['speedup_vs_baseline'] = reference['evaltime']/result['evaltime']
            write_json(out/'results.json', results)
        status('qualification_complete' if a.qualification else 'complete', results=len(results))
    except BaseException as exc:
        status('failed', error=str(exc))
        raise


if __name__ == '__main__':
    main()
