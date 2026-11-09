"""Run one claimed task with saved logs, heartbeats, physical GPU checks and kernel locks."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback
from common import (HERE, DEFAULT_WORKSPACE, Lease, claim, digest, inventory, read, record,
                    show_status, source_hash, state_root, write)


def environment(workspace):
    package = Path(workspace)/'c1-combined-papers-bpb0.99090686-20261108'
    env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONNOUSERSITE='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
               FLA_TILELANG='1', OMP_NUM_THREADS='4',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', CUDA_DEVICE_ORDER='PCI_BUS_ID')
    env['PYTHONPATH'] = ':'.join([str(HERE/'runtime'), str(package/'compat-source')])
    env['CUDA_HOME'] = os.environ.get('CUDA_HOME', '/usr/local/cuda-12.8')
    env['PATH'] = env['CUDA_HOME']+'/bin:'+env['PATH']
    return env


def run_child(command, cwd, env, logfile, heartbeat):
    """Forward a stop only to this task's child process group."""
    with Path(logfile).open('a') as out:
        child = subprocess.Popen(command, cwd=cwd, env=env, stdout=out, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        prior = {}
        def stop(signum, frame):
            raise KeyboardInterrupt('Stop requested')
        for sig in (signal.SIGTERM, signal.SIGINT):
            prior[sig] = signal.signal(sig, stop)
        try:
            while child.poll() is None:
                heartbeat(child.pid)
                time.sleep(20)
            if child.returncode:
                raise RuntimeError(f'Child exited {child.returncode}; inspect {logfile}')
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            for sig, handler in prior.items():
                signal.signal(sig, handler)


def acquire(a, task):
    while True:
        try:
            return Lease(a.workspace, a.gpus, a.owner, task).acquire()
        except (RuntimeError, BlockingIOError) as e:
            if not a.wait:
                raise
            record(a.workspace, task, a.owner, 'waiting', dict(reason=str(e), gpus=a.gpus))
            time.sleep(30)


def run_cpu_preflight(a, root):
    # Imports such as FLA can probe CUDA when GPUs are visible. Keep every CPU
    # check in a separate hidden-GPU interpreter; the controller must stay
    # CUDA-free until it owns the physical GPU leases.
    env = environment(a.workspace)
    env['CUDA_VISIBLE_DEVICES'] = ''
    command = [sys.executable, str(HERE/'preflight.py'), '--workspace', str(a.workspace),
               '--run-id', a.run_id, '--level', a.level]
    with (root/a.level/'preflight-launch.log').open('a') as stream:
        subprocess.run(command, cwd=HERE, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)


def train(a, task):
    root = state_root(a.workspace)/'runs'/a.run_id
    run_cpu_preflight(a, root)
    row = next(r for r in read(root/'manifest.json')['levels'] if r['level'] == a.level)
    if len(a.gpus) != row['world']:
        raise ValueError(f'Prepared native world={row["world"]}; supply that many GPUs, or prepare another run id')
    output = root/a.level
    config = read(output/'config.json')
    if (output/'train').exists() and any((output/'train').iterdir()) and not a.resume:
        raise RuntimeError('Training output exists; --resume is required to preserve its history')
    env = environment(a.workspace)
    lease = acquire(a, task)
    try:
        env['CUDA_VISIBLE_DEVICES'] = ','.join(lease.gpus[g]['uuid'] for g in a.gpus)
        env['TRITON_CACHE_DIR'] = str(state_root(a.workspace)/'cache'/a.run_id/a.level/'triton')
        env['TORCHINDUCTOR_CACHE_DIR'] = str(state_root(a.workspace)/'cache'/a.run_id/a.level/'inductor')
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                   '--nproc_per_node='+str(len(a.gpus)), '-m', 'q38.paper_train',
                   '--config', str(output/'config.json')]
        identity = dict(source_hash=source_hash(), config_sha256=digest(output/'config.json'),
                        world=len(a.gpus), gpu_uuids=env['CUDA_VISIBLE_DEVICES'], command=command)
        previous = output/'launch.json'
        if previous.exists() and read(previous) != identity:
            raise RuntimeError('Resume identity changed; source/config/world/GPU mapping must be inspected')
        write(previous, identity)
        def heartbeat(pid):
            detail = dict(child_pid=pid, command=command, log=str(output/'training.log'), gpus=a.gpus)
            with (output/'training.log').open('rb') as logfile:
                logfile.seek(max(0, logfile.seek(0, 2)-12000))
                text = logfile.read().decode(errors='replace')
            rows = re.findall(r'step (\d+)/(\d+).*? (\d+) tok/s', text)
            if rows:
                step, total, tps = map(int, rows[-1])
                detail.update(step=step, total_steps=total, tokens_per_second=tps,
                              estimated_remaining_seconds=(total-step-1)*config['train']['batch_tokens']/max(tps, 1))
            record(a.workspace, task, a.owner, 'training', detail)
        run_child(command, HERE/'runtime', env, output/'training.log', heartbeat)
        summary = read(output/'train/summary.json')
        if not (summary['completed_budget'] and summary['tokens_trained'] == config['train']['tokens']
                and summary['train_flops'] == config['accounting']['actual_train_flops']):
            raise RuntimeError('Training did not finish the prepared budget')
        sys.path.insert(0, str(HERE/'runtime'))
        from prepare_fp32_controls import ensure_fp32_controls
        controls = ensure_fp32_controls(config, output/'train/hf')
        write(output/'training-complete.json', dict(summary=summary, fp32_controls=controls,
                                                  source_hash=source_hash()))
        record(a.workspace, task, a.owner, 'complete', dict(training_summary=summary,
               hf=str(output/'train/hf'), next='Claim eval separately; complete training is not an A-score'))
    finally:
        lease.release()


def evaluate(a, task):
    if len(a.gpus) != 1:
        raise ValueError('Evaluation requires one GPU')
    root = state_root(a.workspace)/'runs'/a.run_id
    manifest = read(root/'manifest.json')
    if manifest['source_hash'] != source_hash():
        raise RuntimeError('Prepared source differs')
    output = root/a.level
    model = output/'train/hf'
    if a.model:
        model = a.model.resolve()
    if not (model/'config.json').is_file() or not (model/'paper-control-fp32.safetensors').is_file():
        raise RuntimeError('A complete HF export and FP32 control sidecar are required')
    from prepare import hf_config
    prepared = read(output/'config.json')
    actual_hf = read(model/'config.json')
    expected_hf = hf_config(prepared['model'], prepared['train']['seq'])
    for key in ('hidden_size', 'num_hidden_layers', 'num_experts', 'num_experts_per_tok',
                'moe_intermediate_size', 'longconv_layers', 'longconv_channels', 'longconv_taps',
                'paper_engram', 'paper_loss_free', 'paper_dwa'):
        if actual_hf.get(key) != expected_hf.get(key):
            raise RuntimeError('Export differs from this level configuration: '+key)
    summary = read(model.parent/'summary.json')
    if not summary['completed_budget'] or summary['train_flops'] != prepared['accounting']['actual_train_flops']:
        raise RuntimeError('Export training budget differs from this level')
    result_dir = output/'evaluations'/a.eval_tag
    if result_dir.exists() and any(result_dir.iterdir()):
        raise RuntimeError('Evaluation output is not empty; use a new --eval-tag')
    env = environment(a.workspace)
    env.update({key: str(value) for key, value in read(HERE/'runtime/inference/release-defaults.json').items()})
    env.update(Q38_PAPER_NEURAL_MODEL=str(model), Q38_PYTHON=sys.executable, EVAL_GPUS='1',
               EVAL_SEQ=str(a.eval_seq), EVAL_BATCH_SIZE=a.batch, OMP_NUM_THREADS='1')
    for name in ('Q38_PAPER_SPEC', 'Q38_CONTINUOUS_SPEC', 'Q38_JOINT_SPEC', 'Q38_COUNT_BUNDLE'):
        env.pop(name, None)
    profile = dict(mode='raw', sequence_length=a.eval_seq, precision='FP8 trunk/BF16 logits',
                   single_gpu=True, batch_size=a.batch)
    if a.hybrid_spec:
        spec = read(a.hybrid_spec)
        if Path(spec['neural_model']).resolve() != model:
            raise RuntimeError('Hybrid calibration belongs to another neural model')
        if not Path(spec['root']).is_dir() or not Path(spec['neural_config']).is_file():
            raise RuntimeError('Hybrid memory/reference missing')
        env['Q38_PAPER_SPEC'] = str(a.hybrid_spec.resolve())
        profile['mode'] = 'hybrid'
    dataset = Path(a.workspace)/'c1-combined-papers-data-20261108/evaluation-a-v1/dolma3_mix_100M_a.jsonl'
    command = ['bash', str(HERE/'runtime/inference/eval.sh'), str(dataset), str(model)]
    lease = acquire(a, task)
    try:
        env['CUDA_VISIBLE_DEVICES'] = lease.gpus[a.gpus[0]]['uuid']
        result_dir.mkdir(parents=True, exist_ok=False)
        write(result_dir/'launch.json', dict(command=command, model=str(model), profile=profile,
                                            source_hash=source_hash(), gpu=lease.gpus))
        def heartbeat(pid):
            record(a.workspace, task, a.owner, 'evaluating', dict(child_pid=pid, log=str(result_dir/'eval.log'), profile=profile))
        run_child(command, result_dir, env, result_dir/'eval.log', heartbeat)
        result = read(result_dir/'result.json')
        metadata = read(result_dir/'run-metadata.json')
        if result['bpb'] <= 0 or result['evaltime'] <= 0:
            raise RuntimeError('Invalid metrics')
        measured_tokens = metadata.get('scored_tokens', metadata.get('total_scored_tokens'))
        if measured_tokens != 120142976:
            raise RuntimeError('Not the complete expected A set: '+str(measured_tokens))
        evidence = dict(result=result, metadata=str(result_dir/'run-metadata.json'), profile=profile,
                        train_flops=read(output/'config.json')['accounting']['actual_train_flops'],
                        source_hash=source_hash(), complete_A=True)
        write(result_dir/'campaign-result.json', evidence)
        record(a.workspace, task, a.owner, 'complete', evidence)
    finally:
        lease.release()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['train', 'eval', 'status'])
    p.add_argument('--workspace', type=Path, default=DEFAULT_WORKSPACE)
    p.add_argument('--run-id', default='combined-v5')
    p.add_argument('--level', choices=['c1', 'c2', 'c3', 'c4'])
    p.add_argument('--owner')
    p.add_argument('--gpus', type=lambda s: list(map(int, s.split(','))))
    p.add_argument('--wait', action='store_true', help='Explicitly wait for empty GPUs; never preempt')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--eval-tag', default='raw-s10240-auto')
    p.add_argument('--eval-seq', type=int, default=10240)
    p.add_argument('--batch', default='auto')
    p.add_argument('--model', type=Path, help='Explicit existing export, e.g. the frozen C1 baseline')
    p.add_argument('--hybrid-spec', type=Path, help='Calibration from this exact candidate, frozen before A evaluation')
    a = p.parse_args()
    if a.action == 'status':
        print(json.dumps(dict(tasks=show_status(a.workspace), gpus=inventory()), ensure_ascii=False, indent=2))
        return
    if not a.level or not a.owner or not a.gpus:
        p.error('--level, --owner and --gpus are required')
    if not all(re.fullmatch(r'[A-Za-z0-9_.-]+', x) for x in (a.run_id, a.owner, a.eval_tag)):
        p.error('Identifiers may contain letters, numbers, underscores, dots and hyphens only')
    task = ':'.join([a.run_id, a.level, a.action] + ([a.eval_tag] if a.action == 'eval' else []))
    claim(a.workspace, task, a.owner, a.resume)
    try:
        {'train': train, 'eval': evaluate}[a.action](a, task)
    except BaseException as e:
        record(a.workspace, task, a.owner, 'interrupted' if isinstance(e, KeyboardInterrupt) else 'failed',
               dict(error=repr(e), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
