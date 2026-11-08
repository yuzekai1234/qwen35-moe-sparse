"""CPU-only parent; capacity probes and complete evaluations use fresh CUDA children."""
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from capacity_cache import prepare_real_window
from entry_defaults import python_executable


def largest_fitting(probe,upper_bound):
    if not probe(1):
        raise RuntimeError('Model does not fit even batch1')
    low=1
    # A full BF16-logit tensor alone exceeds available memory at this bound.
    high=max(2,upper_bound+1)
    if probe(high):
        raise RuntimeError('Logits-only theoretical bound unexpectedly fits; refuse an unbounded claim')
    while high-low>1:
        mid=(low+high)//2
        if probe(mid):low=mid
        else:high=mid
    return low


def full_backoff(run,batch):
    while batch>0:
        status=run(batch)
        if status=='ok':return batch
        if status!='oom':raise RuntimeError('Complete evaluation failed for a non-OOM reason')
        batch-=1
    raise RuntimeError('No complete-evaluation batch fits')


def visible_gpu_memory():
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','0').split(',')[0].strip()
    if visible in ('','-1'):raise RuntimeError('No visible CUDA GPU')
    value=subprocess.check_output(['nvidia-smi','--id='+visible,'--query-gpu=memory.free',
        '--format=csv,noheader,nounits'],text=True).strip()
    return int(float(value)*1024**2),visible


def run_automatic(args):
    if os.environ.get('WORLD_SIZE','1')!='1':
        raise ValueError('Automatic maximum-batch qualification supports one GPU only')
    source=Path(__file__).resolve().parent
    output=args.output.resolve();output.parent.mkdir(parents=True,exist_ok=True)
    config=json.loads((Path(args.model_path)/'config.json').read_text())
    config=config.get('text_config',config)
    free_bytes,visible=visible_gpu_memory()
    upper=free_bytes//(args.sequence_length*int(config['vocab_size'])*2)
    if upper<1:raise RuntimeError('Insufficient memory even for one BF16-logit batch')
    record=dict(scope='fresh_process_maximum_stable_batch',visible_gpu=visible,
                free_bytes_before_CUDA_parent_context=free_bytes,
                logits_only_upper_bound=upper,no_safety_margin=True,probes=[],full_attempts=[])
    status_path=output.with_name('auto-batch-metadata.json')
    def save():status_path.write_text(json.dumps(record,indent=2)+'\n')
    save()
    with tempfile.TemporaryDirectory(prefix='.capacity-',dir=output.parent) as temporary:
        temporary=Path(temporary)
        cache=prepare_real_window(args.dataset_path,args.model_path,upper+1,args.sequence_length,temporary/'cache')
        torch_module=sys.modules.get('torch')
        initialized=bool(torch_module and torch_module.cuda.is_initialized())
        record['CUDA_parent_initialized_after_CPU_cache']=initialized
        if initialized:
            raise RuntimeError('Automatic-batch parent must not hold a CUDA context')
        record['capacity_cache_metadata']=json.loads((cache/'metadata.json').read_text())
        def probe(batch):
            path=temporary/f'capacity-b{batch}.json'
            command=[python_executable(),'-u',str(source/'capacity_worker.py'),'--model-path',args.model_path,
                '--capacity-cache',str(cache),'--batch-size',str(batch),'--sequence-length',str(args.sequence_length),
                '--result',str(path)]
            with (temporary/f'capacity-b{batch}.log').open('w') as log:
                completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
            result=json.loads(path.read_text()) if path.exists() else dict(status='error',error='Capacity child did not produce status')
            record['probes'].append(result);save()
            print('AUTO_CAPACITY',batch,result['status'],flush=True)
            if result['status']=='error' or completed.returncode not in (0,42):
                raise RuntimeError('Capacity child failed: '+result.get('error','unknown'))
            return result['status']=='ok'
        screened=largest_fitting(probe,int(upper))
        record['screened_max_batch']=screened;save()
        def run_full(batch):
            directory=temporary/f'complete-b{batch}';directory.mkdir()
            command=[python_executable(),'-u',str(source/'eval.py'),'--dataset-path',str(args.dataset_path),
                '--model-path',args.model_path,'--batch-size',str(batch),
                '--sequence-length',str(args.sequence_length),'--output',str(directory/'result.json')]
            lines=[]
            with (directory/'evaluation.log').open('w') as log:
                child=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
                for line in child.stdout:
                    lines.append(line);log.write(line);sys.stdout.write(line);sys.stdout.flush()
                code=child.wait()
            message=''.join(lines)
            oom=any(text in message for text in ['CUDA out of memory','CUDA error: out of memory',
                'torch.OutOfMemoryError:','torch.cuda.OutOfMemoryError:'])
            status='ok' if code==0 else ('oom' if oom else 'error')
            record['full_attempts'].append(dict(batch=batch,returncode=code,status=status));save()
            if status=='ok':
                if not (directory/'result.json').exists():raise RuntimeError('Complete child exited0 without result')
                shutil.copy2(directory/'result.json',output)
                shutil.copy2(directory/'run-metadata.json',output.with_name('run-metadata.json'))
            return status
        stable=full_backoff(run_full,screened)
        record.update(status='finished',complete_A_verified_batch=stable);save()
    return stable
