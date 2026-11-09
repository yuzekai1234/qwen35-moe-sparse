"""Wait for a committed checkpoint, preserve four-rank state, then launch native eight-rank resume."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import re
import sqlite3
import subprocess
import sys
import time
from common import HERE,DEFAULT_WORKSPACE,Lease,claim,record,read,state_root,write,inventory


def task_row(workspace,task):
    c=sqlite3.connect(state_root(workspace)/'coordination.sqlite');c.row_factory=sqlite3.Row
    try:
        row=c.execute('SELECT * FROM tasks WHERE id=?',(task,)).fetchone()
        return dict(row) if row else None
    finally:c.close()


def alive(pid):
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False


def controller_identity(pid,source_run,workspace):
    args=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    args=[a.decode() for a in args if a]
    expected=str(state_root(workspace)/'runs'/source_run/'source/run.py')
    if expected not in args or '--run-id' not in args or args[args.index('--run-id')+1]!=source_run:
        raise RuntimeError('PID does not identify the expected training controller')
    fields=Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split()
    return int(fields[19])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,default=DEFAULT_WORKSPACE)
    p.add_argument('--source-run',default='combined-v5')
    p.add_argument('--target-run',default='combined-v6')
    p.add_argument('--owner',default='codex-gzy')
    p.add_argument('--checkpoint-step',type=int,default=2000)
    a=p.parse_args()
    if a.source_run==a.target_run or not all(re.fullmatch(r'[A-Za-z0-9_.-]+',x)
                                              for x in (a.source_run,a.target_run,a.owner)):
        p.error('Distinct valid source and target run identifiers are required')
    root=state_root(a.workspace)
    source=root/'runs'/a.source_run/'c4/train'
    target=root/'runs'/a.target_run/'c4'
    original_task=a.source_run+':c4:train'
    migration=a.target_run+':c4:expand'
    claim(a.workspace,migration,a.owner)
    extra=None;stopped=False
    state=dict(source_run=a.source_run,target_run=a.target_run,owner=a.owner,stage='waiting_for_checkpoint',
               supervisor_pid=os.getpid(),started=time.time(),desired_step=a.checkpoint_step)
    receipt=root/'C4_EXPAND_TO_EIGHT.json'
    def status(stage,**detail):
        state.update(stage=stage,updated=time.time(),**detail);write(receipt,state)
        record(a.workspace,migration,a.owner,'waiting' if stage.startswith('waiting') else 'running',state)
    try:
        while True:
            old=task_row(a.workspace,original_task)
            if not old or old['owner']!=a.owner or old['status']!='training' or not alive(old['pid']):
                raise RuntimeError('Original controller is no longer this owner\'s active training')
            if time.time()-old['updated']>90:raise RuntimeError('Source heartbeat is stale')
            identity=controller_identity(old['pid'],a.source_run,a.workspace)
            if state.get('source_pid_start_ticks',identity)!=identity:
                raise RuntimeError('Source PID identity changed')
            state['source_pid_start_ticks']=identity
            if extra is None:
                try:
                    extra=Lease(a.workspace,[4,5,6,7],a.owner,migration).acquire()
                except (RuntimeError,BlockingIOError) as e:
                    status('waiting_for_extra_GPUs',reason=str(e));time.sleep(30);continue
            required=[source/'ckpt.pt']+[source/f'{family}-rank{r}.pt' for r in range(4) for family in ('opt','longconv')]
            ready=all(f.exists() for f in required) and not list(source.glob('*.pt.tmp'))
            if ready:
                master_time=(source/'ckpt.pt').stat().st_mtime
                shard_times=[f.stat().st_mtime for f in required[1:]]
                ready=max(shard_times)<=master_time and master_time-min(shard_times)<180
            if ready:
                # Inspect only the atomically renamed, fully written master file.
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
                cmd=[sys.executable,'-c',
                     'import torch,sys,json; c=torch.load(sys.argv[1],map_location="cpu",weights_only=False,mmap=True); print(json.dumps({"step":c["step"],"world":c["world"],"tokens":c["tok_seen"]}))',str(source/'ckpt.pt')]
                metadata=json.loads(subprocess.check_output(cmd,text=True,env=env))
                if metadata['step']>=a.checkpoint_step:
                    if metadata['world']!=4:raise RuntimeError('Unexpected source native world')
                    break
            status('waiting_for_checkpoint',source_controller=old['pid'])
            time.sleep(20)
        status('stopping_original_at_saved_checkpoint',checkpoint=metadata,source_controller=old['pid'])
        if controller_identity(old['pid'],a.source_run,a.workspace)!=state['source_pid_start_ticks']:
            raise RuntimeError('Source controller identity changed before signal')
        # Send to our known controller only. Its handler stops its owned child group.
        os.kill(old['pid'],signal.SIGTERM);stopped=True
        deadline=time.time()+120
        while alive(old['pid']):
            if time.time()>deadline:raise RuntimeError('Original controller did not stop; no competing launch attempted')
            time.sleep(2)
        snapshot=root/'checkpoint-snapshots'/f'{a.source_run}-c4-step{metadata["step"]}'
        snapshot.mkdir(parents=True,exist_ok=False)
        for path in required+[source/'run_config.json',source/'training_metrics.csv']:
            if path.exists():os.link(path,snapshot/path.name)
        write(snapshot/'SNAPSHOT_IDENTITY.json',dict(step=metadata['step'],tokens=metadata['tokens'],
            original=str(source),files={p.name:dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
                                      for p in snapshot.iterdir() if p.is_file()}))
        status('resharding_saved_checkpoint',checkpoint=metadata,snapshot=str(snapshot))
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4')
        with (root/'controller-logs'/f'{a.target_run}-reshard.log').open('a') as log:
            subprocess.run([sys.executable,str(HERE/'reshard_checkpoint.py'),'--source',str(snapshot),
                            '--target',str(target/'train'),'--config',str(target/'config.json'),'--world','8'],
                           env=env,cwd=HERE,stdout=log,stderr=subprocess.STDOUT,check=True)
        proof=read(target/'train/RESHARD_PROOF.json')
        if not proof['passed']:raise RuntimeError('Checkpoint conversion proof failed')
        old_cache=root/'cache'/a.source_run/'c4';new_cache=root/'cache'/a.target_run/'c4'
        if old_cache.exists():
            for pth in old_cache.rglob('*'):
                if pth.is_file():
                    dest=new_cache/pth.relative_to(old_cache)
                    if not dest.exists():dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(pth,dest)
        status('checkpoint_expanded_verified',proof=proof)
        extra.release();extra=None
        session='br-codex-gzy-c4-w8'
        command=('cd '+str(HERE)+' && bash launch.sh train --run-id '+a.target_run+
                 ' --level c4 --owner '+a.owner+' --gpus 0,1,2,3,4,5,6,7 --resume >> '+
                 str(root/'controller-logs'/f'{a.target_run}-controller.log')+' 2>&1')
        subprocess.run(['tmux','new-session','-d','-s',session,command],check=True)
        status('eight_card_controller_launched',session=session,command=command)
        target_task=a.target_run+':c4:train';deadline=time.time()+1800
        while time.time()<deadline:
            new=task_row(a.workspace,target_task)
            if new and new['status'] in ('failed','interrupted'):raise RuntimeError('Eight-card controller failed: '+new['detail'])
            if new and new['status']=='training':
                detail=json.loads(new['detail'])
                if detail.get('step',-1)>metadata['step']:
                    state.update(stage='eight_card_resume_verified',updated=time.time(),training=detail)
                    write(receipt,state);record(a.workspace,migration,a.owner,'complete',state)
                    return
            time.sleep(20)
        raise RuntimeError('Eight-card first progress not observed within 30 minutes; inspect logs')
    except BaseException as e:
        state.update(stage='failed',error=repr(e),updated=time.time())
        write(receipt,state);record(a.workspace,migration,a.owner,'failed',state)
        # Never delete or overwrite a recovery checkpoint. If expansion failed
        # before the eight-card controller exists, the original four-card run
        # can resume from the preserved native checkpoint.
        new=task_row(a.workspace,a.target_run+':c4:train')
        safe_fallback=new is None
        if new and new['status'] in ('failed','interrupted'):
            deadline=time.time()+45
            while alive(new['pid']) and time.time()<deadline:time.sleep(2)
            safe_fallback=not alive(new['pid'])
        if stopped and safe_fallback:
            if extra:extra.release();extra=None
            command=('cd '+str(HERE)+' && bash launch.sh train --run-id '+a.source_run+
                     ' --level c4 --owner '+a.owner+' --gpus 0,1,2,3 --resume >> '+
                     str(root/'controller-logs'/f'{a.source_run}-fallback.log')+' 2>&1')
            subprocess.run(['tmux','new-session','-d','-s','br-codex-gzy-c4-fallback',command],check=True)
        raise
    finally:
        if extra:extra.release()


if __name__=='__main__':main()
