"""Complete this task's verified transfer, prepare portable environment, then run C3/C2.

Run in tmux on the allocated new node. Never starts a task before checksums and
CPU preflight pass; native launch.sh owns GPU occupancy checks and task locks.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--transfer-pid', type=int, required=True)
    p.add_argument('--run-id', default='matched-lc3-e512-b15-c23-20261109')
    a = p.parse_args()
    root=a.workspace.resolve(); transfer=root/'transfer'
    campaign=root/'team-repo/campaigns/b_round_20261109'
    state=root/'SETUP_C2C3_STATUS.json'
    child=None
    def update(stage, **extra):
        record=dict(stage=stage,updated_unix=time.time(),run_id=a.run_id,**extra)
        tmp=state.with_suffix('.tmp');tmp.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');tmp.replace(state)
        print(json.dumps(record,ensure_ascii=False),flush=True)
    def stop(signum,frame):
        if child is not None and child.poll() is None:
            child.terminate()
        raise KeyboardInterrupt('User stopped deployment/queue')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def run(cmd, **kw):
        nonlocal child
        child=subprocess.Popen(cmd,**kw)
        code=child.wait();child=None
        if code:raise RuntimeError(f'Command exited {code}: {cmd[:3]}')
    try:
        update('waiting_for_data_and_environment_transfer',transfer_pid=a.transfer_pid)
        deadline=time.monotonic()+3600
        while Path(f'/proc/{a.transfer_pid}').exists():
            stat=Path(f'/proc/{a.transfer_pid}/stat').read_text().split(') ',1)[1]
            if stat.startswith('Z'):break
            if time.monotonic()>deadline:raise TimeoutError('Transfer exceeded one hour')
            time.sleep(10)
        ssh=('ssh -i '+str(Path.home()/'.ssh/q38_transfer_ed25519')+
             ' -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=yes'+
             ' -o UserKnownHostsFile='+str(Path.home()/'.ssh/q38_source_known_hosts')+
             ' -o ConnectTimeout=15 -o ServerAliveInterval=30')
        update('transferring_cuda_toolkit')
        run(['rsync','-a','--partial','--info=progress2','-e',ssh,
             'gool@117.131.86.30:cuda-12.8.tar.zst',
             'gool@117.131.86.30:cuda.sha256',str(transfer)+'/'])
        update('verifying_transfers')
        verified={}
        for name,manifest in [('environment.tar.gz','environment.sha256'),('cuda-12.8.tar.zst','cuda.sha256')]:
            expected=(transfer/manifest).read_text().split()[0]
            actual=sha(transfer/name)
            if actual!=expected:raise RuntimeError('Checksum mismatch: '+name)
            verified[name]=actual
        run(['sha256sum','-c','SHA256SUMS'],cwd=transfer/'data')
        (root/'TRANSFER_VERIFIED.json').write_text(json.dumps(verified,indent=2)+'\n')
        update('extracting_environment')
        run(['tar','-xzf',str(transfer/'environment.tar.gz'),'-C',str(root)])
        runtime=root/'.runtime';runtime.mkdir(exist_ok=True)
        run(['tar','--zstd','-xf',str(transfer/'cuda-12.8.tar.zst'),'-C',str(runtime)])
        old='/data/quark-7db0d9a22ffb'
        python=runtime/'python/cpython-3.12.16-linux-x86_64-gnu/bin/python3.12'
        for f in (root/'.venv/bin').iterdir():
            if f.is_symlink():
                link=os.readlink(f)
                if link.startswith(old):f.unlink();f.symlink_to(link.replace(old,str(root),1))
            elif f.is_file() and f.stat().st_size<1024*1024:
                try:body=f.read_text()
                except UnicodeError:continue
                if old in body:f.write_text(body.replace(old,str(root)))
        cfg=root/'.venv/pyvenv.cfg';cfg.write_text(cfg.read_text().replace(old,str(root)))
        for f in (root/'.venv/lib/python3.12/site-packages').glob('*.pth'):
            f.write_text(f.read_text().replace(old,str(root)))
        data=root/'c1-combined-papers-data-20261108'
        if not data.exists():(transfer/'data').rename(data)
        cuda=Path('/usr/local/cuda-12.8')
        if not cuda.exists():run(['sudo','-n','ln','-s',str(runtime/'cuda-12.8'),str(cuda)])
        activate='''#!/bin/bash
export C1_WORKSPACE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$C1_WORKSPACE/.venv/bin/activate"
export Q38_PYTHON="$VIRTUAL_ENV/bin/python"
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONNOUSERSITE=1
export C1_MODEL_ROOT="$C1_WORKSPACE/c1-combined-papers-bpb0.99090686-20261108"
export C1_DATA_ROOT="$C1_WORKSPACE/c1-combined-papers-data-20261108"
export C1_HF_MODEL="$C1_MODEL_ROOT/runs/combined-papers/train/hf"
'''
        (root/'activate-c1.sh').write_text(activate)
        env=dict(os.environ,Q38_WORKSPACE=str(root),CUDA_HOME=str(cuda),
                 PYTHONPATH=str(campaign/'runtime')+':'+str(root/'c1-combined-papers-bpb0.99090686-20261108/compat-source'),
                 CUDA_VISIBLE_DEVICES='',PYTHONNOUSERSITE='1')
        py=str(root/'.venv/bin/python')
        update('preparing_matched_configs')
        run([py,str(campaign/'prepare_c2c3_matched.py'),'--workspace',str(root),'--run-id',a.run_id,'--world','8'],env=env)
        for level in ('c3','c2'):
            update('cpu_preflight',level=level)
            run([py,str(campaign/'preflight.py'),'--workspace',str(root),'--run-id',a.run_id,'--level',level],env=env)
        env.pop('CUDA_VISIBLE_DEVICES')
        for level in ('c3','c2'):
            update('training',level=level)
            run(['bash',str(campaign/'launch.sh'),'train','--run-id',a.run_id,'--level',level,
                 '--owner','codex-gzy','--gpus','0,1,2,3,4,5,6,7'],env=env,cwd=campaign)
        for level in ('c3','c2'):
            update('evaluating',level=level)
            run(['bash',str(campaign/'launch.sh'),'eval','--run-id',a.run_id,'--level',level,
                 '--owner','codex-gzy','--gpus','0','--eval-tag','raw-s10240-auto'],env=env,cwd=campaign)
        update('complete')
    except BaseException as e:
        update('interrupted' if isinstance(e,KeyboardInterrupt) else 'failed',error=repr(e))
        raise


if __name__=='__main__':main()
