"""Build pinned upstream FA3 privately; a new binary needs independent qualification."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import tarfile
import urllib.request

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--destination', type=Path, required=True, help='New private build directory')
p.add_argument('--cuda-home', type=Path, required=True, help='Real CUDA12.8 toolkit, not system CUDA13')
p.add_argument('--python-include', type=Path, default=Path(sysconfig.get_path('include')))
p.add_argument('--extra-include', type=Path, action='append', default=[])
p.add_argument('--jobs', type=int, default=4)
a = p.parse_args()
if not (a.python_include / 'Python.h').is_file():
    p.error('Python development headers are required; pass --python-include')
if not (a.cuda_home / 'bin/nvcc').is_file():
    p.error('--cuda-home must contain a real nvcc')
root = a.destination.resolve()
root.mkdir(parents=True, exist_ok=False)
manifest = json.loads(Path(__file__).with_name('evaluation').joinpath('fa3-dependency.json').read_text())

def fetch(repo, commit, name):
    archive = root / name
    urllib.request.urlretrieve('https://codeload.github.com/' + repo + '/tar.gz/' + commit, archive)
    with archive.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != manifest['source_archives'][name]:
        raise RuntimeError('Upstream source archive hash differs: ' + name)
    target = root / name.removesuffix('.tar.gz')
    target.mkdir()
    with tarfile.open(archive) as tar:
        tar.extractall(target, filter='data')
    return next(target.iterdir())

source = fetch('Dao-AILab/flash-attention', manifest['FA3_commit'], 'fa3-source.tar.gz')
cutlass = fetch('NVIDIA/cutlass', manifest['CUTLASS_commit'], 'cutlass-source.tar.gz')
target = source / 'csrc/cutlass'
target.parent.mkdir(parents=True, exist_ok=True)
if target.exists():
    target.rmdir()  # Upstream submodule placeholder must be empty.
target.symlink_to(cutlass, target_is_directory=True)
env = os.environ.copy()
env.update(CUDA_HOME=str(a.cuda_home.resolve()), MAX_JOBS=str(a.jobs), NVCC_THREADS='2',
           FLASH_ATTENTION_FORCE_BUILD='TRUE', FLASH_ATTENTION_OFFLINE_BUILD='TRUE', BUILD_TARGET='cuda')
env['PATH'] = str(a.cuda_home.resolve() / 'bin') + ':' + env.get('PATH', '')
includes = [a.python_include, *a.extra_include]
env['CPATH'] = ':'.join(map(str, includes)) + (':' + env['CPATH'] if env.get('CPATH') else '')
env['CPLUS_INCLUDE_PATH'] = env['CPATH']
for feature in ['BACKWARD', 'SPLIT', 'PAGEDKV', 'APPENDKV', 'LOCAL', 'SOFTCAP', 'FP16',
                'VARLEN', 'CLUSTER', 'HDIM96', 'HDIM128', 'HDIM192', 'HDIM256',
                'SM80', 'HDIMDIFF64', 'HDIMDIFF192']:
    env['FLASH_ATTENTION_DISABLE_' + feature] = 'TRUE'
command = [sys.executable, 'setup.py', 'build_ext', '--inplace']
with (root / 'build.log').open('w') as log:
    completed = subprocess.run(command, cwd=source / 'hopper', env=env,
                               stdout=log, stderr=subprocess.STDOUT)
(root / 'build-receipt.json').write_text(json.dumps(dict(command=command,
    returncode=completed.returncode, hopper=str(source / 'hopper'),
    source_commit=manifest['FA3_commit'], cutlass_commit=manifest['CUTLASS_commit'],
    environment={k: v for k, v in env.items() if k.startswith('FLASH_') or k in
                 ['CUDA_HOME', 'CPATH', 'CPLUS_INCLUDE_PATH', 'MAX_JOBS', 'NVCC_THREADS']}), indent=2) + '\n')
if completed.returncode:
    raise SystemExit(completed.returncode)
print('Built:', source / 'hopper')
print('Builds on another host may have different binary hashes. Qualify before updating the frozen dependency manifest.')
