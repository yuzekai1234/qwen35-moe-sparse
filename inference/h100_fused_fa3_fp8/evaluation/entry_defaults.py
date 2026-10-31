"""Qualified release defaults, without importing torch or creating a CUDA context."""
import json
import os
from pathlib import Path
import shutil
import sys


def apply_defaults():
    settings=json.loads(Path(__file__).with_name('release-defaults.json').read_text())
    for name,value in settings.items():
        os.environ.setdefault(name,str(value))
    for name in ['LOCAL_RANK','RANK']:
        os.environ.setdefault(name,'0')
    os.environ.setdefault('WORLD_SIZE','1')
    if os.environ.get('Q38_FLASHQLA_BACKEND')=='1':
        nvcc=shutil.which('nvcc')
        if not os.environ.get('CUDA_HOME') and nvcc:
            os.environ['CUDA_HOME']=str(Path(nvcc).resolve().parent.parent)
        os.environ.setdefault('MAX_JOBS','2')


def python_executable():
    requested=os.environ.get('Q38_PYTHON',sys.executable)
    found=shutil.which(requested)
    if found is None:raise ValueError('Q38_PYTHON does not name an executable Python')
    # Resolve neither venv executable nor its symlink: that would lose pyvenv.cfg.
    return os.path.abspath(found)


def ensure_python():
    selected=python_executable()
    if os.path.abspath(sys.executable)!=selected:
        os.execv(selected,[selected,*sys.argv])


def explicit_batch(cli_value=None):
    if cli_value is not None:
        value=int(cli_value)
    else:
        raw=os.environ.get('EVAL_BATCH_SIZE','').strip()
        value=None if raw.lower() in ('','auto') else int(raw)
    if value is not None and value<1:
        raise ValueError('Batch size must be positive')
    return value
