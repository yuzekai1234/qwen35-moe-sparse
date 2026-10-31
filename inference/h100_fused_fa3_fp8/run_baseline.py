"""Fixed qualified C4 operating point; run only into a new directory."""
import hashlib
import json
import os
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
if len(sys.argv) != 4:
    raise SystemExit('Usage: run_baseline.sh /absolute/data.jsonl /absolute/model /new/output-directory')
data, model, out = map(Path, sys.argv[1:])
if not data.is_absolute() or not model.is_absolute():
    raise SystemExit('Dataset and model paths must be absolute')
for name, expected in json.loads((root / 'evaluation-source-sha256.json').read_text()).items():
    if hashlib.sha256((root / 'evaluation' / name).read_bytes()).hexdigest() != expected:
        raise SystemExit('Frozen evaluation source changed: ' + name)
env = os.environ.copy()
env.update(json.loads((root / 'evaluation/release-defaults.json').read_text()))
env.update(EVAL_BATCH_SIZE='938', EVAL_SEQ='4096', EVAL_GPUS='1',
           Q38_ATTENTION_BATCH_LIMIT='512', Q38_CUDA_GRAPH='0')
env.setdefault('Q38_PYTHON', sys.executable)
out = out.resolve()
out.mkdir()
os.chdir(out)
os.execve('/bin/bash', ['bash', str(root / 'evaluation/eval.sh'), str(data), str(model)], env)
