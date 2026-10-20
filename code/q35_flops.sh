#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-$ROOT/.venv_q35/bin/python}
RUNS=${RUNS:-$ROOT/runs}
"$PY" - "$RUNS/${1:?run name required}" "$ROOT/code" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[2])
from count_flops_q35 import count
run = Path(sys.argv[1])
config = json.loads((run / 'run_config.json').read_text())
summary = json.loads((run / 'summary.json').read_text())
result = count(config['config'], config['seq'])
result.update(train_tokens=summary['tokens_trained'], train_flops=summary['tokens_trained'] * result['train_flops_per_token'])
(run / 'flops.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
PY
