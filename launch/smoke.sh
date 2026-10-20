#!/usr/bin/env bash
# Includes all eight C2-m layers, 64 experts, actual EXAONE vocabulary and FLA kernels.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-$ROOT/.venv_q35/bin/python}
"$PY" "$ROOT/code/prepare_smoke.py"
DATA="$ROOT/data/smoke/train.bin" bash "$ROOT/launch/train.sh" q35_c2_m \
  --run_dir "$ROOT/runs/smoke" --seq 256 --tokens 16384 --batch_tokens 8192 \
  --micro_bs 1 --eval_tokens 2048 --warmup 1 --log_every 1 --ckpt_every 1 --resume 0 "$@"
