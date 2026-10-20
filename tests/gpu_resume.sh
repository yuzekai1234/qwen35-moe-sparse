#!/usr/bin/env bash
# End-to-end: two real GPUs, interruption/resume vs uninterrupted training.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
export DATA="$ROOT/data/smoke_gzy10k/train.bin"
export GPUS=${GPUS:-1,2}
PY=${PY:-$ROOT/.venv_q35/bin/python}
COMMON=(--seq 256 --tokens 1024 --batch_tokens 512 --micro_bs 1
  --eval_tokens 1024 --warmup 1 --log_every 1 --ckpt_every 1)
bash launch/train.sh q35_c2_m "${COMMON[@]}" --run_dir "$ROOT/runs/ddp_resume_gzy10k" --max_steps 1 --resume 0
Q35_VERIFY_RESUME=1 bash launch/train.sh q35_c2_m "${COMMON[@]}" --run_dir "$ROOT/runs/ddp_resume_gzy10k" --resume 1
bash launch/train.sh q35_c2_m "${COMMON[@]}" --run_dir "$ROOT/runs/ddp_reference_gzy10k" --resume 0
"$PY" tests/check_resume.py runs/ddp_resume_gzy10k runs/ddp_reference_gzy10k
