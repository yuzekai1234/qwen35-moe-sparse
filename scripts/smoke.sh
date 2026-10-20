#!/usr/bin/env bash
# GPUS=1 bash scripts/smoke.sh [q35.train args...]
# Full C2-m model, GZY 10K vocab, FLA kernels; seq 256, two steps, checkpoint and HF export.
set -euo pipefail
source "$(dirname "$0")/env.sh"
"$PY" -m q35.data sample --out data/smoke_gzy10k/train.bin
run_module q35.train --preset q35_c2_m --data data/smoke_gzy10k/train.bin --run_dir runs/smoke \
  --seq 256 --tokens 16384 --batch_tokens 8192 --micro_bs 1 --eval_tokens 2048 \
  --warmup 1 --log_every 1 --ckpt_every 1 --resume 0 "$@"
