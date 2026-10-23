#!/usr/bin/env bash
# GPUS=1 bash scripts/smoke.sh [q38.train args...]
# C1 model, GZY 10K vocab, FLA kernels; seq 256, two steps, checkpoint and HF export.
set -euo pipefail
source "$(dirname "$0")/env.sh"
"$PY" -m q38.data sample --out data/smoke_gzy10k/train.bin
run_module q38.train --preset q38_c1 --data data/smoke_gzy10k/train.bin --val_data "" --run_dir runs/smoke \
  --seq 256 --tokens 16384 --batch_tokens 8192 --micro_bs 1 --eval_tokens 2048 \
  --warmup 1 --log_every 1 --ckpt_every 1 --resume 0 "$@"
