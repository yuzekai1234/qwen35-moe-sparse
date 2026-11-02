#!/usr/bin/env bash
# A full C1 shape/seq/global-batch short run, then resume and portable-export check.
# PY=/path/to/python GPUS=0 timeout 900 bash scripts/smoke_dense.sh /new/output baseline
set -euo pipefail
source "$(dirname "$0")/env.sh"
OUT=${1:?usage: scripts/smoke_dense.sh <new-output-directory>}
[[ "$OUT" = /* ]] || OUT="$ROOT/$OUT"
[[ ! -e "$OUT" ]] || { echo "Refusing to reuse qualification output: $OUT" >&2; exit 1; }
mkdir -p "$OUT"
"$PY" -m unittest tests.test_qsa_gpu -v > "$OUT/qsa-regression.log" 2>&1
TARGET=${2:-c1}
COMMON=("$TARGET" --run_dir "$OUT/run" --warmup 1 --micro_bs 4 --auto_micro_bs 0
        --eval_tokens 8193 --final_eval_tokens 8193 --log_every 1 --tensorboard 0)
# The one-update warmup is deliberately only for qualification. Formal runs use
# the generated full-budget schedule and never resume this short-run checkpoint.
bash "$ROOT/scripts/train_dense.sh" "${COMMON[@]}" --max_steps 2 > "$OUT/train-first.log" 2>&1
bash "$ROOT/scripts/train_dense.sh" "${COMMON[@]}" --max_steps 3 > "$OUT/train-resume.log" 2>&1
"$PY" "$ROOT/scripts/check_dense_gpu.py" --run-dir "$OUT/run" --updates 3 > "$OUT/check.log" 2>&1
cat "$OUT/run/qualification.json"
