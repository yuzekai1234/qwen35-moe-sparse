#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PRESET=${1:-q35_c2_m}
if (($#)); then shift; fi
PY=${PY:-$ROOT/.venv_q35/bin/python}
export TRITON_LIBCUDA_PATH=${LIBCUDA:-$ROOT/libcuda}
export LIBRARY_PATH="$TRITON_LIBCUDA_PATH${LIBRARY_PATH:+:$LIBRARY_PATH}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export CUDA_VISIBLE_DEVICES=${GPUS:-${CUDA_VISIBLE_DEVICES:-0}}
IFS=, read -r -a DEVICES <<< "$CUDA_VISIBLE_DEVICES"
mkdir -p "$ROOT/logs"
"$PY" "$ROOT/code/launch.py" "$PRESET" --nproc "${#DEVICES[@]}" "$@" 2>&1 |
  tee -a "$ROOT/logs/${PRESET}.log"
