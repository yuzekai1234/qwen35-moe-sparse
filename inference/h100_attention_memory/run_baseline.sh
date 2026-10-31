#!/usr/bin/env bash
# Fixed, qualified C4 operating point; writes only into a new result directory.
set -euo pipefail
if [[ $# -ne 3 ]]; then
  echo 'Usage: run_baseline.sh /absolute/data.jsonl /absolute/model /new/output-directory' >&2
  exit 2
fi
if [[ "$1" != /* || "$2" != /* ]]; then
  echo 'Dataset and model paths must be absolute.' >&2
  exit 2
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
dataset="$1"
model="$2"
mkdir -- "$3"
cd -- "$3"
export EVAL_BATCH_SIZE=940 Q38_ATTENTION_BATCH_LIMIT=512 EVAL_SEQ=4096 EVAL_GPUS=1
export Q38_CUDA_GRAPH=0
exec bash "$script_dir/evaluation/eval.sh" "$dataset" "$model"
