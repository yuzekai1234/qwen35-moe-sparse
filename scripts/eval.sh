#!/usr/bin/env bash
# GPUS=1 bash scripts/eval.sh <model_dir (runs/<run>/hf)> [dataset.jsonl] [seq_len] [batch_per_gpu] [evaluate options...]
# Writes runs/<run>/eval_<dataset>_seq<seq>.json.
set -euo pipefail
MODEL=$(realpath "${1:?usage: scripts/eval.sh <model_dir> [dataset] [seq] [batch]}")
DATASET=$(realpath "${2:-/home/shared/data/dolma3_mix_100M_a.jsonl}"); SEQ=${3:-4096}; BS=${4:-2}
source "$(dirname "$0")/env.sh"
OUT=$(dirname "$MODEL")/eval_$(basename "$DATASET" .jsonl)_seq$SEQ.json
rm -f "$OUT"
# Evaluation always initializes a process group, so use torchrun even on one GPU.
"$PY" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" -m q38.evaluate \
  --dataset-path "$DATASET" --model-path "$MODEL" --batch-size "$BS" --sequence-length "$SEQ" --output "$OUT" "${@:5}"
echo "RESULT_FILE $OUT"; cat "$OUT"
