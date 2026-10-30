#!/usr/bin/env bash
# Contest entry point: bash evaluation/eval.sh <dataset.jsonl> <model_dir>
# Writes result.json (model_path, bpb, evaltime) into the current working directory.
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo 'Usage: bash evaluation/eval.sh /path/data.jsonl /path/model' >&2
  exit 2
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export FLASHINFER_DISABLE_JIT="${FLASHINFER_DISABLE_JIT:-1}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export Q38_VARIANT="${Q38_VARIANT:-general_dense_fused_norm}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MASTER_ADDR=127.0.0.1



args=(--dataset-path "$1" --model-path "$2"
      --sequence-length "${EVAL_SEQ:-4096}"
      --output "$PWD/result.json")
if [[ -n "${EVAL_BATCH_SIZE:-}" && "${EVAL_BATCH_SIZE}" != "auto" ]]; then
  args+=(--batch-size "$EVAL_BATCH_SIZE")
fi

if [[ "${EVAL_GPUS:-1}" == "1" ]]; then
  # Single GPU: skip the distributed launcher entirely, so the run does not
  # depend on hostname/DNS resolution (unavailable under --network none).
  export LOCAL_RANK=0 RANK=0 WORLD_SIZE=1
  exec "${Q38_PYTHON:-python}" "$script_dir/eval.py" "${args[@]}"
fi

# Multi-GPU: torch.distributed needs its own hostname to resolve. When the
# container runs with --network none, pass `--hostname eval --add-host eval:127.0.0.1`
# to docker run (documented in REPRODUCTION.md).
exec "${Q38_PYTHON:-python}" -m torch.distributed.run --standalone \
  --master-addr=127.0.0.1 --master-port="${EVAL_PORT:-29531}" --nnodes=1 \
  --nproc-per-node="${EVAL_GPUS:-1}" \
  "$script_dir/eval.py" "${args[@]}"
