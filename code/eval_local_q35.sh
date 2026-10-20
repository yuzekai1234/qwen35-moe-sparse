#!/usr/bin/env bash
# 本地单卡版官方评测: bash code/eval_local_q35.sh <dataset.jsonl> <model_dir(runs/<run>/hf)> [seq_len] [batch] [gpu]
# 例: bash code/eval_local_q35.sh /home/shared/data/dolma3_mix_100M_a.jsonl runs/q35_c2_m/hf 8192 2 0
# 可改的变量: PY=Python 解释器（默认 <本目录>/../.venv_q35/bin/python），LIBCUDA=放 libcuda.so 软链的目录（默认 $HOME/libcuda）
set -euo pipefail
d=$(cd "$(dirname "$0")" && pwd); ds=$1; m=$2; seq=${3:-2048}; bs=${4:-8}; gpu=${5:-0}
PY=${PY:-$d/../.venv_q35/bin/python}; LIBCUDA=${LIBCUDA:-$d/../libcuda}
out=$m/../eval_$(basename "$ds" .jsonl)_seq${seq}.json; rm -f "$out"
TRITON_LIBCUDA_PATH=$LIBCUDA LIBRARY_PATH=$LIBCUDA CUDA_VISIBLE_DEVICES=$gpu $PY -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=1 \
  "$d/eval_q35.py" --dataset-path "$ds" --model-path "$m" --batch-size "$bs" --sequence-length "$seq" --output "$out"
echo "RESULT_FILE $out"; cat "$out"
