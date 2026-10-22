# Shared environment for scripts/*.sh (sourced). Overridable: PY, LIBCUDA, GPUS.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
PY=${PY:-$ROOT/.venv_q35/bin/python}
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TRITON_LIBCUDA_PATH=${LIBCUDA:-$ROOT/libcuda}
export LIBRARY_PATH="$TRITON_LIBCUDA_PATH${LIBRARY_PATH:+:$LIBRARY_PATH}"
export TOKENIZERS_PARALLELISM=false
# Reduces fragmentation so the micro_bs probe's peak holds over long runs.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export CUDA_VISIBLE_DEVICES=${GPUS:-${CUDA_VISIBLE_DEVICES:-0}}
IFS=, read -r -a DEVICES <<< "$CUDA_VISIBLE_DEVICES"
NPROC=${#DEVICES[@]}
# Run a module on all visible GPUs: plain python for one GPU, torchrun otherwise.
run_module() {
  if ((NPROC > 1)); then
    "$PY" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" -m "$@"
  else
    "$PY" -m "$@"
  fi
}
