#!/usr/bin/env bash
# Example: bash launch.sh train --level c4 --owner yzk --gpus 0,1,2,3
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
workspace="${Q38_WORKSPACE:-/data/quark-7db0d9a22ffb}"
source "$workspace/activate-c1.sh"
run_id=combined-v5
args=("$@")
for ((i=0; i<${#args[@]}; i++)); do
  if [[ "${args[i]}" == --run-id ]]; then run_id="${args[i+1]}"; fi
done
if [[ "${1:-}" != status ]]; then
  [[ "$run_id" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo 'Invalid run id' >&2; exit 2; }
  here="$workspace/b-round-20261109/runs/$run_id/source"
  [[ -f "$here/run.py" ]] || { echo 'Prepare this run id before launching' >&2; exit 2; }
fi
export PYTHONPATH="$here/runtime:$C1_MODEL_ROOT/compat-source"
exec "$Q38_PYTHON" "$here/run.py" "$@" --workspace "$workspace"
