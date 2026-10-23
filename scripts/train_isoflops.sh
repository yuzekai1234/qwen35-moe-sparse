#!/usr/bin/env bash
# GPUS=0 [VARIANT=config] bash scripts/train_isoflops.sh 1e16 h256_L5 [args...]
set -euo pipefail
source "$(dirname "$0")/env.sh"
echo "Q35 IsoFLOPs configs are archived at configs/q35_isoflops; run them from the Q35 baseline commit." >&2
exit 1
BUDGET=${1:?usage: scripts/train_isoflops.sh <budget> <h...> [args...]}
MODEL=${2:?usage: scripts/train_isoflops.sh <budget> <h...> [args...]}
shift 2
CONFIG="$ROOT/configs/q35_isoflops/$BUDGET/$MODEL/${VARIANT:-config}.json"
test -f "$CONFIG" || { echo "Missing config: $CONFIG" >&2; exit 1; }
DRY=0
for arg in "$@"; do
  case "$arg" in
    --dry-run|--dry_run) DRY=1 ;;
    -h|--help) "$PY" -m q35.train --config "$CONFIG" --help; exit 0 ;;
  esac
done
# This preflight does not allocate a GPU model or create output files.
PLAN=$(CUDA_VISIBLE_DEVICES="" WORLD_SIZE="$NPROC" "$PY" -m q35.train --config "$CONFIG" "$@" --dry-run)
if ((DRY)); then printf '%s\n' "$PLAN"; exit 0; fi
RUN_DIR=$(printf '%s\n' "$PLAN" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["run_dir"])')
mkdir -p "$RUN_DIR"
run_module q35.train --config "$CONFIG" "$@" 2>&1 | tee -a "$RUN_DIR/train.log"
