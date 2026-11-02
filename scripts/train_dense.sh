#!/usr/bin/env bash
# Defaults to baseline; explicit c1-c4 select the historical FFN-only configs.
set -euo pipefail
source "$(dirname "$0")/env.sh"
TARGET=baseline
if (($#)) && [[ "$1" != --* ]]; then TARGET=$1; shift; fi
case "$TARGET" in
  baseline) CONFIG="$ROOT/configs/baseline.json" ;;
  c1|c2|c3|c4) CONFIG="$ROOT/configs/dense/q38_$TARGET/config.json" ;;
  *) CONFIG="$TARGET" ;;
esac
test -f "$CONFIG" || { echo "Missing dense config: $CONFIG" >&2; exit 1; }
PLAN=$(CUDA_VISIBLE_DEVICES="" WORLD_SIZE="$NPROC" "$PY" -m q38.train --config "$CONFIG" "$@" --dry-run)
for arg in "$@"; do
  case "$arg" in --dry-run|--dry_run) printf '%s\n' "$PLAN"; exit 0 ;; esac
done
RUN_DIR=$(printf '%s\n' "$PLAN" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["run_dir"])')
mkdir -p "$RUN_DIR"
run_module q38.train --config "$CONFIG" "$@" 2>&1 | tee -a "$RUN_DIR/train.log"
