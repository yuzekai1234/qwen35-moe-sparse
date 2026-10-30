#!/usr/bin/env bash
# Defaults to the current dense, no-mHC, full-attention C1 baseline.
set -euo pipefail
source "$(dirname "$0")/env.sh"
PRESET=baseline
if (($#)) && [[ "$1" != --* ]]; then PRESET=$1; shift; fi
mkdir -p logs
run_module q38.train --preset "$PRESET" "$@" 2>&1 | tee -a "logs/$PRESET.log"
