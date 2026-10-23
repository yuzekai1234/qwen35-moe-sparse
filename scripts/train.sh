#!/usr/bin/env bash
# GPUS=1,2 bash scripts/train.sh <preset> [q38.train args...]   e.g. --max_steps 100 --run_dir runs/x
set -euo pipefail
source "$(dirname "$0")/env.sh"
PRESET=${1:?usage: scripts/train.sh <preset> [args...]}; shift
mkdir -p logs
run_module q38.train --preset "$PRESET" "$@" 2>&1 | tee -a "logs/$PRESET.log"
