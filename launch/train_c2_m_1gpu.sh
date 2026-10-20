#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
GPUS=${GPUS:-0} bash "$ROOT/launch/train.sh" q35_c2_m "$@"
