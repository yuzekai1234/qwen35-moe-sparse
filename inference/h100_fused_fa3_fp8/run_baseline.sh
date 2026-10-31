#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${Q38_PYTHON:-python3}" "$script_dir/run_baseline.py" "$@"
