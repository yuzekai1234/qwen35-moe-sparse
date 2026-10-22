#!/usr/bin/env bash
# [GPUS=0,1,2,3,4,5,6,7] [EVAL=1] bash scripts/run_isoflops_all.sh [budget...]   e.g. 3e15 (default: all)
# One IsoFLOPs config per GPU, smallest budget first; each GPU takes the next config when it finishes.
# Every job trains (incl. final_eval.json), then runs scripts/eval.sh on its hf/ export (EVAL=0 skips).
# Re-running skips finished jobs and resumes interrupted ones from ckpt.pt. Status: runs/isoflops_queue.log
set -euo pipefail
export GPUS=${GPUS:-0,1,2,3,4,5,6,7}
source "$(dirname "$0")/env.sh"
EVAL=${EVAL:-1}
LOG=runs/isoflops_queue.log
QUEUE=$(mktemp)
trap 'rm -f "$QUEUE" "$QUEUE.lock"' EXIT
trap 'kill 0' INT TERM

# One line per job: <budget> <model> <variant> <run_dir>
"$PY" - "$@" > "$QUEUE" <<'EOF'
import json, sys
from pathlib import Path
jobs = []
for path in Path("configs/isoflops").glob("*/M*/*.json"):
    c = json.loads(path.read_text()); e = c["experiment"]
    if len(sys.argv) == 1 or e["budget"] in sys.argv[1:]:
        jobs.append((c["compute"], e["model"], path.stem, e["budget"], c["train"]["run_dir"]))
for _, model, variant, budget, run_dir in sorted(jobs):
    print(budget, model, variant, run_dir)
EOF

say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
next_job() { flock "$QUEUE.lock" sh -c 'head -n 1 "$1"; sed -i 1d "$1"' _ "$QUEUE"; }
trained() {
  test -f "$1/final_eval.json" &&
    "$PY" -c 'import json, sys; sys.exit(not json.load(open(sys.argv[1]))["completed_budget"])' "$1/summary.json"
}
evaluated() { compgen -G "$1/eval_*.json" > /dev/null; }

worker() {
  local gpu=$1 job budget model variant dir name
  while job=$(next_job) && [[ -n $job ]]; do
    read -r budget model variant dir <<< "$job"
    name="$budget/$model/$variant"; mkdir -p "$dir"
    if ! trained "$dir"; then
      say "GPU$gpu train $name"
      GPUS=$gpu VARIANT=$variant bash scripts/train_isoflops.sh "$budget" "$model" > /dev/null 2>> "$dir/train.log" ||
        { say "GPU$gpu FAIL train $name (see $dir/train.log)"; continue; }
    fi
    if ((EVAL)) && ! evaluated "$dir"; then
      say "GPU$gpu eval $name"
      GPUS=$gpu bash scripts/eval.sh "$dir/hf" > "$dir/eval.log" 2>&1 ||
        { say "GPU$gpu FAIL eval $name (see $dir/eval.log)"; continue; }
    fi
    say "GPU$gpu done $name"
  done
}

say "queue: $(wc -l < "$QUEUE") jobs on GPUs $GPUS"
for gpu in "${DEVICES[@]}"; do worker "$gpu" & done
wait
say "all workers finished"
