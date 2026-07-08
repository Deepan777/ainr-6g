#!/usr/bin/env bash
# Sequential experiment suite (single 6 GB GPU — one job at a time).
set -u
cd "$(dirname "$0")"
PY=venv/Scripts/python.exe
LOG=results/exp_logs

run() {
  name=$1; shift
  echo "=== [$name] START $(date +%H:%M:%S) ==="
  "$PY" "$@" > "$LOG/$name.log" 2>&1
  rc=$?
  echo "=== [$name] EXIT $rc $(date +%H:%M:%S) ==="
  tail -3 "$LOG/$name.log"
}

run latent    exp_latent_target.py
run roc       exp_roc_stats.py
run latency   exp_latency.py
run s4ci      exp_s4_ci.py
run staged    exp_stage_d_stats.py
run ablation  exp_ablation.py
echo "ALL_EXPERIMENTS_DONE"
