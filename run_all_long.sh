#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$ROOT/reports"
PID_DIR="$LOG_DIR/pids"
mkdir -p "$LOG_DIR" "$PID_DIR"

cd "$ROOT"

start_job() {
  local name="$1"
  local cmd="$2"
  local log="$LOG_DIR/${name}.log"
  local pidf="$PID_DIR/${name}.pid"

  if [[ -f "$pidf" ]]; then
    local old_pid
    old_pid="$(cat "$pidf" || true)"
    if [[ -n "${old_pid}" ]] && kill -0 "$old_pid" 2>/dev/null; then
      echo "[skip] $name already running (pid=$old_pid)"
      return
    fi
  fi

  echo "[start] $name"
  nohup bash -lc "$cmd" >"$log" 2>&1 &
  local pid=$!
  echo "$pid" > "$pidf"
  echo "[pid] $name=$pid"
}

start_job "standard_suite_full" "python3 bench/standard_suite.py --instances 8 --seeds 8 --fast-mode 0 --rc-nodes 1000 --rc-warmup-steps 40 --out-json reports/standard_suite_report_full.json --out-md reports/standard_suite_report_full.md"

start_job "two_stage_full" "python3 bench/phase_guided_two_stage.py --bits 8,10,12,14,16,18,20 --reps 8 --shortlist-ratio 0.02 --out-json reports/two_stage_phase_guided_full.json --out-md reports/two_stage_phase_guided_full.md"

start_job "topk_cuda_1m_4k" "python3 bench/topk_cuda_wrapper.py --keys 1048576 --k 4096 --out reports/topk_cuda_1m_4k.txt"

start_job "public_dataset_full" "python3 bench/public_dataset_runner.py --n-cap 120 --out-json reports/public_dataset_benchmark_full.json"

echo
echo "All jobs submitted."
echo "Use: ./watch_runs.sh"
