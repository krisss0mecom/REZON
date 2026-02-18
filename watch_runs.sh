#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$ROOT/reports"
PID_DIR="$LOG_DIR/pids"

show_one() {
  local name="$1"
  local pidf="$PID_DIR/${name}.pid"
  local log="$LOG_DIR/${name}.log"

  if [[ ! -f "$pidf" ]]; then
    echo "[$name] no pid file"
    return
  fi

  local pid
  pid="$(cat "$pidf" || true)"
  if [[ -z "$pid" ]]; then
    echo "[$name] empty pid file"
    return
  fi

  if kill -0 "$pid" 2>/dev/null; then
    echo "[$name] RUNNING pid=$pid"
    ps -p "$pid" -o etime=,pcpu=,pmem=,cmd= | sed 's/^/  /'
  else
    echo "[$name] STOPPED pid=$pid"
  fi

  if [[ -f "$log" ]]; then
    echo "  log: $log"
    tail -n 3 "$log" | sed 's/^/    /'
  else
    echo "  log: (missing)"
  fi
}

show_one "standard_suite_full"
show_one "two_stage_full"
show_one "topk_cuda_1m_4k"
show_one "public_dataset_full"
