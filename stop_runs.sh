#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PID_DIR="$ROOT/reports/pids"

stop_one() {
  local name="$1"
  local pidf="$PID_DIR/${name}.pid"
  if [[ ! -f "$pidf" ]]; then
    echo "[skip] $name no pid file"
    return
  fi
  local pid
  pid="$(cat "$pidf" || true)"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid" || true
    echo "[stop] $name pid=$pid"
  else
    echo "[skip] $name not running"
  fi
}

stop_one "standard_suite_full"
stop_one "two_stage_full"
stop_one "topk_cuda_1m_4k"
stop_one "public_dataset_full"
