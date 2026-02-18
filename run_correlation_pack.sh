#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

SEEDS="${1:-16}"
OUT_DIR="reports/correlation_pack"
mkdir -p "$OUT_DIR"

for N in 80 160 320 640 1000; do
  echo "[run] nodes=$N seeds=$SEEDS"
  python3 bench/phase_entanglement_ablation.py \
    --seeds "$SEEDS" \
    --nodes "$N" \
    --warmup 120 \
    --steps 240 \
    --out-json "$OUT_DIR/phase_entanglement_nodes${N}_s${SEEDS}.json" \
    --out-md "$OUT_DIR/phase_entanglement_nodes${N}_s${SEEDS}.md"
done

echo "[analyze] correlation summary"
python3 bench/analyze_phase_quality_correlation.py \
  --report-dir "$OUT_DIR" \
  --pattern "phase_entanglement_nodes*_s${SEEDS}.json" \
  --out-json "reports/correlation_pack_summary.json" \
  --out-md "reports/correlation_pack_summary.md"

echo "done"
