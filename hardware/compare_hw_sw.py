#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def load_csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def mean(vals):
    return sum(vals) / max(1, len(vals))


def main():
    ap = argparse.ArgumentParser(description="Compare hardware and software runs on same protocol")
    ap.add_argument("--sw", type=str, default="reports/standard_suite_report.json")
    ap.add_argument("--hw", type=str, default="hardware/hw_runs.csv")
    ap.add_argument("--out", type=str, default="reports/hw_sw_comparison.json")
    args = ap.parse_args()

    sw = json.loads(Path(args.sw).read_text(encoding="utf-8"))
    hw_rows = load_csv(args.hw)

    # aggregate software by problem+method
    sw_agg = defaultdict(list)
    for r in sw["rows"]:
        sw_agg[(r["problem"], r["ablation"])].append(r)

    hw_agg = defaultdict(list)
    for r in hw_rows:
        try:
            q = float(r["quality"])
            t = float(r["runtime_s"])
            e = float(r["energy_j"])
        except Exception:
            continue
        hw_agg[(r["problem"], r["method"])].append((q, t, e))

    out = []
    for (problem, method), vals in hw_agg.items():
        sw_key = method.replace("hw_", "")
        sw_rows = sw_agg.get((problem, sw_key), [])
        if not sw_rows:
            continue
        sw_q = mean([x["quality_mean"] for x in sw_rows])
        sw_t = mean([x["time_mean_s"] for x in sw_rows])
        sw_e = mean([x["energy_mean_j"] for x in sw_rows])

        qh = mean([v[0] for v in vals])
        th = mean([v[1] for v in vals])
        eh = mean([v[2] for v in vals])

        out.append(
            {
                "problem": problem,
                "method": method,
                "hw_quality_mean": qh,
                "sw_quality_mean": sw_q,
                "quality_delta_hw_minus_sw": qh - sw_q,
                "hw_time_mean_s": th,
                "sw_time_mean_s": sw_t,
                "time_ratio_hw_over_sw": th / max(sw_t, 1e-9),
                "hw_energy_mean_j": eh,
                "sw_energy_mean_j": sw_e,
                "energy_ratio_hw_over_sw": eh / max(sw_e, 1e-9),
            }
        )

    payload = {"rows": out}
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
