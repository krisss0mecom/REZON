#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import numpy as np

try:
    from scipy import stats as scipy_stats
except Exception:
    scipy_stats = None


METRICS = ["pairwise_correlation", "mutual_info", "bipartition_entropy", "chsh_proxy"]


def _rankdata(x):
    # simple average rank for ties
    arr = np.asarray(x, dtype=np.float64)
    order = np.argsort(arr)
    ranks = np.empty_like(order, dtype=np.float64)
    i = 0
    while i < len(arr):
        j = i
        while j + 1 < len(arr) and arr[order[j + 1]] == arr[order[i]]:
            j += 1
        r = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = r
        i = j + 1
    return ranks


def corr_pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or y.size < 2:
        return math.nan, math.nan
    if scipy_stats is not None:
        try:
            r, p = scipy_stats.pearsonr(x, y)
            return float(r), float(p)
        except Exception:
            pass
    r = float(np.corrcoef(x, y)[0, 1])
    return r, math.nan


def corr_spearman(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or y.size < 2:
        return math.nan, math.nan
    if scipy_stats is not None:
        try:
            r, p = scipy_stats.spearmanr(x, y)
            return float(r), float(p)
        except Exception:
            pass
    rx = _rankdata(x)
    ry = _rankdata(y)
    r = float(np.corrcoef(rx, ry)[0, 1])
    return r, math.nan


def read_reports(report_dir: Path, pattern: str):
    files = sorted(report_dir.glob(pattern))
    payloads = []
    for p in files:
        try:
            payloads.append((p.name, json.loads(p.read_text(encoding="utf-8"))))
        except Exception:
            continue
    return payloads


def analyze_one(payload):
    rows = payload["ablation_results"]
    quality = np.array([r["quality_mean"] for r in rows], dtype=np.float64)
    out = {}
    for m in METRICS:
        vals = np.array([r["metrics_mean"][m] for r in rows], dtype=np.float64)
        pr, pp = corr_pearson(vals, quality)
        sr, sp = corr_spearman(vals, quality)
        out[m] = {
            "pearson_r": pr,
            "pearson_p": pp,
            "spearman_r": sr,
            "spearman_p": sp,
        }
    return out


def to_markdown(summary):
    lines = [
        "# Phase Metrics vs Quality Correlation Pack",
        "",
        "Scope: classical proxy metrics vs Max-Cut quality proxy.",
        "",
    ]
    for block in summary["per_run"]:
        lines.append(f"## {block['run_name']}")
        lines.append("")
        lines.append("| metric | pearson r | pearson p | spearman r | spearman p |")
        lines.append("|---|---:|---:|---:|---:|")
        for m in METRICS:
            v = block["correlations"][m]
            pp = v["pearson_p"]
            sp = v["spearman_p"]
            pp_s = "nan" if pp != pp else f"{pp:.4g}"
            sp_s = "nan" if sp != sp else f"{sp:.4g}"
            lines.append(
                f"| {m} | {v['pearson_r']:.4f} | {pp_s} | {v['spearman_r']:.4f} | {sp_s} |"
            )
        lines.append("")

    lines.append("## Aggregate")
    lines.append("")
    lines.append("| metric | pearson r | spearman r |")
    lines.append("|---|---:|---:|")
    for m in METRICS:
        a = summary["aggregate"][m]
        lines.append(f"| {m} | {a['pearson_r']:.4f} | {a['spearman_r']:.4f} |")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Analyze correlation between phase metrics and quality across ablation runs")
    ap.add_argument("--report-dir", type=str, default="reports/correlation_pack")
    ap.add_argument("--pattern", type=str, default="phase_entanglement_nodes*_s*.json")
    ap.add_argument("--out-json", type=str, default="reports/correlation_pack_summary.json")
    ap.add_argument("--out-md", type=str, default="reports/correlation_pack_summary.md")
    args = ap.parse_args()

    report_dir = Path(args.report_dir)
    payloads = read_reports(report_dir, args.pattern)
    if not payloads:
        raise SystemExit("No input reports found")

    per_run = []
    all_quality = []
    all_metrics = {m: [] for m in METRICS}

    for name, payload in payloads:
        corr = analyze_one(payload)
        per_run.append({"run_name": name, "correlations": corr})
        for r in payload["ablation_results"]:
            all_quality.append(float(r["quality_mean"]))
            for m in METRICS:
                all_metrics[m].append(float(r["metrics_mean"][m]))

    all_quality = np.asarray(all_quality, dtype=np.float64)
    aggregate = {}
    for m in METRICS:
        vals = np.asarray(all_metrics[m], dtype=np.float64)
        pr, _ = corr_pearson(vals, all_quality)
        sr, _ = corr_spearman(vals, all_quality)
        aggregate[m] = {"pearson_r": pr, "spearman_r": sr}

    summary = {"per_run": per_run, "aggregate": aggregate}

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    Path(args.out_md).write_text(to_markdown(summary), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"saved: {args.out_json}")
    print(f"saved: {args.out_md}")


if __name__ == "__main__":
    main()
