#!/usr/bin/env python3
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from bench.phase_entanglement_metrics import compute_all_metrics

try:
    from scipy import stats as scipy_stats
except Exception:
    scipy_stats = None

ANCHOR_HZ = 200.0


def ensure_anchor_immutable(freq_hz: float) -> None:
    if abs(float(freq_hz) - ANCHOR_HZ) > 1e-6:
        raise ValueError(f"anchor must be exactly {ANCHOR_HZ} Hz")


def bootstrap_ci(x, alpha=0.05, n_boot=500):
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0:
        return [math.nan, math.nan]
    rng = np.random.default_rng(123)
    vals = []
    for _ in range(n_boot):
        s = rng.choice(arr, size=arr.size, replace=True)
        vals.append(float(np.mean(s)))
    return [float(np.quantile(vals, alpha / 2.0)), float(np.quantile(vals, 1 - alpha / 2.0))]


def pvalue_vs_baseline(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or y.size < 2:
        return math.nan
    if scipy_stats is not None:
        try:
            return float(scipy_stats.ttest_ind(x, y, equal_var=False).pvalue)
        except Exception:
            pass
    return math.nan


def gen_maxcut(n, edge_p, seed):
    rng = np.random.default_rng(seed)
    W = rng.uniform(-1.0, 1.0, size=(n, n))
    M = rng.random((n, n)) < edge_p
    W = np.where(M, W, 0.0)
    W = np.triu(W, 1)
    W = W + W.T
    np.fill_diagonal(W, 0.0)
    return W


def eval_maxcut(W, x_pm1):
    n = W.shape[0]
    v = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if x_pm1[i] != x_pm1[j]:
                v += W[i, j]
    return float(v)


def simulate_phase(n, coupling, noise_amp, leak, anchor_amp, seed, warmup, steps):
    rng = np.random.default_rng(seed)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    omega = rng.uniform(-0.25, 0.25, size=n)
    t = 0.0
    ensure_anchor_immutable(ANCHOR_HZ)

    for _ in range(warmup + steps):
        z = np.mean(np.exp(1j * phi))
        R = np.abs(z)
        theta = np.angle(z)
        dphi = omega + coupling * R * np.sin(theta - phi)
        if anchor_amp > 0.0:
            dphi += anchor_amp * np.sin(2.0 * np.pi * ANCHOR_HZ * t - phi)
        dphi -= leak * phi
        if noise_amp > 0:
            dphi += rng.normal(0.0, noise_amp, size=n)
        phi = (phi + 0.01 * dphi) % (2.0 * np.pi)
        t += 0.01

    return phi


def run_ablation(seeds, n_nodes, warmup, steps):
    grid = []
    for coupling in (1.0, 1.8, 2.4):
        for noise in (0.0, 0.02, 0.05):
            for leak in (0.005, 0.02):
                for anchor in (0.0, 0.4):
                    grid.append((coupling, noise, leak, anchor))

    baseline = (1.0, 0.05, 0.02, 0.0)
    results = []

    for cfg in grid:
        c, noise, leak, anch = cfg
        metric_rows = []
        quality_rows = []
        for s in range(seeds):
            phi = simulate_phase(
                n=n_nodes,
                coupling=c,
                noise_amp=noise,
                leak=leak,
                anchor_amp=anch,
                seed=1000 + 17 * s,
                warmup=warmup,
                steps=steps,
            )
            m = compute_all_metrics(phi)
            metric_rows.append(m)

            W = gen_maxcut(n_nodes, edge_p=0.2, seed=500 + s)
            x = np.where(np.cos(phi) >= 0.0, 1.0, -1.0)
            q = eval_maxcut(W, x)
            quality_rows.append(q)

        row = {
            "params": {
                "coupling": c,
                "noise": noise,
                "leak": leak,
                "anchor": anch,
                "nodes": n_nodes,
                "warmup": warmup,
                "steps": steps,
            },
            "metrics_mean": {
                "pairwise_correlation": float(np.mean([r["pairwise_correlation"] for r in metric_rows])),
                "mutual_info": float(np.mean([r["mutual_info"] for r in metric_rows])),
                "bipartition_entropy": float(np.mean([r["bipartition_entropy"] for r in metric_rows])),
                "chsh_proxy": float(np.mean([r["chsh_proxy"] for r in metric_rows])),
            },
            "metrics_raw": metric_rows,
            "quality_raw": [float(x) for x in quality_rows],
            "quality_mean": float(np.mean(quality_rows)),
            "quality_std": float(np.std(quality_rows)),
        }
        results.append(row)

    base_rows = [r for r in results if (
        r["params"]["coupling"],
        r["params"]["noise"],
        r["params"]["leak"],
        r["params"]["anchor"],
    ) == baseline]
    base = base_rows[0] if base_rows else None

    stats_out = {}
    metric_keys = ["pairwise_correlation", "mutual_info", "bipartition_entropy", "chsh_proxy"]
    for mk in metric_keys:
        vals = [r["metrics_mean"][mk] for r in results]
        if base is not None:
            base_vals = [x[mk] for x in base["metrics_raw"]]
            pool_vals = []
            for r in results:
                for rr in r["metrics_raw"]:
                    pool_vals.append(rr[mk])
            p = pvalue_vs_baseline(pool_vals, base_vals)
        else:
            p = math.nan
        stats_out[mk] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "ci95": bootstrap_ci(vals),
            "p_value_vs_baseline": p,
        }

    q_vals = [r["quality_mean"] for r in results]
    if base is not None:
        base_q = np.asarray(base["quality_raw"], dtype=np.float64)
        pool_q = np.asarray([q for r in results for q in r["quality_raw"]], dtype=np.float64)
        p_q = pvalue_vs_baseline(pool_q, base_q)
    else:
        p_q = math.nan

    stats_out["quality"] = {
        "mean": float(np.mean(q_vals)),
        "std": float(np.std(q_vals)),
        "ci95": bootstrap_ci(q_vals),
        "p_value_vs_baseline": p_q,
    }

    return {"ablation_results": results, "stats": stats_out}


def to_markdown(payload):
    lines = [
        "# Phase Entanglement-Like Ablation (Classical Proxy)",
        "",
        "Claim boundary: proxy metrics on classical phase dynamics (not quantum entanglement).",
        "",
        "## Aggregate Stats",
        "",
        "| metric | mean | std | ci95 | p-value vs baseline |",
        "|---|---:|---:|---:|---:|",
    ]
    for k, v in payload["stats"].items():
        ci = v["ci95"]
        p = v["p_value_vs_baseline"]
        p_s = "nan" if p != p else f"{p:.4g}"
        lines.append(f"| {k} | {v['mean']:.6f} | {v['std']:.6f} | [{ci[0]:.6f}, {ci[1]:.6f}] | {p_s} |")
    lines.append("")
    lines.append("## Notes")
    lines.append("- `chsh_proxy` is CHSH-like classical proxy, not QM Bell test.")
    lines.append("- `quality` is Max-Cut objective from phase-derived partition.")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description="Ablation for phase-entanglement-like proxy metrics")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--nodes", type=int, default=80)
    ap.add_argument("--warmup", type=int, default=120)
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--out-json", type=str, default="reports/phase_entanglement_report.json")
    ap.add_argument("--out-md", type=str, default="reports/phase_entanglement_report.md")
    args = ap.parse_args()

    payload = {
        "config": {
            "seeds": int(args.seeds),
            "nodes": int(args.nodes),
            "warmup": int(args.warmup),
            "steps": int(args.steps),
            "anchor_hz": ANCHOR_HZ,
            "scope": "classical proxy metrics, not quantum entanglement",
        }
    }
    payload.update(run_ablation(args.seeds, args.nodes, args.warmup, args.steps))

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    Path(args.out_md).write_text(to_markdown(payload), encoding="utf-8")

    print(json.dumps({"config": payload["config"], "stats": payload["stats"]}, indent=2))
    print(f"saved: {args.out_json}")
    print(f"saved: {args.out_md}")


if __name__ == "__main__":
    main()
