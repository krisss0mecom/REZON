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


def pvalue_vs_constant(x, c):
    x = np.asarray(x, dtype=np.float64)
    if x.size < 2:
        return math.nan
    if scipy_stats is not None:
        try:
            return float(scipy_stats.ttest_1samp(x, popmean=c).pvalue)
        except Exception:
            pass
    return math.nan


def gen_maxcut(n, edge_p, seed):
    rng = np.random.default_rng(seed)
    # Classical Max-Cut benchmark: non-negative edge weights.
    W = rng.uniform(0.0, 1.0, size=(n, n))
    M = rng.random((n, n)) < edge_p
    W = np.where(M, W, 0.0)
    W = np.triu(W, 1)
    W = W + W.T
    np.fill_diagonal(W, 0.0)
    return W


def eval_maxcut(W, x_pm1):
    # For x in {-1,+1}^n and symmetric W:
    # cut(x) = 1/4 * sum_ij W_ij * (1 - x_i x_j)
    #        = 1/4 * (sum(W) - x^T W x)
    x = np.asarray(x_pm1, dtype=np.float64)
    total_w = float(np.sum(W))
    quad = float(x @ (W @ x))
    return 0.25 * (total_w - quad)


def degree_baseline_partition(W):
    d = np.sum(W, axis=1)
    med = float(np.median(d))
    x = np.where(d >= med, 1.0, -1.0)
    if np.all(x == x[0]):
        x[::2] = 1.0
        x[1::2] = -1.0
    return x


def one_flip_local_search(W, x_init, max_flips=5000):
    """Greedy 1-flip refinement for Max-Cut in +/-1 encoding."""
    x = np.asarray(x_init, dtype=np.float64).copy()
    h = W @ x
    n = x.shape[0]
    flips = 0

    while flips < max_flips:
        gains = x * h  # positive gain means cut improves after flip
        i = int(np.argmax(gains))
        g = float(gains[i])
        if g <= 1e-12:
            break
        s = x[i]
        x[i] = -s
        h = h - 2.0 * s * W[:, i]
        flips += 1
        if flips >= 8 * n:
            break
    return x, flips


def random_baseline_best(W, rng, trials=8):
    n = W.shape[0]
    X = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=(trials, n))
    WX = X @ W
    quad = np.einsum("bi,bi->b", WX, X)
    total_w = float(np.sum(W))
    cuts = 0.25 * (total_w - quad)
    return float(np.max(cuts))


def random_baseline_best_local_search(W, rng, trials=8):
    n = W.shape[0]
    best = -math.inf
    for _ in range(trials):
        x0 = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=n)
        x1, _ = one_flip_local_search(W, x0)
        q = eval_maxcut(W, x1)
        if q > best:
            best = q
    return float(best)


def simulate_phase(n, coupling, noise_amp, leak, anchor_amp, seed, warmup, steps, shil_amp=0.18):
    rng = np.random.default_rng(seed)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    omega = rng.uniform(-0.25, 0.25, size=n)
    t = 0.0
    ensure_anchor_immutable(ANCHOR_HZ)

    total_steps = warmup + steps
    for k in range(total_steps):
        prog = (k / float(max(1, total_steps - 1)))
        # Simple annealing schedule inspired by oscillator-Ising literature:
        # grow coupling/locking, decay noise over time.
        c_eff = coupling * (0.30 + 0.70 * prog)
        noise_eff = noise_amp * (1.0 - 0.85 * prog)
        anchor_eff = anchor_amp * (0.35 + 0.65 * prog)
        shil_eff = shil_amp * (0.25 + 0.75 * prog)

        z = np.mean(np.exp(1j * phi))
        R = np.abs(z)
        theta = np.angle(z)
        dphi = omega + c_eff * R * np.sin(theta - phi)
        # SHIL-like binarization (encourages 0/pi phase states).
        dphi -= shil_eff * np.sin(2.0 * phi)
        if anchor_amp > 0.0:
            dphi += anchor_eff * np.sin(2.0 * np.pi * ANCHOR_HZ * t - phi)
        dphi -= leak * phi
        if noise_amp > 0:
            dphi += rng.normal(0.0, noise_eff, size=n)
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
    graphs = [gen_maxcut(n_nodes, edge_p=0.2, seed=500 + s) for s in range(seeds)]

    for cfg in grid:
        c, noise, leak, anch = cfg
        metric_rows = []
        quality_rows = []
        quality_vs_ls_rows = []
        phase_cut_rows = []
        baseline_cut_rows = []
        baseline_ls_cut_rows = []
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
                shil_amp=0.18,
            )
            m = compute_all_metrics(phi)
            metric_rows.append(m)

            W = graphs[s]
            x = np.where(np.cos(phi) >= 0.0, 1.0, -1.0)
            x_phase_ls, _ = one_flip_local_search(W, x)
            q_phase = eval_maxcut(W, x_phase_ls)

            rng_b = np.random.default_rng(9000 + 131 * s + int(100 * c) + int(1000 * noise) + int(100 * anch))
            q_rand = random_baseline_best(W, rng_b, trials=8)
            x_deg = degree_baseline_partition(W)
            q_deg = eval_maxcut(W, x_deg)
            q_base = float(max(q_rand, q_deg, 1e-12))

            q_rand_ls = random_baseline_best_local_search(W, rng_b, trials=4)
            x_deg_ls, _ = one_flip_local_search(W, x_deg)
            q_deg_ls = eval_maxcut(W, x_deg_ls)
            q_base_ls = float(max(q_rand_ls, q_deg_ls, 1e-12))

            phase_cut_rows.append(q_phase)
            baseline_cut_rows.append(q_base)
            baseline_ls_cut_rows.append(q_base_ls)
            quality_rows.append(float(q_phase / q_base))
            quality_vs_ls_rows.append(float(q_phase / q_base_ls))

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
            "quality_vs_ls_raw": [float(x) for x in quality_vs_ls_rows],
            "quality_vs_ls_mean": float(np.mean(quality_vs_ls_rows)),
            "quality_vs_ls_std": float(np.std(quality_vs_ls_rows)),
            "phase_cut_raw": [float(x) for x in phase_cut_rows],
            "phase_cut_mean": float(np.mean(phase_cut_rows)),
            "phase_cut_std": float(np.std(phase_cut_rows)),
            "baseline_cut_raw": [float(x) for x in baseline_cut_rows],
            "baseline_cut_mean": float(np.mean(baseline_cut_rows)),
            "baseline_cut_std": float(np.std(baseline_cut_rows)),
            "baseline_ls_cut_raw": [float(x) for x in baseline_ls_cut_rows],
            "baseline_ls_cut_mean": float(np.mean(baseline_ls_cut_rows)),
            "baseline_ls_cut_std": float(np.std(baseline_ls_cut_rows)),
            "phase_gap_raw": [float(p - b) for p, b in zip(phase_cut_rows, baseline_cut_rows)],
            "phase_gap_mean": float(np.mean([p - b for p, b in zip(phase_cut_rows, baseline_cut_rows)])),
            "phase_gap_std": float(np.std([p - b for p, b in zip(phase_cut_rows, baseline_cut_rows)])),
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
    q_ls_vals = [r["quality_vs_ls_mean"] for r in results]
    gap_vals = [r["phase_gap_mean"] for r in results]
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
        "p_value_vs_unity": pvalue_vs_constant(np.asarray([q for r in results for q in r["quality_raw"]], dtype=np.float64), 1.0),
    }
    stats_out["quality_vs_ls_baseline"] = {
        "mean": float(np.mean(q_ls_vals)),
        "std": float(np.std(q_ls_vals)),
        "ci95": bootstrap_ci(q_ls_vals),
        "p_value_vs_unity": pvalue_vs_constant(np.asarray([q for r in results for q in r["quality_vs_ls_raw"]], dtype=np.float64), 1.0),
    }
    stats_out["phase_gap"] = {
        "mean": float(np.mean(gap_vals)),
        "std": float(np.std(gap_vals)),
        "ci95": bootstrap_ci(gap_vals),
        "p_value_vs_zero": pvalue_vs_constant(np.asarray([g for r in results for g in r["phase_gap_raw"]], dtype=np.float64), 0.0),
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
        p = v.get("p_value_vs_baseline", v.get("p_value_vs_zero", v.get("p_value_vs_unity", math.nan)))
        p_s = "nan" if p != p else f"{p:.4g}"
        lines.append(f"| {k} | {v['mean']:.6f} | {v['std']:.6f} | [{ci[0]:.6f}, {ci[1]:.6f}] | {p_s} |")
    lines.append("")
    lines.append("## Notes")
    lines.append("- `chsh_proxy` is CHSH-like classical proxy, not QM Bell test.")
    lines.append("- `quality` is phase+local-search cut divided by best simple baseline cut (random-8 vs degree).")
    lines.append("- `quality_vs_ls_baseline` compares phase+local-search to local-search baselines (random-LS / degree-LS).")
    lines.append("- `phase_gap` is phase cut minus best baseline cut.")
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
