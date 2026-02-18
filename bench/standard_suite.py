#!/usr/bin/env python3
import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

try:
    from scipy import stats as scipy_stats
except Exception:
    scipy_stats = None

ANCHOR_HZ = 200.0


@dataclass
class RCParams:
    n_nodes: int = 1000
    coupling: float = 1.8
    leak: float = 0.02
    anchor_amp: float = 0.4
    dt: float = 0.01
    warmup_steps: int = 40
    steps: int = 24
    seed: int = 42


def ensure_anchor_immutable(freq_hz: float) -> None:
    if abs(float(freq_hz) - ANCHOR_HZ) > 1e-6:
        raise ValueError(f"anchor must be exactly {ANCHOR_HZ} Hz")


def energy_proxy_j(time_s: float, power_w: float = 15.0) -> float:
    return float(time_s * power_w)


def bootstrap_ci(values, alpha=0.05, n_boot=500):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return [math.nan, math.nan]
    rng = np.random.default_rng(123)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=arr.size, replace=True)
        means.append(float(np.mean(sample)))
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return [lo, hi]


def paired_pvalue(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size != b.size or a.size == 0:
        return math.nan
    if a.size < 2:
        return math.nan
    if scipy_stats is not None:
        try:
            return float(scipy_stats.ttest_rel(a, b).pvalue)
        except Exception:
            pass
    # fallback: sign test approx
    d = a - b
    nz = d[d != 0]
    if nz.size == 0:
        return 1.0
    k = int(np.sum(nz > 0))
    n = int(nz.size)
    # two-sided binomial tail with p=0.5
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / (2 ** n)
    return float(min(1.0, 2.0 * tail))


# --------------------------
# Problem instances
# --------------------------

def gen_maxcut_instance(n: int, edge_p: float, seed: int):
    rng = np.random.default_rng(seed)
    W = rng.uniform(-1.0, 1.0, size=(n, n))
    M = rng.random((n, n)) < edge_p
    W = np.where(M, W, 0.0)
    W = np.triu(W, 1)
    W = W + W.T
    np.fill_diagonal(W, 0.0)
    return W


def eval_maxcut(W: np.ndarray, x: np.ndarray) -> float:
    # x in {-1,+1}
    n = W.shape[0]
    val = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if x[i] != x[j]:
                val += W[i, j]
    return float(val)


def gen_qubo_instance(n: int, density: float, seed: int):
    rng = np.random.default_rng(seed)
    Q = rng.normal(0.0, 1.0, size=(n, n))
    M = rng.random((n, n)) < density
    Q = np.where(M, Q, 0.0)
    Q = np.triu(Q)
    Q = Q + np.triu(Q, 1).T
    return Q


def eval_qubo(Q: np.ndarray, x: np.ndarray) -> float:
    # maximize x^T Q x with x in {0,1}
    return float(x @ Q @ x)


def gen_sat_3cnf(n_vars: int, n_clauses: int, seed: int):
    rng = np.random.default_rng(seed)
    clauses = []
    for _ in range(n_clauses):
        vars_idx = rng.choice(n_vars, size=3, replace=False)
        signs = rng.choice([-1, 1], size=3)
        clauses.append([(int(signs[i]), int(vars_idx[i])) for i in range(3)])
    return clauses


def eval_sat(clauses, a01: np.ndarray) -> float:
    sat = 0
    for c in clauses:
        ok = False
        for sign, idx in c:
            bit = int(a01[idx])
            lit = bit if sign > 0 else (1 - bit)
            if lit == 1:
                ok = True
                break
        sat += int(ok)
    return float(sat)


# --------------------------
# Solvers / ablations
# --------------------------

def random_solver(n, eval_fn, rng, domain="pm1", budget=1000):
    best = -1e30
    for _ in range(budget):
        if domain == "pm1":
            x = rng.choice([-1.0, 1.0], size=n)
        else:
            x = rng.integers(0, 2, size=n)
        best = max(best, eval_fn(x))
    return float(best)


def hillclimb_solver(n, eval_fn, rng, domain="pm1", steps=500):
    if domain == "pm1":
        x = rng.choice([-1.0, 1.0], size=n)
    else:
        x = rng.integers(0, 2, size=n)
    best = eval_fn(x)
    for _ in range(steps):
        i = int(rng.integers(0, n))
        x2 = x.copy()
        if domain == "pm1":
            x2[i] *= -1
        else:
            x2[i] = 1 - x2[i]
        v2 = eval_fn(x2)
        if v2 >= best:
            x, best = x2, v2
    return float(best)


def anneal_solver(n, eval_fn, rng, domain="pm1", steps=1000):
    if domain == "pm1":
        x = rng.choice([-1.0, 1.0], size=n)
    else:
        x = rng.integers(0, 2, size=n)
    cur = eval_fn(x)
    best = cur
    T0 = 1.0
    for t in range(steps):
        i = int(rng.integers(0, n))
        x2 = x.copy()
        if domain == "pm1":
            x2[i] *= -1
        else:
            x2[i] = 1 - x2[i]
        v2 = eval_fn(x2)
        d = v2 - cur
        T = T0 * (1.0 - t / max(1, steps)) + 1e-6
        if d >= 0 or rng.random() < math.exp(d / max(T, 1e-9)):
            x, cur = x2, v2
            best = max(best, cur)
    return float(best)


def rc_score_vector(xv: np.ndarray, params: RCParams, use_anchor=True):
    # simple RC-inspired phase dynamics projection to scalar score
    rng = np.random.default_rng(params.seed)
    n = params.n_nodes
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    W = rng.normal(0.0, 1.0, size=(n, xv.size))
    W = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)
    inb = W @ xv
    t = 0.0
    # Warmup: settle reservoir dynamics before scoring readout.
    # Use Kuramoto order-parameter form (O(N)) instead of dense pairwise O(N^2).
    for _ in range(max(0, int(params.warmup_steps))):
        z = np.mean(np.exp(1j * phi))
        R = np.abs(z)
        theta = np.angle(z)
        dphi = params.coupling * R * np.sin(theta - phi)
        if use_anchor:
            ensure_anchor_immutable(ANCHOR_HZ)
            dphi += params.anchor_amp * np.sin(2 * np.pi * ANCHOR_HZ * t - phi)
        dphi += -params.leak * phi
        phi = (phi + params.dt * dphi) % (2.0 * np.pi)
        t += params.dt

    for _ in range(params.steps):
        z = np.mean(np.exp(1j * phi))
        R = np.abs(z)
        theta = np.angle(z)
        dphi = params.coupling * R * np.sin(theta - phi)
        if use_anchor:
            ensure_anchor_immutable(ANCHOR_HZ)
            dphi += params.anchor_amp * np.sin(2 * np.pi * ANCHOR_HZ * t - phi)
        dphi += 0.25 * np.tanh(inb) - params.leak * phi
        phi = (phi + params.dt * dphi) % (2.0 * np.pi)
        t += params.dt
    return float(np.mean(np.cos(phi)))


def rc_search_solver(
    n,
    eval_fn,
    rng,
    domain="pm1",
    budget=800,
    use_anchor=True,
    use_rls=False,
    rls_steps=100,
    polish_steps=80,
    rc_nodes=1000,
    rc_warmup_steps=40,
):
    # candidate pool from random states + RC scoring, then local refine
    cands = []
    params = RCParams(
        n_nodes=int(rc_nodes),
        warmup_steps=int(rc_warmup_steps),
        seed=int(rng.integers(0, 1_000_000)),
    )
    for _ in range(budget):
        if domain == "pm1":
            x = rng.choice([-1.0, 1.0], size=n)
        else:
            x = rng.integers(0, 2, size=n)
        s = rc_score_vector(x.astype(np.float64), params, use_anchor=use_anchor)
        cands.append((s, x))

    # top-K without full sort for speed
    K = min(64, len(cands))
    scores = np.array([c[0] for c in cands])
    top_idx = np.argpartition(scores, -K)[-K:]

    # optional linear head (RLS-like) over rc score + simple stats
    w = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    P = np.eye(3) * 10.0
    lam = 0.995
    if use_rls:
        for _ in range(rls_steps):
            i = int(rng.integers(0, len(cands)))
            x = cands[i][1]
            feat = np.array([cands[i][0], float(np.mean(x)), float(np.std(x))], dtype=np.float64)
            y = eval_fn(x)
            yhat = float(feat @ w)
            err = y - yhat
            px = P @ feat
            denom = lam + float(feat @ px)
            k = px / max(denom, 1e-9)
            w = w + k * err
            P = (P - np.outer(k, feat) @ P) / lam

    best = -1e30
    for i in top_idx:
        x = cands[int(i)][1].copy()
        rc_cached = cands[int(i)][0]
        # local polishing
        for _ in range(polish_steps):
            j = int(rng.integers(0, n))
            x2 = x.copy()
            if domain == "pm1":
                x2[j] *= -1
            else:
                x2[j] = 1 - x2[j]
            v1 = eval_fn(x)
            v2 = eval_fn(x2)
            if use_rls:
                rc2 = rc_score_vector(x2.astype(np.float64), params, use_anchor)
                f1 = np.array([rc_cached, float(np.mean(x)), float(np.std(x))])
                f2 = np.array([rc2, float(np.mean(x2)), float(np.std(x2))])
                r1 = float(f1 @ w)
                r2 = float(f2 @ w)
                if v2 > v1 or r2 >= r1:
                    x = x2
                    rc_cached = rc2
            else:
                if v2 >= v1:
                    x = x2
        best = max(best, eval_fn(x))
    return float(best)


ABLATIONS = {
    "no_rc": dict(kind="anneal"),
    "rc_no_anchor": dict(kind="rc", use_anchor=False, use_rls=False),
    "rc_anchor": dict(kind="rc", use_anchor=True, use_rls=False),
    "rc_anchor_rls": dict(kind="rc", use_anchor=True, use_rls=True),
}


def solve_with_config(
    cfg_name: str,
    n: int,
    eval_fn: Callable,
    rng,
    domain: str,
    anneal_steps: int,
    rc_budget: int,
    rc_polish_steps: int,
    rc_rls_steps: int,
    rc_nodes: int,
    rc_warmup_steps: int,
):
    cfg = ABLATIONS[cfg_name]
    if cfg["kind"] == "anneal":
        return anneal_solver(n, eval_fn, rng, domain=domain, steps=anneal_steps)
    return rc_search_solver(
        n,
        eval_fn,
        rng,
        domain=domain,
        budget=rc_budget,
        use_anchor=cfg.get("use_anchor", True),
        use_rls=cfg.get("use_rls", False),
        rls_steps=rc_rls_steps,
        polish_steps=rc_polish_steps,
        rc_nodes=rc_nodes,
        rc_warmup_steps=rc_warmup_steps,
    )


def run_suite(instances=8, seeds=8, fast_mode=False, rc_nodes=1000, rc_warmup_steps=40):
    rng = np.random.default_rng(2026)
    rows = []
    per_problem = {"maxcut": [], "qubo": [], "sat": []}

    for problem in ("maxcut", "qubo", "sat"):
        for inst_id in range(instances):
            n = 28 if problem != "sat" else 32
            inst_seed = 10_000 + 97 * inst_id + (0 if problem == "maxcut" else 1 if problem == "qubo" else 2)
            if problem == "maxcut":
                W = gen_maxcut_instance(n, edge_p=0.25, seed=inst_seed)
                eval_fn = lambda x: eval_maxcut(W, x)
                domain = "pm1"
            elif problem == "qubo":
                Q = gen_qubo_instance(n, density=0.30, seed=inst_seed)
                eval_fn = lambda x: eval_qubo(Q, x)
                domain = "01"
            else:
                clauses = gen_sat_3cnf(n_vars=n, n_clauses=4 * n, seed=inst_seed)
                eval_fn = lambda x: eval_sat(clauses, x)
                domain = "01"

            for ab in ABLATIONS:
                vals, times, energies = [], [], []
                for s in range(seeds):
                    rr = np.random.default_rng(50_000 + 13 * s + 997 * inst_id)
                    if fast_mode:
                        anneal_steps = 300
                        rc_budget = 220
                        rc_polish_steps = 24
                        rc_rls_steps = 40
                    else:
                        anneal_steps = 1000
                        rc_budget = 600
                        rc_polish_steps = 80
                        rc_rls_steps = 100
                    t0 = time.perf_counter()
                    v = solve_with_config(
                        ab,
                        n,
                        eval_fn,
                        rr,
                        domain,
                        anneal_steps=anneal_steps,
                        rc_budget=rc_budget,
                        rc_polish_steps=rc_polish_steps,
                        rc_rls_steps=rc_rls_steps,
                        rc_nodes=rc_nodes,
                        rc_warmup_steps=rc_warmup_steps,
                    )
                    dt = time.perf_counter() - t0
                    vals.append(float(v))
                    times.append(float(dt))
                    energies.append(energy_proxy_j(dt, power_w=15.0))

                row = {
                    "problem": problem,
                    "instance_id": inst_id,
                    "ablation": ab,
                    "quality_mean": float(np.mean(vals)),
                    "quality_std": float(np.std(vals)),
                    "quality_ci95": bootstrap_ci(vals),
                    "time_mean_s": float(np.mean(times)),
                    "time_std_s": float(np.std(times)),
                    "energy_mean_j": float(np.mean(energies)),
                    "stability_cv": float(np.std(vals) / max(abs(np.mean(vals)), 1e-9)),
                    "raw_quality": vals,
                    "raw_time": times,
                }
                rows.append(row)
                per_problem[problem].append(row)

    # paired stats versus no_rc
    stats_rows = []
    for problem in per_problem:
        base = [r for r in per_problem[problem] if r["ablation"] == "no_rc"]
        for ab in ("rc_no_anchor", "rc_anchor", "rc_anchor_rls"):
            tgt = [r for r in per_problem[problem] if r["ablation"] == ab]
            base_q = [x["quality_mean"] for x in base]
            tgt_q = [x["quality_mean"] for x in tgt]
            p = paired_pvalue(np.array(tgt_q), np.array(base_q))
            stats_rows.append(
                {
                    "problem": problem,
                    "compare": f"{ab} vs no_rc",
                    "delta_quality_mean": float(np.mean(np.array(tgt_q) - np.array(base_q))),
                    "p_value": p,
                }
            )

    return {"rows": rows, "stats": stats_rows}


def to_markdown(report):
    lines = [
        "# Standard Suite Benchmark",
        "",
        "Claims scope: quantum-inspired analog search heuristic (not QC replacement).",
        "",
        "## Aggregate Stats vs no_rc",
        "",
        "| problem | compare | delta quality mean | p-value |",
        "|---|---|---:|---:|",
    ]
    for s in report["stats"]:
        pv = s["p_value"]
        pv_s = f"{pv:.4g}" if pv == pv else "nan"
        lines.append(f"| {s['problem']} | {s['compare']} | {s['delta_quality_mean']:.6f} | {pv_s} |")
    lines.append("")
    lines.append("## Notes")
    lines.append("- Quality/time/energy/stability measured across many seeds.")
    lines.append("- Ablations include no RC, RC without anchor, RC with anchor, RC with anchor+RLS.")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description="Standard benchmark suite for QUBO/Max-Cut/SAT with ablations.")
    ap.add_argument("--instances", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--fast-mode", type=int, default=1)
    ap.add_argument("--rc-nodes", type=int, default=1000)
    ap.add_argument("--rc-warmup-steps", type=int, default=40)
    ap.add_argument("--out-json", type=str, default="reports/standard_suite_report.json")
    ap.add_argument("--out-md", type=str, default="reports/standard_suite_report.md")
    args = ap.parse_args()

    report = run_suite(
        instances=args.instances,
        seeds=args.seeds,
        fast_mode=bool(args.fast_mode),
        rc_nodes=int(args.rc_nodes),
        rc_warmup_steps=int(args.rc_warmup_steps),
    )
    payload = {
        "config": {
            "instances": int(args.instances),
            "seeds": int(args.seeds),
            "rc_nodes": int(args.rc_nodes),
            "rc_warmup_steps": int(args.rc_warmup_steps),
            "anchor_hz": ANCHOR_HZ,
            "claim_scope": "quantum-inspired analog search heuristic; not QC replacement",
        },
        **report,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    Path(args.out_md).write_text(to_markdown(payload), encoding="utf-8")

    print(json.dumps({"config": payload["config"], "stats": payload["stats"]}, indent=2))
    print(f"saved: {args.out_json}")
    print(f"saved: {args.out_md}")


if __name__ == "__main__":
    main()
