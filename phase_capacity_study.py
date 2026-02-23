#!/usr/bin/env python3
"""
Phase Hopfield Capacity Study — empirical storage capacity on S¹.

════════════════════════════════════════════════════════════════════════
RESEARCH QUESTION 1: What is the storage capacity of Phase Hopfield?

Classical Hopfield (Ising, ±1):
  P_max ≈ 0.138 · N  (Amit, Gutfreund, Sompolinsky 1985)

Phase Hopfield ({0,π} attractors):
  E = −½ Σ W_ij cos(φ_i − φ_j)
  Restricted to {0,π}^N: cos(φ_i−φ_j) = cos(φ_i)·cos(φ_j) [sin=0]
  → E = −½ Σ W_ij s_i s_j  ← identical to Hopfield Ising!
  → PREDICTION: capacity ≈ 0.138·N  (same universality class)

But: patterns on S¹ carry MORE information per neuron than binary.
  Each node can be ANY angle, not just {0,π}.
  Continuous Phase Hopfield (XY model) has different capacity.
  → MEASURED EMPIRICALLY HERE.

════════════════════════════════════════════════════════════════════════
RESULT FORMAT:
  For each N ∈ {16, 32, 64, 128}:
    For each P from 1 to P_max:
      Recall rate at 10% flip  →  find critical P* (rate drops < 0.5)
  Plot: P*/N vs N → should converge to ~0.138 for binary patterns.

Anchor: 200 Hz (immutable REZON constraint).
"""

import argparse
import json
import os
import time

import numpy as np

from phase_gate_universal import ANCHOR_HZ, ensure_anchor_immutable
from phase_hopfield import PhaseHopfield

ensure_anchor_immutable(ANCHOR_HZ)

TWO_PI = 2.0 * np.pi


# ── Single capacity probe ─────────────────────────────────────────────────────

def probe_capacity(
    N: int,
    P: int,
    *,
    flip_fraction: float = 0.10,
    trials: int = 3,
    warmup_steps: int = 4000,
    recall_steps: int = 8000,
    K: float = 3.0,
    dt: float = 0.001,
    a_anchor: float = 0.08,
    seed: int = 0,
) -> dict:
    """
    Store P patterns in N-node Phase Hopfield, measure recall at flip_fraction.
    Returns success_rate ∈ [0, 1].
    """
    rng = np.random.default_rng(seed)
    patterns = [rng.integers(0, 2, N).tolist() for _ in range(P)]

    net = PhaseHopfield(
        N, patterns, K=K, dt=dt, a_anchor=a_anchor, seed=seed,
    )

    ok = 0
    total = trials * P
    for trial in range(trials):
        for pat_idx in range(P):
            r = net.recall(
                pat_idx,
                flip_fraction=flip_fraction,
                warmup_steps=warmup_steps,
                recall_steps=recall_steps,
            )
            if r["recovered"]:
                ok += 1

    return {
        "N": N, "P": P,
        "success_rate": round(ok / total, 3),
        "ok": ok,
        "total": total,
        "alpha": round(P / N, 3),   # loading ratio
    }


# ── Capacity sweep ────────────────────────────────────────────────────────────

def capacity_sweep(
    N: int,
    *,
    P_max_factor: float = 0.35,   # sweep up to P = 0.35·N
    flip_fraction: float = 0.10,
    trials: int = 3,
    warmup_steps: int = 4000,
    recall_steps: int = 8000,
    K: float = 3.0,
    dt: float = 0.001,
    a_anchor: float = 0.08,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """
    Sweep P from 1 to floor(P_max_factor·N), find critical capacity P*.

    Critical capacity P*: largest P where success_rate >= 0.5.
    """
    P_max = max(2, int(np.floor(P_max_factor * N)))
    theoretical = round(0.138 * N, 1)

    rows = []
    P_star = 0   # critical capacity

    if verbose:
        print(f"  N={N:4d}  sweeping P=1..{P_max}  (classical limit={theoretical})")

    for P in range(1, P_max + 1):
        r = probe_capacity(
            N, P,
            flip_fraction=flip_fraction,
            trials=trials,
            warmup_steps=warmup_steps,
            recall_steps=recall_steps,
            K=K, dt=dt, a_anchor=a_anchor,
            seed=seed + P * 17,
        )
        rows.append(r)
        if r["success_rate"] >= 0.5:
            P_star = P

        marker = " ← P*" if r["success_rate"] >= 0.5 and (
            P == P_max or rows[-1]["success_rate"] >= 0.5
        ) else ""
        if verbose:
            bar = "█" * int(r["success_rate"] * 20)
            print(f"    P={P:3d}  α={r['alpha']:.3f}  "
                  f"recall={r['success_rate']:.2f}  {bar}{marker}")

    alpha_star = round(P_star / N, 3) if P_star > 0 else 0.0

    return {
        "N": N,
        "P_star": P_star,
        "alpha_star": alpha_star,
        "theoretical_Pmax": theoretical,
        "theoretical_alpha": 0.138,
        "rows": rows,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Phase Hopfield Capacity Study"
    )
    p.add_argument("--sizes",    type=int,   nargs="+", default=[16, 32, 64])
    p.add_argument("--trials",   type=int,   default=3)
    p.add_argument("--flip",     type=float, default=0.10)
    p.add_argument("--warmup",   type=int,   default=4000)
    p.add_argument("--recall",   type=int,   default=8000)
    p.add_argument("--K",        type=float, default=3.0)
    p.add_argument("--dt",       type=float, default=0.001)
    p.add_argument("--a-anchor", type=float, default=0.08)
    p.add_argument("--out-json", type=str,
                   default="reports/phase_capacity_report.json")
    args = p.parse_args()

    net_kw = dict(
        flip_fraction=args.flip,
        trials=args.trials,
        warmup_steps=args.warmup,
        recall_steps=args.recall,
        K=args.K, dt=args.dt, a_anchor=args.a_anchor,
    )

    print("\n" + "═" * 64)
    print("  PHASE HOPFIELD CAPACITY STUDY")
    print("  Binary patterns {0,π}^N, flip=10%, measure P* where recall≥50%")
    print(f"  Theoretical prediction (classical Hopfield): α* ≈ 0.138")
    print("═" * 64)

    all_results = {}
    t0 = time.time()

    for N in args.sizes:
        print(f"\n[N={N}]")
        res = capacity_sweep(N, **net_kw)
        all_results[f"N{N}"] = res
        print(f"  → P* = {res['P_star']}  α* = {res['alpha_star']:.3f}  "
              f"(theory: {res['theoretical_alpha']:.3f})")

    elapsed = time.time() - t0

    # ── Summary table ────────────────────────────────────────────────────────
    print(f"\n{'═' * 64}")
    print("  CAPACITY SUMMARY:")
    print(f"  {'N':>6}  {'P*':>6}  {'α* (meas)':>12}  {'α* (theory)':>14}")
    print(f"  {'─'*6}  {'─'*6}  {'─'*12}  {'─'*14}")
    for N in args.sizes:
        r = all_results[f"N{N}"]
        print(f"  {N:>6}  {r['P_star']:>6}  {r['alpha_star']:>12.3f}  "
              f"{r['theoretical_alpha']:>14.3f}")

    print(f"\n  Theoretical α* = 0.138 (Amit-Gutfreund-Sompolinsky 1985)")
    print(f"  Phase Hopfield: binary {'{0,π}'}^N ≡ Ising → same universality class")
    print(f"  [{elapsed:.0f}s total]")
    print("═" * 64)

    # ── Save ─────────────────────────────────────────────────────────────────
    report = {
        "title": "Phase Hopfield Capacity Study",
        "question": "What is the empirical storage capacity of Phase Hopfield?",
        "prediction": (
            "Binary {0,π}^N patterns: capacity ≡ classical Hopfield (0.138N). "
            "Phase Hopfield restricted to {0,π} reduces exactly to Ising model. "
            "Empirical α* should match 0.138."
        ),
        "params": {**net_kw, "sizes": args.sizes},
        "results": all_results,
        "anchor_hz": ANCHOR_HZ,
        "ts": int(time.time()),
        "elapsed_s": round(elapsed, 1),
    }
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n  Saved → {args.out_json}")


if __name__ == "__main__":
    main()
