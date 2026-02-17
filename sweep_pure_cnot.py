#!/usr/bin/env python3
import argparse
import itertools
import json
import math
import os
from typing import Dict, List, Tuple

import numpy as np

from reservoir_phase_cnot_pure import run_truth_table


def evaluate_params(params: Dict[str, float], steps: int, seeds: int) -> Dict[str, float]:
    scores = []
    for s in range(seeds):
        sc, _ = run_truth_table(seed=s, steps=steps, **params)
        scores.append(int(sc))
    scores_arr = np.asarray(scores, dtype=np.float64)
    pass_4 = float(np.mean(scores_arr == 4.0))
    pass_ge3 = float(np.mean(scores_arr >= 3.0))
    mean_score = float(np.mean(scores_arr))
    std_score = float(np.std(scores_arr))
    dist = {str(i): int(np.sum(scores_arr == i)) for i in range(5)}
    return {
        "pass_rate_4of4": pass_4,
        "pass_rate_ge3": pass_ge3,
        "mean_score": mean_score,
        "std_score": std_score,
        "distribution": dist,
    }


def build_grid() -> List[Dict[str, float]]:
    # Focus around physically plausible region for this 2-node pure PoC.
    coupling_vals = [1.5, 2.0, 3.0, 4.5, 6.0, 8.0]
    anti_vals = [8.0, 12.0, 16.0, 22.0, 30.0, 40.0]
    anchor_vals = [0.05, 0.1, 0.2, 0.4]
    dt_vals = [0.001, 0.0025, 0.005, 0.01]
    noise_vals = [0.0, 0.001, 0.003, 0.01, 0.05]
    control_window_div_vals = [1.2, 1.5, 1.8, 2.2]
    direction_threshold_vals = [0.0, 0.05, 0.1, 0.2]
    inject_window_vals = [0.02, 0.05, 0.1]

    grid = []
    for vals in itertools.product(
        coupling_vals,
        anti_vals,
        anchor_vals,
        dt_vals,
        noise_vals,
        control_window_div_vals,
        direction_threshold_vals,
        inject_window_vals,
    ):
        (
            coupling,
            anti_coupling,
            a_anchor,
            dt,
            noise_amp,
            control_window_div,
            direction_threshold,
            inject_window_s,
        ) = vals
        grid.append(
            {
                "coupling": float(coupling),
                "anti_coupling": float(anti_coupling),
                "a_anchor": float(a_anchor),
                "dt": float(dt),
                "noise_amp": float(noise_amp),
                "control_window_div": float(control_window_div),
                "direction_threshold": float(direction_threshold),
                "inject_window_s": float(inject_window_s),
            }
        )
    return grid


def main() -> None:
    p = argparse.ArgumentParser(description="Sweep pure CNOT-like phase PoC without RLS.")
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--seeds", type=int, default=60)
    p.add_argument("--max-trials", type=int, default=120)
    p.add_argument("--target-pass-4", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-json", type=str, default="results/pure_cnot_sweep.json")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    all_grid = build_grid()
    rng.shuffle(all_grid)
    trials = all_grid[: max(1, int(args.max_trials))]

    best = None
    rows = []
    hit_target = False

    print(f"sweep_trials={len(trials)} seeds_per_trial={args.seeds} steps={args.steps}")
    for i, params in enumerate(trials, start=1):
        m = evaluate_params(params, steps=args.steps, seeds=args.seeds)
        row = {"idx": i, "params": params, "metrics": m}
        rows.append(row)

        score_key = (m["pass_rate_4of4"], m["pass_rate_ge3"], m["mean_score"], -m["std_score"])
        if best is None:
            best = row
        else:
            b = best["metrics"]
            best_key = (b["pass_rate_4of4"], b["pass_rate_ge3"], b["mean_score"], -b["std_score"])
            if score_key > best_key:
                best = row

        print(
            f"[{i:03d}/{len(trials):03d}] pass4={m['pass_rate_4of4']:.3f} "
            f"pass>=3={m['pass_rate_ge3']:.3f} mean={m['mean_score']:.3f} std={m['std_score']:.3f}"
        )

        if m["pass_rate_4of4"] >= args.target_pass_4:
            hit_target = True
            print(f"target_hit at trial {i}")
            break

    rows_sorted = sorted(
        rows,
        key=lambda r: (
            r["metrics"]["pass_rate_4of4"],
            r["metrics"]["pass_rate_ge3"],
            r["metrics"]["mean_score"],
            -r["metrics"]["std_score"],
        ),
        reverse=True,
    )

    payload = {
        "config": vars(args),
        "hit_target": bool(hit_target),
        "best": best,
        "top10": rows_sorted[:10],
        "trials_ran": len(rows),
    }

    if args.out_json:
        out_dir = os.path.dirname(args.out_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"saved: {args.out_json}")

    if best is not None:
        bm = best["metrics"]
        print("best:")
        print(
            f"pass4={bm['pass_rate_4of4']:.3f} pass>=3={bm['pass_rate_ge3']:.3f} "
            f"mean={bm['mean_score']:.3f} std={bm['std_score']:.3f}"
        )
        print(json.dumps(best["params"], indent=2))


if __name__ == "__main__":
    main()
