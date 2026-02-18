#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np

from cnot_rls import RCConfig, rls_train_eval
from cnot_variant_audit import run_truth_table


def score_stats(scores):
    arr = np.asarray(scores, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "pass4_rate": float(np.mean(arr == 4)),
        "distribution": {str(i): int(np.sum(arr == i)) for i in range(5)},
    }


def run_pure_and_xordriven(eval_seeds, warmup, steps, base_params):
    out = {}
    for mode in ("pure", "xor_driven"):
        scores = []
        for seed in range(eval_seeds):
            sc, _ = run_truth_table(seed, mode, warmup, steps, base_params)
            scores.append(sc)
        out[mode] = score_stats(scores)
    return out


def run_rls_grid(eval_seeds, noise_levels, n_values, train_steps, warmup, collect):
    rows = []
    for n in n_values:
        for noise in noise_levels:
            scores = []
            for seed in range(eval_seeds):
                cfg = RCConfig(
                    n=int(n),
                    coupling=1.9,
                    a_anchor=1.0,
                    leak=0.01,
                    input_gain=2.2,
                    noise_amp=float(noise),
                    seed=int(seed),
                )
                res = rls_train_eval(
                    cfg,
                    warmup=warmup,
                    collect=collect,
                    train_steps=train_steps,
                    rls_lambda=0.995,
                    seed=seed,
                )
                scores.append(int(res["score_4"]))
            rows.append(
                {
                    "n": int(n),
                    "noise_amp": float(noise),
                    **score_stats(scores),
                }
            )
    return rows


def to_markdown(report):
    lines = []
    lines.append("# REZON Proof Benchmark")
    lines.append("")
    lines.append("## Pure vs XOR-driven")
    lines.append("")
    lines.append("| mode | mean_score (/4) | std | pass4_rate |")
    lines.append("|---|---:|---:|---:|")
    for mode in ("pure", "xor_driven"):
        m = report["phase_modes"][mode]
        lines.append(f"| {mode} | {m['mean']:.3f} | {m['std']:.3f} | {m['pass4_rate']:.3f} |")
    lines.append("")
    lines.append("## RLS Robustness Grid")
    lines.append("")
    lines.append("| n | noise_amp | mean_score (/4) | std | pass4_rate |")
    lines.append("|---:|---:|---:|---:|---:|")
    for row in report["rls_grid"]:
        lines.append(
            f"| {row['n']} | {row['noise_amp']:.3f} | {row['mean']:.3f} | {row['std']:.3f} | {row['pass4_rate']:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Run proof-style benchmarks for REZON phase computing.")
    ap.add_argument("--eval-seeds", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5000)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--train-steps", type=int, default=1200)
    ap.add_argument("--collect", type=int, default=20)
    ap.add_argument("--out-json", type=str, default="results/rezon_proof_benchmark.json")
    ap.add_argument("--out-md", type=str, default="results/rezon_proof_benchmark.md")
    args = ap.parse_args()

    base_params = {
        "coupling": 2.0,
        "a_anchor": 0.4,
        "dt": 0.001,
        "noise_amp": 0.05,
        "phase_leak": 0.01,
        "bias_scale": 5.0,
        "control_force": 3.0,
        "anti_gain": 2.2,
        "control_threshold": 0.6,
        "control_window_div": 1.5,
        "inject_window_s": 0.05,
        "f_anchor": 200.0,
        "seed": 42,
    }

    phase_modes = run_pure_and_xordriven(
        eval_seeds=args.eval_seeds,
        warmup=args.warmup,
        steps=args.steps,
        base_params=base_params,
    )

    rls_grid = run_rls_grid(
        eval_seeds=args.eval_seeds,
        noise_levels=[0.0, 0.01, 0.03, 0.05, 0.08],
        n_values=[16, 24, 40],
        train_steps=args.train_steps,
        warmup=120,
        collect=args.collect,
    )

    report = {
        "config": {
            "eval_seeds": int(args.eval_seeds),
            "phase_warmup": int(args.warmup),
            "phase_steps": int(args.steps),
            "rls_train_steps": int(args.train_steps),
            "rls_collect": int(args.collect),
        },
        "phase_modes": phase_modes,
        "rls_grid": rls_grid,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    out_md = Path(args.out_md)
    out_md.write_text(to_markdown(report), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"saved: {out_json}")
    print(f"saved: {out_md}")


if __name__ == "__main__":
    main()
