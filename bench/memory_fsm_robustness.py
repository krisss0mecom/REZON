#!/usr/bin/env python3
import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from phase_automaton import PhaseAutomaton
from phase_dlatch import PhaseDLatch, PhaseRegister
from phase_turing_demo import run_turing_demo


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    den = 1.0 + (z * z) / n
    ctr = p + (z * z) / (2.0 * n)
    rad = z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * n)) / n)
    lo = max(0.0, (ctr - rad) / den)
    hi = min(1.0, (ctr + rad) / den)
    return (float(lo), float(hi))


def test_latch(seed: int, noise: float, hold_steps: int) -> Dict:
    latch = PhaseDLatch(seed=seed, noise_amp=noise)
    seq = [0, 1, 0, 1, 0]
    ok = True
    bad = []
    for i, bit in enumerate(seq):
        latch.write(bit, warmup=3000)
        got = latch.hold(n_steps=hold_steps)
        if got != bit:
            ok = False
            bad.append({"step": i + 1, "expected": bit, "got": got})
    return {"ok": ok, "failures": bad[:3]}


def test_register(seed: int, noise: float, hold_steps: int) -> Dict:
    reg = PhaseRegister(8, seed=seed, noise_amp=noise)
    pattern = [1, 0, 1, 1, 0, 1, 0, 0]
    reg.write_all(pattern, warmup=3000)
    read_back = reg.read_all(hold_steps=hold_steps)
    ok = read_back == pattern
    return {"ok": ok, "expected": pattern, "got": read_back if not ok else None}


def test_automaton(seed: int, noise: float, transitions: int) -> Dict:
    fsm = PhaseAutomaton(seed=seed, noise_amp=noise)
    bad = []
    for i in range(transitions):
        got = fsm.tick()
        exp = (i + 1) % 3
        if got != exp:
            bad.append({"step": i + 1, "expected": exp, "got": got})
    return {"ok": len(bad) == 0, "failures": bad[:5]}


def summarize_boolean(results: List[Dict]) -> Dict:
    n = len(results)
    k = sum(1 for r in results if r.get("ok"))
    ci_lo, ci_hi = wilson_ci(k, n)
    failures = [r for r in results if not r.get("ok")]
    return {
        "n_trials": n,
        "n_pass": k,
        "pass_rate": (k / n) if n else 0.0,
        "pass_rate_ci95": [ci_lo, ci_hi],
        "n_fail": n - k,
        "failure_examples": failures[:5],
    }


def run_suite(
    seeds: int,
    noise_grid: List[float],
    hold_steps: int,
    automaton_transitions: int,
    turing_noises: List[float],
    turing_extra_random: int,
) -> Dict:
    all_out = {
        "latch": {},
        "register": {},
        "automaton": {},
        "turing_demo": {},
    }
    per_metric = {
        "latch": [],
        "register": [],
        "automaton": [],
        "turing_demo": [],
    }

    for noise in noise_grid:
        latch_rows = [test_latch(seed=s, noise=noise, hold_steps=hold_steps) for s in range(seeds)]
        reg_rows = [test_register(seed=s, noise=noise, hold_steps=hold_steps) for s in range(seeds)]
        auto_rows = [
            test_automaton(seed=s, noise=noise, transitions=automaton_transitions) for s in range(seeds)
        ]
        all_out["latch"][str(noise)] = summarize_boolean(latch_rows)
        all_out["register"][str(noise)] = summarize_boolean(reg_rows)
        all_out["automaton"][str(noise)] = summarize_boolean(auto_rows)
        per_metric["latch"].append(all_out["latch"][str(noise)]["pass_rate"])
        per_metric["register"].append(all_out["register"][str(noise)]["pass_rate"])
        per_metric["automaton"].append(all_out["automaton"][str(noise)]["pass_rate"])

    for noise in turing_noises:
        rows = []
        for s in range(seeds):
            r = run_turing_demo(
                noise_amp=noise,
                seed=s,
                n_extra_random=turing_extra_random,
                verbose=False,
            )
            rows.append({"ok": bool(r.get("all_correct")), "n_cases": r.get("n_cases"), "n_correct": r.get("n_correct")})
        all_out["turing_demo"][str(noise)] = summarize_boolean(rows)
        per_metric["turing_demo"].append(all_out["turing_demo"][str(noise)]["pass_rate"])

    envelope = {}
    for key, vals in per_metric.items():
        if not vals:
            continue
        envelope[key] = {
            "min_pass_rate_over_sweep": float(min(vals)),
            "mean_pass_rate_over_sweep": float(sum(vals) / len(vals)),
            "max_pass_rate_over_sweep": float(max(vals)),
        }
    all_out["failure_envelope"] = envelope
    return all_out


def markdown_report(rep: Dict, args: argparse.Namespace) -> str:
    lines = []
    lines.append("# Memory/FSM Robustness Report")
    lines.append("")
    lines.append("## Config")
    lines.append(f"- seeds: {args.seeds}")
    lines.append(f"- noise_grid: {args.noise_grid}")
    lines.append(f"- hold_steps: {args.hold_steps}")
    lines.append(f"- automaton_transitions: {args.automaton_transitions}")
    lines.append(f"- turing_noises: {args.turing_noises}")
    lines.append(f"- turing_extra_random: {args.turing_extra_random}")
    lines.append("")
    lines.append("## Failure Envelope")
    for k, v in rep["failure_envelope"].items():
        lines.append(
            f"- {k}: min={v['min_pass_rate_over_sweep']:.3f}, "
            f"mean={v['mean_pass_rate_over_sweep']:.3f}, max={v['max_pass_rate_over_sweep']:.3f}"
        )
    lines.append("")
    lines.append("## By Noise")
    for metric in ("latch", "register", "automaton", "turing_demo"):
        if metric not in rep:
            continue
        lines.append(f"### {metric}")
        lines.append("| noise | pass_rate | ci95 | n_pass/n_trials |")
        lines.append("|---|---:|---|---:|")
        for noise, row in rep[metric].items():
            ci = row["pass_rate_ci95"]
            lines.append(
                f"| {noise} | {row['pass_rate']:.3f} | [{ci[0]:.3f}, {ci[1]:.3f}] | "
                f"{row['n_pass']}/{row['n_trials']} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Robustness/statistics sweep for phase memory + FSM + Turing demo")
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--noise-grid", type=str, default="0.0,0.05,0.1,0.3,0.5")
    ap.add_argument("--hold-steps", type=int, default=5000)
    ap.add_argument("--automaton-transitions", type=int, default=50)
    ap.add_argument("--turing-noises", type=str, default="0.0,0.02,0.05,0.1")
    ap.add_argument("--turing-extra-random", type=int, default=8)
    ap.add_argument("--out-json", type=str, default="reports/memory_fsm_robustness.json")
    ap.add_argument("--out-md", type=str, default="reports/memory_fsm_robustness.md")
    args = ap.parse_args()

    noise_grid = parse_float_list(args.noise_grid)
    turing_noises = parse_float_list(args.turing_noises)
    rep = run_suite(
        seeds=args.seeds,
        noise_grid=noise_grid,
        hold_steps=args.hold_steps,
        automaton_transitions=args.automaton_transitions,
        turing_noises=turing_noises,
        turing_extra_random=args.turing_extra_random,
    )
    rep["config"] = {
        "seeds": args.seeds,
        "noise_grid": noise_grid,
        "hold_steps": args.hold_steps,
        "automaton_transitions": args.automaton_transitions,
        "turing_noises": turing_noises,
        "turing_extra_random": args.turing_extra_random,
    }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(markdown_report(rep, args))
    print(f"saved {args.out_json}")
    print(f"saved {args.out_md}")


if __name__ == "__main__":
    main()
