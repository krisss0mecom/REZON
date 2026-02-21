#!/usr/bin/env python3
import argparse
import json
from collections import Counter

import numpy as np


ANCHOR_HZ = 200.0


def ensure_anchor_immutable(v: float) -> None:
    if abs(float(v) - ANCHOR_HZ) > 1e-6:
        raise ValueError(f"f_anchor must be exactly {ANCHOR_HZ} Hz")


class PhaseCNOTAudit:
    """
    Two-node phase system for CNOT-like experiments.

    modes:
    - pure: no direct XOR target forcing, only conditional anti-sync dynamics
    - xor_driven: explicit target attractor to expected XOR bit
    """

    def __init__(self, mode: str = "pure", **params):
        if mode not in ("pure", "xor_driven"):
            raise ValueError("mode must be 'pure' or 'xor_driven'")
        self.mode = mode
        self.params = params

        self.N = 2
        self.dt = float(params["dt"])
        self.coupling = float(params["coupling"])
        self.a_anchor = float(params["a_anchor"])
        self.noise_amp = float(params["noise_amp"])
        self.phase_leak = float(params["phase_leak"])
        self.bias_scale = float(params["bias_scale"])
        self.control_force = float(params.get("control_force", 3.0))
        self.anti_gain = float(params.get("anti_gain", 2.2))
        self.control_threshold = float(params.get("control_threshold", 0.6))
        self.control_window_div = float(params.get("control_window_div", 1.5))
        self.inject_window_s = float(params.get("inject_window_s", 0.05))

        ensure_anchor_immutable(float(params.get("f_anchor", ANCHOR_HZ)))
        self.f_anchor = ANCHOR_HZ

        np.random.seed(int(params["seed"]))
        self.phi = np.random.uniform(0.0, 2.0 * np.pi, self.N)
        self.omega = np.random.uniform(-0.3, 0.3, self.N)
        self._t = 0.0

    @staticmethod
    def bit_to_phase(bit: int) -> float:
        return 0.0 if int(bit) == 0 else np.pi

    def reset(self):
        np.random.seed(int(self.params["seed"]))
        self.phi = np.random.uniform(0.0, 2.0 * np.pi, self.N)
        self._t = 0.0

    def step(self, control_bit=None, target_bit=None, is_warmup=False):
        dphi = self.omega.copy()

        # Base Kuramoto coupling
        diff01 = self.phi[1] - self.phi[0]
        diff10 = self.phi[0] - self.phi[1]
        dphi[0] += self.coupling * np.sin(diff01)
        dphi[1] += self.coupling * np.sin(diff10)

        if (not is_warmup) and (control_bit is not None) and (target_bit is not None):
            control_phase = self.bit_to_phase(control_bit)
            target_phase = self.bit_to_phase(target_bit)

            # Input injection window (uses control_bit and target_bit explicitly)
            if self._t < self.inject_window_s:
                self.phi[0] = control_phase
                self.phi[1] = target_phase

            # Keep control near commanded phase
            dphi[0] += self.control_force * np.sin(control_phase - self.phi[0])

            # Conditional anti-sync influence based on control state
            dist_to_pi = min(abs(self.phi[0] - np.pi), 2 * np.pi - abs(self.phi[0] - np.pi))
            control_factor = max(0.0, 1.0 - dist_to_pi / (np.pi / max(self.control_window_div, 1e-6)))
            if control_factor > self.control_threshold:
                dphi[1] += -self.coupling * self.anti_gain * np.sin(self.phi[0] - self.phi[1])
                dphi[1] += control_factor * self.bias_scale * np.cos(self.phi[1])

            if self.mode == "pure":
                # Pure mode: only mild hold to input target, no direct XOR forcing.
                dphi[1] += 0.35 * self.bias_scale * np.sin(target_phase - self.phi[1])
            else:
                # XOR-driven sanity mode: explicit attractor to expected CNOT target.
                expected = int(control_bit) ^ int(target_bit)
                expected_phase = self.bit_to_phase(expected)
                dphi[1] += self.bias_scale * np.sin(expected_phase - self.phi[1])

        # Anchor + leak + noise
        anchor = self.a_anchor * np.sin(2.0 * np.pi * self.f_anchor * self._t - self.phi)
        dphi += anchor
        dphi -= self.phase_leak * self.phi
        if self.noise_amp > 0.0 and not is_warmup:
            dphi += np.random.normal(0.0, self.noise_amp, self.N)

        self.phi = (self.phi + self.dt * dphi) % (2.0 * np.pi)
        self._t += self.dt

    def readout(self):
        cos0 = np.cos(self.phi[0])
        cos1 = np.cos(self.phi[1])
        bit0 = 0 if cos0 > 0 else 1
        bit1 = 0 if cos1 > 0 else 1
        return (float(cos0), float(cos1)), (int(bit0), int(bit1))

    def run_case(self, control, target, warmup=5000, steps=2000):
        self.reset()
        for _ in range(int(warmup)):
            self.step(is_warmup=True)
        for _ in range(int(steps)):
            self.step(control, target)

        cos, bits = self.readout()
        expected_target = int(control) ^ int(target)
        ok = int(bits[1] == expected_target)
        return {
            "control": int(control),
            "target": int(target),
            "cos_control": round(cos[0], 3),
            "cos_target": round(cos[1], 3),
            "pred_target": int(bits[1]),
            "expected": int(expected_target),
            "ok": ok,
        }


def run_truth_table(seed: int, mode: str, warmup: int, steps: int, params: dict):
    p = dict(params)
    p["seed"] = int(seed)
    tester = PhaseCNOTAudit(mode=mode, **p)
    cases = [(0, 0), (0, 1), (1, 0), (1, 1)]
    rows = [tester.run_case(c, t, warmup=warmup, steps=steps) for c, t in cases]
    score = int(sum(r["ok"] for r in rows))
    return score, rows


def summarize_scores(scores):
    c = Counter(scores)
    return {k: c.get(k, 0) for k in range(5)}


def main():
    ap = argparse.ArgumentParser(description="Audit pure vs xor-driven CNOT-like phase dynamics.")
    ap.add_argument("--warmup", type=int, default=5000)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--eval-seeds", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--coupling", type=float, default=2.0)
    ap.add_argument("--a-anchor", type=float, default=0.4)
    ap.add_argument("--dt", type=float, default=0.001)
    ap.add_argument("--noise-amp", type=float, default=0.05)
    ap.add_argument("--phase-leak", type=float, default=0.01)
    ap.add_argument("--bias-scale", type=float, default=5.0)
    ap.add_argument("--control-force", type=float, default=3.0)
    ap.add_argument("--anti-gain", type=float, default=2.2)
    ap.add_argument("--control-threshold", type=float, default=0.6)
    ap.add_argument("--control-window-div", type=float, default=1.5)
    ap.add_argument("--inject-window-s", type=float, default=0.05)

    ap.add_argument("--out-json", type=str, default="results/cnot_variant_audit.json")
    args = ap.parse_args()

    base_params = {
        "coupling": args.coupling,
        "a_anchor": args.a_anchor,
        "dt": args.dt,
        "noise_amp": args.noise_amp,
        "phase_leak": args.phase_leak,
        "bias_scale": args.bias_scale,
        "control_force": args.control_force,
        "anti_gain": args.anti_gain,
        "control_threshold": args.control_threshold,
        "control_window_div": args.control_window_div,
        "inject_window_s": args.inject_window_s,
        "f_anchor": ANCHOR_HZ,
        "seed": args.seed,
    }

    report = {"config": {**base_params, "warmup": args.warmup, "steps": args.steps, "eval_seeds": args.eval_seeds}}

    for mode in ("pure", "xor_driven"):
        score, rows = run_truth_table(args.seed, mode, args.warmup, args.steps, base_params)
        print(f"\nmode={mode} seed={args.seed} score={score}/4")
        for r in rows:
            print(
                f"C={r['control']} T={r['target']} -> cos=({r['cos_control']:+.3f}, {r['cos_target']:+.3f}) "
                f"pred={r['pred_target']} expected={r['expected']} {'OK' if r['ok'] else 'FAIL'}"
            )

        scores = []
        for s in range(args.eval_seeds):
            sc, _ = run_truth_table(s, mode, args.warmup, args.steps, base_params)
            scores.append(sc)
        dist = summarize_scores(scores)
        mean_score = float(np.mean(scores))
        pass4 = float(np.mean(np.array(scores) == 4))
        print(f"mode={mode} multi_seed_mean={mean_score:.3f}/4 pass4={pass4:.3f} dist={dist}")

        report[mode] = {
            "single_seed": int(args.seed),
            "single_score": int(score),
            "single_rows": rows,
            "multi_seed_mean": mean_score,
            "pass_rate_4of4": pass4,
            "distribution": dist,
        }

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nsaved: {args.out_json}")


if __name__ == "__main__":
    main()
