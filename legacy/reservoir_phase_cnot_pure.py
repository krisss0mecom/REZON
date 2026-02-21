#!/usr/bin/env python3
import argparse
import json
import numpy as np


class ReservoirPhaseCNOTPure:
    """
    Emergent/unstable pure phase CNOT-like proof-of-concept.
    No trained readout. Included for research comparison.

    Typical behavior: ~2-3 correct cases out of 4, depending on seed.
    """

    ANCHOR_HZ = 200.0

    def __init__(
        self,
        coupling=6.0,
        anti_coupling=22.0,
        a_anchor=0.10,
        dt=0.005,
        noise_amp=0.003,
        control_window_div=1.8,
        direction_threshold=0.1,
        inject_window_s=0.05,
    ):
        self.n = 2
        self.dt = float(dt)
        self.coupling = float(coupling)
        self.anti_coupling = float(anti_coupling)
        self.a_anchor = float(a_anchor)
        self.noise_amp = float(noise_amp)
        self.control_window_div = float(control_window_div)
        self.direction_threshold = float(direction_threshold)
        self.inject_window_s = float(inject_window_s)
        self.phi = np.zeros(2, dtype=np.float64)
        self.omega = np.random.uniform(-0.15, 0.15, 2).astype(np.float64)
        self._t = 0.0

    def reset(self):
        self.phi.fill(0.0)
        self._t = 0.0

    def step(self, control_bit: int, target_bit: int):
        # Initial hard injection only at startup.
        if self._t < self.inject_window_s:
            self.phi[0] = 0.0 if int(control_bit) == 0 else np.pi
            self.phi[1] = 0.0 if int(target_bit) == 0 else np.pi

        dphi = self.omega.copy()

        # Weak base synchronization.
        dphi[0] += self.coupling * np.sin(self.phi[1] - self.phi[0])
        dphi[1] += self.coupling * np.sin(self.phi[0] - self.phi[1])

        # Emergent anti-sync push when control ~ pi.
        dist_to_pi = min(abs(self.phi[0] - np.pi), 2 * np.pi - abs(self.phi[0] - np.pi))
        control_factor = max(0.0, 1.0 - dist_to_pi / (np.pi / max(self.control_window_div, 1e-6)))
        direction = (
            np.sign(np.sin(self.phi[0]))
            if abs(np.sin(self.phi[0])) > self.direction_threshold
            else 0.0
        )
        flip_term = control_factor * self.anti_coupling * direction * np.cos(self.phi[1])
        dphi[1] += flip_term

        # Immutable 200 Hz anchor.
        anchor = self.a_anchor * np.sin(2.0 * np.pi * self.ANCHOR_HZ * self._t - self.phi)
        dphi += anchor

        if self.noise_amp > 0.0:
            dphi += np.random.normal(0.0, self.noise_amp, 2)

        self.phi = (self.phi + dphi * self.dt) % (2.0 * np.pi)
        self._t += self.dt

    def readout(self):
        cos_vals = np.cos(self.phi)
        bits = (cos_vals <= 0.0).astype(int)
        return cos_vals, bits


def run_truth_table(seed: int, steps: int, **rc_kwargs):
    np.random.seed(seed)
    inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    rows = []
    score = 0
    for c, t in inputs:
        rc = ReservoirPhaseCNOTPure(**rc_kwargs)
        rc.reset()
        for _ in range(int(steps)):
            rc.step(c, t)
        cos_vals, bits = rc.readout()
        expected = c ^ t
        ok = int(bits[1] == expected)
        score += ok
        rows.append(
            {
                "control": c,
                "target": t,
                "cos_control": float(cos_vals[0]),
                "cos_target": float(cos_vals[1]),
                "pred_target": int(bits[1]),
                "expected": int(expected),
                "ok": ok,
            }
        )
    return score, rows


def main():
    p = argparse.ArgumentParser(description="Pure/emergent CNOT-like phase PoC (unstable).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--eval-seeds", type=int, default=20)
    p.add_argument("--coupling", type=float, default=6.0)
    p.add_argument("--anti-coupling", type=float, default=22.0)
    p.add_argument("--a-anchor", type=float, default=0.10)
    p.add_argument("--dt", type=float, default=0.005)
    p.add_argument("--noise-amp", type=float, default=0.003)
    p.add_argument("--control-window-div", type=float, default=1.8)
    p.add_argument("--direction-threshold", type=float, default=0.1)
    p.add_argument("--inject-window-s", type=float, default=0.05)
    p.add_argument("--out-json", type=str, default="")
    args = p.parse_args()

    rc_kwargs = {
        "coupling": float(args.coupling),
        "anti_coupling": float(args.anti_coupling),
        "a_anchor": float(args.a_anchor),
        "dt": float(args.dt),
        "noise_amp": float(args.noise_amp),
        "control_window_div": float(args.control_window_div),
        "direction_threshold": float(args.direction_threshold),
        "inject_window_s": float(args.inject_window_s),
    }
    score, rows = run_truth_table(args.seed, args.steps, **rc_kwargs)
    print(f"single_seed={args.seed} score={score}/4")
    for r in rows:
        print(
            f"In: C={r['control']}, T={r['target']} -> cos=({r['cos_control']:+.2f}, {r['cos_target']:+.2f}) "
            f"bit={r['pred_target']} expected={r['expected']} {'OK' if r['ok'] else 'FAIL'}"
        )

    scores = []
    for s in range(args.eval_seeds):
        sc, _ = run_truth_table(s, args.steps, **rc_kwargs)
        scores.append(sc)
    mean_score = float(np.mean(scores))
    dist = {k: int(scores.count(k)) for k in range(5)}
    print(f"multi_seed_mean={mean_score:.3f}/4  distribution={dist}")

    if args.out_json:
        payload = {
            "single_seed": int(args.seed),
            "single_score": int(score),
            "rows": rows,
            "eval_seeds": int(args.eval_seeds),
            "multi_seed_mean": mean_score,
            "distribution": dist,
            "steps": int(args.steps),
            "params": rc_kwargs,
            "note": "Proof-of-concept only; pure emergent variant is not stable across seeds.",
        }
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"saved: {args.out_json}")


if __name__ == "__main__":
    main()
