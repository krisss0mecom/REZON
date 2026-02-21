#!/usr/bin/env python3
"""
CNOT Phase Gate — pure emergent CNOT without RLS readout.

Architecture: 3-oscillator circuit
  φ_c  : control oscillator — strongly driven to control_bit * π
  φ_t  : target oscillator  — strongly driven to target_bit * π
  φ_out: output oscillator  — free, CNOT-coupled to φ_t via φ_c modulation

CNOT coupling equation (the key):
  dφ_out/dt = ω_out + anchor + K_cnot * cos(φ_c) * sin(φ_t - φ_out)

Physics:
  cos(φ_c) ≈ +1 when φ_c ≈ 0 (control=0) → synchronizing  → φ_out → φ_t        (preserve)
  cos(φ_c) ≈ −1 when φ_c ≈ π (control=1) → anti-synchronizing → φ_out → φ_t+π  (flip)

Mathematical proof of CNOT correctness:
  Fixed points of cos(φ_c)·sin(φ_t − φ_out) = 0: φ_out ∈ {φ_t, φ_t+π}
  Stability: d/dφ_out[cos(φ_c)·sin(φ_t−φ_out)] = −cos(φ_c)·cos(φ_t−φ_out)
    control=0 (cos>0): φ_out=φ_t  STABLE,  φ_out=φ_t+π UNSTABLE → preserve target
    control=1 (cos<0): φ_out=φ_t  UNSTABLE, φ_out=φ_t+π STABLE  → flip target

Truth table verification:
  (C=0,T=0): φ_t=0,  control=0 → φ_out→0   → cos>0 → bit=0  expected=0 ✓
  (C=0,T=1): φ_t=π,  control=0 → φ_out→π   → cos<0 → bit=1  expected=1 ✓
  (C=1,T=0): φ_t=0,  control=1 → φ_out→π   → cos<0 → bit=1  expected=1 ✓
  (C=1,T=1): φ_t=π,  control=1 → φ_out→2π≡0 → cos>0 → bit=0 expected=0 ✓

Readout: mean(cos(φ_out)) over collect steps. Threshold=0. No RLS, no weights.
Immutable: f_anchor = 200.0 Hz (REZON constraint)
"""
import argparse
import json
import os

import numpy as np

ANCHOR_HZ = 200.0


def ensure_anchor_immutable(v: float) -> None:
    if abs(float(v) - ANCHOR_HZ) > 1e-6:
        raise ValueError(f"f_anchor must be exactly {ANCHOR_HZ} Hz — immutable")


class CNOTPhaseGate:
    """
    Pure phase-based CNOT gate. No RLS, no trained weights.

    3 oscillators:
      0 = control  (strongly injected → control_bit * π)
      1 = target   (strongly injected → target_bit * π)
      2 = output   (free, coupled via CNOT equation)

    Readout: mean(cos(φ_out)) > 0 → bit=0, else → bit=1
    """

    def __init__(
        self,
        dt: float = 0.001,
        K_inj: float = 8.0,
        K_cnot: float = 5.0,
        a_anchor: float = 0.08,
        leak: float = 0.0,
        noise_amp: float = 0.0,
        seed: int = 42,
    ):
        ensure_anchor_immutable(ANCHOR_HZ)
        self.dt = float(dt)
        self.K_inj = float(K_inj)
        self.K_cnot = float(K_cnot)
        self.a_anchor = float(a_anchor)
        self.leak = float(leak)
        self.noise_amp = float(noise_amp)
        self._rng = np.random.default_rng(seed)

        # Small natural frequency for output oscillator (independent of seed for reproducibility)
        self.omega_out = float(self._rng.uniform(-0.05, 0.05))

        self.phi_c = 0.0
        self.phi_t = 0.0
        self.phi_out = 0.0
        self._t = 0.0

    def reset(self, control_bit: int, target_bit: int, run_seed: int = 0) -> None:
        rng = np.random.default_rng(run_seed)
        self.phi_c = float(control_bit) * np.pi
        self.phi_t = float(target_bit) * np.pi
        self.phi_out = float(rng.uniform(0.0, 2.0 * np.pi))  # random IC — gate must overcome it
        self._t = 0.0

    def step(self, control_bit: int, target_bit: int) -> None:
        c_target = float(control_bit) * np.pi
        t_target = float(target_bit) * np.pi
        ap = 2.0 * np.pi * ANCHOR_HZ * self._t  # anchor phase

        # 200Hz anchor (immutable) applied per-oscillator
        anchor_c   = self.a_anchor * np.sin(ap - self.phi_c)
        anchor_t   = self.a_anchor * np.sin(ap - self.phi_t)
        anchor_out = self.a_anchor * np.sin(ap - self.phi_out)

        # Control: strong injection keeps φ_c ≈ control_bit * π
        dphi_c = self.K_inj * np.sin(c_target - self.phi_c) + anchor_c

        # Target: strong injection keeps φ_t ≈ target_bit * π
        dphi_t = self.K_inj * np.sin(t_target - self.phi_t) + anchor_t

        # Output: THE CNOT EQUATION
        # K_cnot * cos(φ_c) * sin(φ_t − φ_out)
        #   control=0 → cos(φ_c)≈+1 → sync   → φ_out → φ_t       ← preserve
        #   control=1 → cos(φ_c)≈−1 → anti-sync → φ_out → φ_t+π  ← flip
        dphi_out = (
            self.omega_out
            + self.K_cnot * np.cos(self.phi_c) * np.sin(self.phi_t - self.phi_out)
            + anchor_out
        )
        if self.leak > 0.0:
            dphi_out -= self.leak * np.sin(self.phi_out)  # phase-preserving leak
        if self.noise_amp > 0.0:
            dphi_out += self._rng.normal(0.0, self.noise_amp)

        self.phi_c   = (self.phi_c   + self.dt * dphi_c)   % (2.0 * np.pi)
        self.phi_t   = (self.phi_t   + self.dt * dphi_t)   % (2.0 * np.pi)
        self.phi_out = (self.phi_out + self.dt * dphi_out) % (2.0 * np.pi)
        self._t += self.dt

    def run_case(
        self,
        control_bit: int,
        target_bit: int,
        warmup: int = 2000,
        collect: int = 400,
        run_seed: int = 0,
    ) -> dict:
        """Run one CNOT input, return result dict."""
        self.reset(control_bit, target_bit, run_seed)

        for _ in range(warmup):
            self.step(control_bit, target_bit)

        cos_acc = 0.0
        for _ in range(collect):
            self.step(control_bit, target_bit)
            cos_acc += np.cos(self.phi_out)
        mean_cos = cos_acc / float(collect)

        pred_bit = 0 if mean_cos > 0.0 else 1
        expected = int(control_bit) ^ int(target_bit)

        return {
            "control": int(control_bit),
            "target": int(target_bit),
            "phi_c_deg": round(float(np.degrees(self.phi_c)), 2),
            "phi_t_deg": round(float(np.degrees(self.phi_t)), 2),
            "phi_out_deg": round(float(np.degrees(self.phi_out)), 2),
            "cos_c": round(float(np.cos(self.phi_c)), 4),
            "mean_cos_out": round(float(mean_cos), 4),
            "pred_bit": int(pred_bit),
            "expected": int(expected),
            "ok": int(pred_bit == expected),
        }


def run_truth_table(
    gate: "CNOTPhaseGate",
    warmup: int = 800,
    collect: int = 400,
    run_seed: int = 0,
) -> tuple:
    inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    rows = [
        gate.run_case(c, t, warmup=warmup, collect=collect, run_seed=run_seed * 1000 + i * 17)
        for i, (c, t) in enumerate(inputs)
    ]
    score = int(sum(r["ok"] for r in rows))
    return score, rows


def evaluate_multi_seed(
    cfg: dict,
    warmup: int,
    collect: int,
    n_seeds: int,
    gate_seed: int = 42,
) -> dict:
    scores = []
    for s in range(n_seeds):
        gate = CNOTPhaseGate(**cfg, seed=gate_seed)
        sc, _ = run_truth_table(gate, warmup=warmup, collect=collect, run_seed=s)
        scores.append(sc)
    arr = np.array(scores)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "pass4_rate": float((arr == 4).mean()),
        "distribution": {str(k): int((arr == k).sum()) for k in range(5)},
        "n_seeds": n_seeds,
    }


def print_truth_table(score: int, rows: list, prefix: str = "") -> None:
    print(f"\n{prefix}=== CNOT Phase Gate (pure, no RLS) ===")
    print(f"{prefix}Equation: dφ_out = ω + anchor + K_cnot·cos(φ_c)·sin(φ_t − φ_out)")
    print(f"{prefix}Readout:  bit = 0 if mean(cos(φ_out)) > 0 else 1  [no weights]\n")
    for r in rows:
        m = "✓" if r["ok"] else "✗"
        print(
            f"{prefix}  [{m}] C={r['control']} T={r['target']}"
            f"  φ_c={r['phi_c_deg']:>6.1f}°  cos_c={r['cos_c']:+.4f}"
            f"  mean_cos_out={r['mean_cos_out']:+.4f}  bit={r['pred_bit']}  expected={r['expected']}"
        )
    print(f"{prefix}\nScore: {score}/4")


def main() -> None:
    p = argparse.ArgumentParser(description="CNOT Phase Gate — pure, no RLS")
    p.add_argument("--dt", type=float, default=0.001)
    p.add_argument("--K-inj", type=float, default=8.0)
    p.add_argument("--K-cnot", type=float, default=5.0)
    p.add_argument("--a-anchor", type=float, default=0.08)
    p.add_argument("--leak", type=float, default=0.0)
    p.add_argument("--noise", type=float, default=0.0,
                   help="Noise on output oscillator (0=clean)")
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--collect", type=int, default=400)
    p.add_argument("--eval-seeds", type=int, default=100)
    p.add_argument("--gate-seed", type=int, default=42)
    p.add_argument("--noise-sweep", action="store_true",
                   help="Run noise robustness sweep")
    p.add_argument("--out-json", type=str,
                   default="reports/cnot_phase_gate_report.json")
    args = p.parse_args()

    cfg = dict(
        dt=args.dt, K_inj=args.K_inj, K_cnot=args.K_cnot,
        a_anchor=args.a_anchor, leak=args.leak, noise_amp=args.noise,
    )

    # ── single seed truth table ──────────────────────────────────────────────
    gate = CNOTPhaseGate(**cfg, seed=args.gate_seed)
    score, rows = run_truth_table(gate, args.warmup, args.collect, run_seed=args.gate_seed)
    print_truth_table(score, rows)

    # ── multi-seed evaluation ─────────────────────────────────────────────────
    print(f"\nMulti-seed ({args.eval_seeds} seeds, noise={args.noise:.3f})…")
    multi = evaluate_multi_seed(cfg, args.warmup, args.collect, args.eval_seeds, args.gate_seed)
    print(
        f"  mean={multi['mean']:.3f}/4  pass4={multi['pass4_rate']:.3f}"
        f"  dist={multi['distribution']}"
    )

    report = {
        "method": "cnot_phase_gate",
        "architecture": "3-oscillator pure CNOT, no RLS",
        "key_equation": "dphi_out = omega + anchor + K_cnot * cos(phi_c) * sin(phi_t - phi_out)",
        "physics_insight": (
            "cos(phi_c) acts as CNOT gate: +1 when control=0 (sync=preserve),"
            " -1 when control=1 (anti-sync=flip). XOR emerges from attractor symmetry."
        ),
        "params": {**cfg, "warmup": args.warmup, "collect": args.collect, "gate_seed": args.gate_seed},
        "single_seed_score": score,
        "single_seed_rows": rows,
        "multi_seed_clean": multi,
        "anchor_hz": ANCHOR_HZ,
        "comparison": {
            "old_pure_mode_pass4": 0.0,
            "rls_mode_pass4": 1.0,
            "this_gate_pass4": multi["pass4_rate"],
        },
    }

    # ── optional noise sweep ──────────────────────────────────────────────────
    if args.noise_sweep:
        print("\nNoise robustness sweep…")
        noise_levels = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
        noise_results = {}
        for nl in noise_levels:
            cfg_n = {**cfg, "noise_amp": nl}
            res = evaluate_multi_seed(cfg_n, args.warmup, args.collect, args.eval_seeds, args.gate_seed)
            noise_results[str(nl)] = res
            print(f"  noise={nl:.2f}  pass4={res['pass4_rate']:.3f}  mean={res['mean']:.3f}/4")
        report["noise_sweep"] = noise_results

    # ── save ──────────────────────────────────────────────────────────────────
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved → {args.out_json}")


if __name__ == "__main__":
    main()
