#!/usr/bin/env python3
"""
PhaseAutomaton — 3-state Finite State Machine from pure phase dynamics. No RLS.

════════════════════════════════════════════════════════════════
STATES: S0, S1, S2  encoded as φ ∈ {0, 2π/3, 4π/3}

POTENTIAL:
  V(φ) = −cos(3φ)/3   →   dφ/dt = −K_hold · sin(3φ)

  Fixed points of sin(3φ) = 0:  φ = 0, π/3, 2π/3, π, 4π/3, 5π/3
  Stability: f'(φ) = −3K_hold · cos(3φ)
    φ = 0    : cos(0)=+1   → f' = −3K_hold < 0 → STABLE   ← S0
    φ = π/3  : cos(π)=−1   → f' = +3K_hold > 0 → UNSTABLE (barrier)
    φ = 2π/3 : cos(2π)=+1  → f' = −3K_hold < 0 → STABLE   ← S1
    φ = π    : cos(3π)=−1  → f' = +3K_hold > 0 → UNSTABLE (barrier)
    φ = 4π/3 : cos(4π)=+1  → f' = −3K_hold < 0 → STABLE   ← S2
    φ = 5π/3 : cos(5π)=−1  → f' = +3K_hold > 0 → UNSTABLE (barrier)
  → Exactly 3 stable states matching our encoding. □

TRANSITION (clock=1):
  Clock bit=1 → φ_clk = π → gain = (1−cos π)/2 = 1.
  Full equation during push:
    dφ/dt = −K_hold · sin(3φ) + K_step · gain + anchor + noise

  K_step is set so that K_step · step_duration · dt ≈ 2π/3 (gross advance).
  The hold term counters some of this, but K_step >> K_hold guarantees forward motion:
    dφ/dt ≥ K_step − K_hold · max|sin(3φ)| = K_step − K_hold > 0  ✓

  After step_duration steps: φ is past barrier (π/3) but before next barrier (π).
  Then relax_steps with push=0 pulls φ to next attractor. □

CLOCK=0:
  gain = 0 → only −K_hold·sin(3φ) acts → holds current state. □

FSM demonstrated: mod-3 counter S0→S1→S2→S0→...

Anchor: 200.0 Hz (immutable hardware constraint).
════════════════════════════════════════════════════════════════
"""
import json
import os
import time

import numpy as np
from phase_gate_universal import ANCHOR_HZ, ensure_anchor_immutable

ANCHOR_AMP = 0.08
# State encoding: state i → φ = i * 2π/3
STATE_PHASES = [0.0, 2 * np.pi / 3, 4 * np.pi / 3]


def _anchor(t: float, phi: float) -> float:
    return ANCHOR_AMP * np.sin(2.0 * np.pi * ANCHOR_HZ * t - phi)


class PhaseAutomaton:
    """
    3-state FSM as phase oscillator (mod-3 counter).

    Potential: −K_hold·sin(3φ) → 3 stable attractors at {0, 2π/3, 4π/3}
    Clock=1: forward push (step_duration steps) + relax → next state (mod 3)
    Clock=0: hold at current attractor.
    """

    def __init__(
        self,
        dt: float = 0.001,
        K_hold: float = 4.0,
        K_step: float = 25.0,
        step_duration: int = 90,
        relax_steps: int = 2000,
        noise_amp: float = 0.0,
        seed: int = 0,
    ):
        ensure_anchor_immutable(ANCHOR_HZ)
        self.dt = dt
        self.K_hold = K_hold
        self.K_step = K_step
        self.step_duration = step_duration
        self.relax_steps = relax_steps
        self.noise_amp = noise_amp
        self.rng = np.random.default_rng(seed)
        self.phi = 0.0
        self._t = 0.0
        self._history: list = []

    def _integrate(self, n: int, push: float) -> None:
        K_hold = self.K_hold
        dt = self.dt
        phi = self.phi
        t = self._t
        rng = self.rng
        noise_amp = self.noise_amp

        for _ in range(n):
            hold = -K_hold * np.sin(3.0 * phi)
            anchor = _anchor(t, phi)
            noise = rng.normal(0.0, noise_amp) if noise_amp > 0 else 0.0
            phi += (hold + push + anchor + noise) * dt
            t += dt

        self.phi = phi
        self._t = t

    def tick(self) -> int:
        """Clock=1: advance state by 1 (mod 3). Returns new state."""
        # Clock=1 → φ_clk=π → gain=(1−cos π)/2 = 1
        push = self.K_step * 1.0
        self._integrate(self.step_duration, push)
        self._integrate(self.relax_steps, 0.0)

        s = self.read_state()
        self._history.append(s)
        return s

    def idle(self, n_steps: int = 1000) -> int:
        """Clock=0: hold current state."""
        self._integrate(n_steps, 0.0)
        return self.read_state()

    def read_state(self) -> int:
        """Map φ → nearest state index in {0, 1, 2}."""
        phi_mod = self.phi % (2.0 * np.pi)
        dists = [
            min(abs(phi_mod - sp), 2 * np.pi - abs(phi_mod - sp))
            for sp in STATE_PHASES
        ]
        return int(np.argmin(dists))

    def reset(self, state: int = 0) -> None:
        self.phi = STATE_PHASES[state]
        self._t = 0.0
        self._history = []


# ─────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────

def run_automaton_demo(
    n_transitions: int = 50,
    noise_amp: float = 0.0,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Run mod-3 counter for n_transitions. Verify each step is (prev+1)%3."""
    fsm = PhaseAutomaton(noise_amp=noise_amp, seed=seed)
    expected = [i % 3 for i in range(1, n_transitions + 1)]
    actual = []

    for i in range(n_transitions):
        s = fsm.tick()
        actual.append(s)
        exp = expected[i]
        ok = (s == exp)
        if verbose:
            if i < 6 or i >= n_transitions - 3 or not ok:
                print(f"  tick {i+1:3d}: state={s}  exp={exp}  {'✓' if ok else '✗ FAIL'}")
            elif i == 6:
                print(f"  ... ({n_transitions - 9} more ticks) ...")

    n_correct = sum(a == e for a, e in zip(actual, expected))
    return {
        "n_transitions": n_transitions,
        "n_correct": n_correct,
        "pass_rate": n_correct / n_transitions,
        "actual": actual,
        "expected": expected,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("  PhaseAutomaton — 3-state FSM from pure phase dynamics")
    print("  Potential: dφ/dt = -K·sin(3φ)  [3 stable attractors]")
    print("  Transition: brief push + relax → advance to next state")
    print("=" * 60)

    print("\n[1] Mod-3 counter, 50 transitions, no noise:")
    r1 = run_automaton_demo(50, noise_amp=0.0)
    print(f"\n    Result: {r1['n_correct']}/{r1['n_transitions']}  "
          f"pass_rate={r1['pass_rate']:.3f}\n")

    print("[2] Mod-3 counter, 50 transitions, noise=0.3:")
    r2 = run_automaton_demo(50, noise_amp=0.3)
    print(f"\n    Result: {r2['n_correct']}/{r2['n_transitions']}  "
          f"pass_rate={r2['pass_rate']:.3f}\n")

    print("[3] Mod-3 counter, 100 transitions, noise=0.1:")
    r3 = run_automaton_demo(100, noise_amp=0.1)
    print(f"\n    Result: {r3['n_correct']}/{r3['n_transitions']}  "
          f"pass_rate={r3['pass_rate']:.3f}\n")

    os.makedirs("reports", exist_ok=True)
    report = {
        "ts": int(time.time()),
        "clean_50": r1,
        "noisy03_50": r2,
        "long_noisy01_100": r3,
    }
    with open("reports/phase_automaton_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("  Saved → reports/phase_automaton_report.json")
