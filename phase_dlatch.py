#!/usr/bin/env python3
"""
PhaseDLatch — D-latch (1-bit memory) from pure phase dynamics. No RLS. No ML.

════════════════════════════════════════════════════════════════
EQUATION:

  dφ_Q/dt = K_e · (1−cos φ_E)/2 · sin(φ_D − φ_Q)
           − K_hold · sin(2·φ_Q)
           + ω + anchor(200Hz) + noise

════════════════════════════════════════════════════════════════
PROOF THAT THIS IS A D-LATCH:

Enable=1 → φ_E = π → (1−cos π)/2 = 1.0 → full coupling to D  (WRITE mode)
Enable=0 → φ_E = 0 → (1−cos 0)/2  = 0.0 → coupling = 0        (HOLD mode)

In HOLD mode, only the bistable term acts:
  dφ_Q/dt = −K_hold · sin(2·φ_Q)

  Fixed points of −K_hold·sin(2φ): φ = 0, π/2, π, 3π/2
  Stability: f'(φ) = −2K_hold·cos(2φ)
    φ = 0   : f' = −2K_hold < 0  → STABLE   ← bit=0 attractor
    φ = π/2 : f' = +2K_hold > 0  → UNSTABLE
    φ = π   : f' = −2K_hold < 0  → STABLE   ← bit=1 attractor
    φ = 3π/2: f' = +2K_hold > 0  → UNSTABLE

  → Once written, φ_Q stays at 0 (bit=0) or π (bit=1) indefinitely. □

Readout: cos(φ_Q) > 0 → bit=0, cos(φ_Q) < 0 → bit=1.

Anchor: 200.0 Hz (immutable hardware constraint).
════════════════════════════════════════════════════════════════
"""
import json
import os
import time

import numpy as np
from phase_gate_universal import ANCHOR_HZ, ensure_anchor_immutable

ANCHOR_AMP = 0.08


def _anchor(t: float, phi: float) -> float:
    return ANCHOR_AMP * np.sin(2.0 * np.pi * ANCHOR_HZ * t - phi)


class PhaseDLatch:
    """
    D-latch: 1-bit phase memory.

    Write: enable=1 for warmup steps → φ_Q converges to φ_D.
    Hold:  enable=0 → φ_Q held at bistable attractor {0, π}.
    Read:  cos(φ_Q) > 0 → 0,  cos(φ_Q) < 0 → 1.
    """

    def __init__(
        self,
        dt: float = 0.001,
        K_e: float = 16.0,   # stronger write coupling for robust 0<->1 switching with anchor
        K_hold: float = 4.0,
        noise_amp: float = 0.0,
        omega: float = 0.0,
        seed: int = 0,
    ):
        ensure_anchor_immutable(ANCHOR_HZ)
        self.dt = dt
        self.K_e = K_e
        self.K_hold = K_hold
        self.noise_amp = noise_amp
        self.omega = omega
        self.rng = np.random.default_rng(seed)
        self.phi_Q = 0.0
        self._t = 0.0

    def _step_n(self, phi_D: float, phi_E: float, n: int) -> None:
        """Integrate n steps with given data/enable phase."""
        K_e = self.K_e
        K_hold = self.K_hold
        dt = self.dt
        enable_gain = (1.0 - np.cos(phi_E)) / 2.0

        phi_Q = self.phi_Q
        t = self._t
        rng = self.rng
        noise_amp = self.noise_amp
        omega = self.omega

        for _ in range(n):
            couple_D = K_e * enable_gain * np.sin(phi_D - phi_Q)
            hold = -K_hold * np.sin(2.0 * phi_Q)
            anchor = _anchor(t, phi_Q)
            noise = rng.normal(0.0, noise_amp) if noise_amp > 0 else 0.0
            phi_Q += (omega + couple_D + hold + anchor + noise) * dt
            t += dt

        self.phi_Q = phi_Q
        self._t = t

    def write(self, bit: int, warmup: int = 3000) -> int:
        """Enable=1 for warmup steps. φ_Q converges to bit·π."""
        # Break exact symmetry when starting exactly at a saddle (e.g. 0 -> 1 or π -> 0)
        # in deterministic/noiseless runs. This keeps dynamics physical while avoiding
        # a measure-zero deadlock due to perfectly aligned phases.
        target_phi = float(bit) * np.pi
        if abs(np.sin(target_phi - self.phi_Q)) < 1e-12 and self.read() != int(bit):
            self.phi_Q += 1e-6 if int(bit) == 1 else -1e-6
        self._step_n(target_phi, np.pi, warmup)
        return self.read()

    def hold(self, n_steps: int = 5000) -> int:
        """Enable=0 for n_steps. φ_Q should stay at current attractor."""
        self._step_n(0.0, 0.0, n_steps)
        return self.read()

    def read(self) -> int:
        return 0 if np.cos(self.phi_Q) > 0.0 else 1

    def reset(self, bit: int = 0) -> None:
        self.phi_Q = float(bit) * np.pi
        self._t = 0.0


# ─────────────────────────────────────────────────────────────
# N-bit addressable register
# ─────────────────────────────────────────────────────────────

class PhaseRegister:
    """N-bit addressable memory — array of independent D-latches."""

    def __init__(self, n_bits: int, seed: int = 0, **latch_kwargs):
        rng = np.random.default_rng(seed)
        self.n_bits = n_bits
        self.latches = [
            PhaseDLatch(seed=int(rng.integers(0, 2**31)), **latch_kwargs)
            for _ in range(n_bits)
        ]

    def write(self, addr: int, bit: int, warmup: int = 3000) -> int:
        return self.latches[addr].write(bit, warmup=warmup)

    def read(self, addr: int, hold_steps: int = 2000) -> int:
        return self.latches[addr].hold(n_steps=hold_steps)

    def write_all(self, bits, warmup: int = 3000) -> list:
        return [self.latches[i].write(bits[i], warmup=warmup)
                for i in range(self.n_bits)]

    def read_all(self, hold_steps: int = 2000) -> list:
        return [self.latches[i].hold(n_steps=hold_steps)
                for i in range(self.n_bits)]


# ─────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────

def run_dlatch_demo(
    seed: int = 42,
    noise_amp: float = 0.05,
    warmup: int = 3000,
    hold_steps: int = 10_000,
    verbose: bool = True,
) -> list:
    """Write/hold cycle: 0→1→0→1→0, verify each."""
    latch = PhaseDLatch(noise_amp=noise_amp, seed=seed)
    results = []
    for bit in [0, 1, 0, 1, 0]:
        latch.write(bit, warmup=warmup)
        held = latch.hold(n_steps=hold_steps)
        ok = (held == bit)
        results.append({"write": bit, "read_after_hold": held,
                        "hold_steps": hold_steps, "ok": ok})
        if verbose:
            print(f"  write={bit}  hold={hold_steps:,} steps  read={held}  {'✓' if ok else '✗'}")
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("  PhaseDLatch — 1-bit memory from pure phase dynamics")
    print("  dφ_Q = K_e·(1-cosφ_E)/2·sin(φ_D-φ_Q) - K_hold·sin(2φ_Q)")
    print("=" * 60)

    print("\n[1] Write/hold cycle (noise=0.05, hold=10,000 steps each):")
    r1 = run_dlatch_demo(noise_amp=0.05, hold_steps=10_000)
    ok1 = sum(r["ok"] for r in r1)
    print(f"\n    Result: {ok1}/{len(r1)}\n")

    print("[2] Heavy noise (noise=1.0, hold=10,000 steps each):")
    r2 = run_dlatch_demo(noise_amp=1.0, warmup=4000, hold_steps=10_000)
    ok2 = sum(r["ok"] for r in r2)
    print(f"\n    Result: {ok2}/{len(r2)}\n")

    print("[3] PhaseRegister — 8-bit addressable memory:")
    reg = PhaseRegister(8, seed=42, noise_amp=0.05)
    pattern = [1, 0, 1, 1, 0, 1, 0, 0]
    reg.write_all(pattern)
    read_back = reg.read_all(hold_steps=5000)
    ok3 = (read_back == pattern)
    print(f"    written:   {pattern}")
    print(f"    read back: {read_back}")
    print(f"    match: {'✓' if ok3 else '✗'}\n")

    os.makedirs("reports", exist_ok=True)
    report = {
        "ts": int(time.time()),
        "noise005_5cycles": r1,
        "noise10_5cycles": r2,
        "register_8bit": {"written": pattern, "read_back": read_back, "match": ok3},
    }
    with open("reports/phase_dlatch_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("  Saved → reports/phase_dlatch_report.json")
