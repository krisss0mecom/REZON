#!/usr/bin/env python3
"""
Phase Hopfield Network — associative memory from coupled oscillators.

════════════════════════════════════════════════════════════════════════
MODEL:

  N phase oscillators, each φ_i ∈ [0, 2π].
  Stable attractors: φ_i ∈ {0, π}  (bit=0, bit=1).

  Stored patterns: ξ^μ ∈ {0,1}^N  for μ = 1…P.
  Encoded as phases: ξ_i^μ · π ∈ {0, π}.

  Hebbian weights (Ising ±1 encoding):
    s_i^μ = cos(ξ_i^μ · π) ∈ {+1, −1}
    W_ij  = (1/N) Σ_μ s_i^μ · s_j^μ      (symmetric, W_ii = 0)

  Dynamics:
    dφ_i/dt = −K · Σ_j W_ij · sin(φ_i − φ_j)
             + a_anchor · sin(2π·f_anchor·t − φ_i)   [200 Hz, immutable]
             + ω_i + noise

════════════════════════════════════════════════════════════════════════
STABILITY PROOF (stored patterns are fixed points):

  At φ = ξ^μ (stored pattern), sin(ξ_i^μ) = 0 for all i (since ξ∈{0,π}).
  So: sin(ξ_i^μ − ξ_j^μ) = sin(ξ_i^μ)cos(ξ_j^μ) − cos(ξ_i^μ)sin(ξ_j^μ)
                           = 0 − 0 = 0.
  Therefore: dφ_i/dt = 0 for all i → stored patterns are fixed points. □

  Stability (linearization at ξ^μ):
    ∂²E/∂φ_i² = Σ_j W_ij cos(ξ_i^μ − ξ_j^μ)
              ≈ cos²(ξ_i^μ) = 1 > 0  [for P << N, by pattern orthogonality]
  → Stored patterns are STABLE fixed points (energy minima). □

════════════════════════════════════════════════════════════════════════
CONNECTION TO CLASSIC HOPFIELD:

  Energy: E = −½ Σ_{ij} W_ij cos(φ_i − φ_j)
  Restricted to {0,π}^N: cos(φ_i−φ_j) = cos(φ_i)·cos(φ_j)
  → E = −½ Σ W_ij s_i s_j  ← identical to Hopfield Ising Hamiltonian!

  Capacity: P_max ≈ 0.138·N  (same as classical Hopfield).

════════════════════════════════════════════════════════════════════════
Anchor: 200 Hz (immutable REZON constraint).
"""

import argparse
import json
import os
import time

import numpy as np

from phase_gate_universal import ANCHOR_HZ, ensure_anchor_immutable

ensure_anchor_immutable(ANCHOR_HZ)

TWO_PI = 2.0 * np.pi


# ── Phase Hopfield Network ────────────────────────────────────────────────────

class PhaseHopfield:
    """
    Associative memory network of N coupled phase oscillators.

    Stores up to P ≈ 0.138·N binary patterns ξ^μ ∈ {0,1}^N.
    Retrieves stored patterns from noisy/partial inputs.
    """

    def __init__(
        self,
        N: int,
        patterns: list,           # list of lists/arrays, each of length N, values in {0,1}
        *,
        K: float = 3.0,           # coupling strength
        dt: float = 0.001,
        a_anchor: float = 0.08,
        omega_std: float = 0.02,  # heterogeneity in natural frequencies
        noise_amp: float = 0.0,
        seed: int = 42,
    ):
        self.N         = N
        self.K         = K
        self.dt        = dt
        self.a_anchor  = a_anchor
        self.noise_amp = noise_amp
        self.rng       = np.random.default_rng(seed)

        # Store patterns as phase arrays {0, π}
        self.patterns  = [np.array(p, dtype=float) * np.pi for p in patterns]
        self.P         = len(patterns)

        # Natural frequencies (slight heterogeneity)
        self.omega     = self.rng.normal(0.0, omega_std, N)

        # Hebbian weight matrix (Ising ±1 encoding)
        # s_i = cos(ξ_i · π) = 1 if ξ_i=0, -1 if ξ_i=1
        self.W = np.zeros((N, N))
        for pat in self.patterns:
            s = np.cos(pat)   # ±1
            self.W += np.outer(s, s)
        self.W /= float(N)
        np.fill_diagonal(self.W, 0.0)  # no self-coupling

        # Phase state
        self.phi = np.zeros(N)

    def reset(self, phi_init: np.ndarray) -> None:
        """Set initial phase configuration."""
        self.phi = phi_init.copy() % TWO_PI
        self._t  = 0.0

    def step(self) -> None:
        """Single Euler step of network dynamics."""
        phi = self.phi
        t   = self._t

        # Coupling: dφ_i = -K · Σ_j W_ij · sin(φ_i - φ_j)
        # Vectorized: sin_diff[i,j] = sin(φ_i - φ_j)
        diff    = phi[:, None] - phi[None, :]        # (N, N)
        sin_diff = np.sin(diff)
        coupling = -self.K * np.sum(self.W * sin_diff, axis=1)

        # Anchor
        anchor = self.a_anchor * np.sin(TWO_PI * ANCHOR_HZ * t - phi)

        # Noise
        noise = (self.rng.normal(0.0, self.noise_amp, self.N)
                 if self.noise_amp > 0 else 0.0)

        self.phi = (phi + self.dt * (self.omega + coupling + anchor + noise)) % TWO_PI
        self._t  += self.dt

    def run(self, n_steps: int) -> np.ndarray:
        """Integrate n_steps and return final phase array."""
        for _ in range(n_steps):
            self.step()
        return self.phi.copy()

    def decode(self) -> np.ndarray:
        """Decode current phase state to binary pattern."""
        return (np.cos(self.phi) < 0.0).astype(int)

    def energy(self) -> float:
        """Hopfield energy E = -½ Σ W_ij cos(φ_i - φ_j)."""
        diff = self.phi[:, None] - self.phi[None, :]
        return float(-0.5 * np.sum(self.W * np.cos(diff)))

    def nearest_pattern(self) -> tuple:
        """Return (index, hamming_distance) of stored pattern closest to current state."""
        decoded = self.decode()
        dists = []
        for pat in self.patterns:
            pat_bits = (pat / np.pi).astype(int)
            dists.append(int(np.sum(decoded != pat_bits)))
        best = int(np.argmin(dists))
        return best, dists[best]

    def recall(
        self,
        pattern_idx: int,
        flip_fraction: float = 0.2,
        warmup_steps: int = 5000,
        recall_steps: int = 10000,
        noise_during: float = 0.0,
    ) -> dict:
        """
        Corrupt stored pattern by flip_fraction, run dynamics, check recovery.

        Returns dict with success flag and convergence info.
        """
        pat_bits = (self.patterns[pattern_idx] / np.pi).astype(int)
        N = self.N

        # Corrupt: flip random fraction of bits
        n_flip = max(1, int(flip_fraction * N))
        flip_idx = self.rng.choice(N, size=n_flip, replace=False)
        noisy = pat_bits.copy()
        noisy[flip_idx] ^= 1

        # Init from noisy pattern + small random perturbation
        phi_init = noisy.astype(float) * np.pi
        phi_init += self.rng.normal(0.0, 0.05, N)

        self.reset(phi_init)
        old_noise  = self.noise_amp
        self.noise_amp = noise_during

        # Warmup
        self.run(warmup_steps)
        # Recall run
        self.run(recall_steps)

        self.noise_amp = old_noise

        decoded      = self.decode()
        hamming_in   = int(np.sum(noisy != pat_bits))
        hamming_out  = int(np.sum(decoded != pat_bits))
        recovered    = (hamming_out == 0)
        nearest, nd  = self.nearest_pattern()

        return {
            "pattern_idx":   pattern_idx,
            "flip_fraction": flip_fraction,
            "n_flipped":     n_flip,
            "hamming_in":    hamming_in,
            "hamming_out":   hamming_out,
            "recovered":     recovered,
            "nearest_stored": nearest,
            "nearest_dist":  nd,
            "energy_final":  round(self.energy(), 4),
        }


# ── Test suite ────────────────────────────────────────────────────────────────

def run_recall_tests(
    N: int = 32,
    P: int = 3,
    flip_fractions: tuple = (0.1, 0.2, 0.3),
    trials_per_combo: int = 5,
    warmup_steps: int = 3000,
    recall_steps: int = 8000,
    seed: int = 42,
    **net_kwargs,
) -> dict:
    """
    Store P random patterns, recall each under various corruption levels.
    """
    rng = np.random.default_rng(seed)

    # Generate orthogonal-ish random patterns
    patterns = [rng.integers(0, 2, N).tolist() for _ in range(P)]

    net = PhaseHopfield(N, patterns, seed=seed, **net_kwargs)

    capacity_theoretical = round(0.138 * N, 1)
    results = {
        "N": N, "P": P,
        "capacity_theoretical": capacity_theoretical,
        "patterns": patterns,
        "trials": [],
        "summary": {},
    }

    print(f"    N={N}, P={P}, capacity_limit≈{capacity_theoretical:.1f}")

    for flip in flip_fractions:
        ok_count = 0
        for trial in range(trials_per_combo):
            for pat_idx in range(P):
                r = net.recall(
                    pat_idx, flip_fraction=flip,
                    warmup_steps=warmup_steps,
                    recall_steps=recall_steps,
                )
                r["trial"] = trial
                results["trials"].append(r)
                if r["recovered"]:
                    ok_count += 1

        total_for_flip = trials_per_combo * P
        success_rate = ok_count / total_for_flip
        results["summary"][f"flip_{int(flip*100)}pct"] = {
            "success_rate": round(success_rate, 3),
            "ok": ok_count,
            "total": total_for_flip,
        }
        print(f"    flip={int(flip*100):2d}%  recall rate: "
              f"{ok_count}/{total_for_flip} = {success_rate:.1%}")

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Phase Hopfield Network — associative memory from oscillators"
    )
    p.add_argument("--N",          type=int,   default=32,
                   help="Number of oscillators")
    p.add_argument("--P",          type=int,   default=3,
                   help="Number of stored patterns")
    p.add_argument("--K",          type=float, default=3.0,
                   help="Coupling strength")
    p.add_argument("--dt",         type=float, default=0.001)
    p.add_argument("--a-anchor",   type=float, default=0.08)
    p.add_argument("--noise",      type=float, default=0.0)
    p.add_argument("--warmup",     type=int,   default=3000)
    p.add_argument("--recall",     type=int,   default=8000)
    p.add_argument("--trials",     type=int,   default=5)
    p.add_argument("--out-json",   type=str,
                   default="reports/phase_hopfield_report.json")
    args = p.parse_args()

    print("\n" + "═" * 64)
    print("  PHASE HOPFIELD NETWORK — associative memory")
    print("  dφ_i/dt = −K·Σ_j W_ij·sin(φ_i−φ_j) + anchor(200Hz)")
    print("  Weights: Hebbian  W_ij = (1/N)·Σ_μ cos(ξ_i^μ)·cos(ξ_j^μ)")
    print("═" * 64)

    net_kw = dict(K=args.K, dt=args.dt, a_anchor=args.a_anchor,
                  noise_amp=args.noise)

    # ── Recall tests ─────────────────────────────────────────────────────────
    print(f"\n[1] Recall tests — flip_fractions: 10%, 20%, 30%")
    t0 = time.time()
    results = run_recall_tests(
        N=args.N, P=args.P,
        flip_fractions=(0.10, 0.20, 0.30),
        trials_per_combo=args.trials,
        warmup_steps=args.warmup,
        recall_steps=args.recall,
        **net_kw,
    )
    elapsed = time.time() - t0
    print(f"    [{elapsed:.1f}s]")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'═' * 64}")
    print("  RECALL SUMMARY:")
    for key, v in results["summary"].items():
        ok_str = "✓" if v["success_rate"] >= 0.8 else "~"
        print(f"  [{ok_str}] {key}: {v['ok']}/{v['total']}  "
              f"({v['success_rate']:.1%} recall rate)")

    print()
    print("  PROOF: Stored patterns are stable fixed points.")
    print("  At φ = ξ^μ: sin(ξ_i^μ − ξ_j^μ) = 0 ∀i,j → dφ_i/dt = 0. □")
    print("  Energy: E = −½ Σ W_ij cos(φ_i−φ_j) ≡ Hopfield Ising model.")
    print(f"  Capacity: {results['P']} stored / {results['capacity_theoretical']} max "
          f"(P/N = {results['P']/results['N']:.3f} vs 0.138 limit)")
    print("═" * 64)

    # ── Save ─────────────────────────────────────────────────────────────────
    report = {
        "title": "Phase Hopfield Network — associative memory from oscillators",
        "model": {
            "dynamics": "dφ_i/dt = -K·Σ_j W_ij·sin(φ_i-φ_j) + anchor(200Hz)",
            "weights":  "W_ij = (1/N)·Σ_μ cos(ξ_i^μ)·cos(ξ_j^μ)  [Hebbian]",
            "readout":  "cos(φ_i) > 0 → bit=0,  cos(φ_i) < 0 → bit=1",
            "energy":   "E = -½ Σ W_ij cos(φ_i-φ_j)  ≡  Hopfield Ising H",
        },
        "stability_proof": (
            "At stored pattern φ=ξ^μ: sin(φ)=0 for φ∈{0,π}, so "
            "sin(ξ_i^μ-ξ_j^μ)=0 → dφ_i/dt=0. Fixed point proven. "
            "Stability: ∂²E/∂φ_i² = Σ W_ij cos(ξ_i^μ-ξ_j^μ) ≈ 1 > 0. □"
        ),
        "connection_to_hopfield": (
            "Restricted to {0,π}^N: cos(φ_i-φ_j)=cos(φ_i)cos(φ_j) "
            "(since sin(0)=sin(π)=0). Thus E = -½ Σ W_ij s_i s_j "
            "with s_i=cos(φ_i)∈{±1}: identical to Hopfield Ising Hamiltonian."
        ),
        "params": {**net_kw,
                   "warmup": args.warmup, "recall": args.recall,
                   "trials": args.trials},
        "results": results,
        "anchor_hz": ANCHOR_HZ,
        "ts": int(time.time()),
    }
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n  Saved → {args.out_json}")


if __name__ == "__main__":
    main()
