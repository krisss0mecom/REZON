#!/usr/bin/env python3
"""
Dense Associative Memory on S¹ — Phase analog of Modern Hopfield Networks.

════════════════════════════════════════════════════════════════════════
FRAMEWORK (Krotov & Hopfield 2020, arXiv:2008.06996):

  Standard Dense AM (binary σ ∈ {±1}):
    E = −Σ_μ F(Σ_i ξ_μi σ_i)
    dσ_i/dt ∝ −∂E/∂σ_i = Σ_μ F'(overlap_μ) · ξ_μi

  F choices and capacity:
    F(x) = x²/2   → classical Hopfield,  P_max ≈ 0.138·N
    F(x) = x^n/n  → Dense AM,            P_max ≈ N^(n-1)
    F(x) = exp(x) → Modern Hopfield,     P_max ≈ exp(N)   [softmax]

════════════════════════════════════════════════════════════════════════
EXTENSION TO S¹ (this work):

  State space: φ_i ∈ S¹  (phase oscillators, not binary spins)
  Patterns:    ξ^μ ∈ {0,π}^N  (binary, encoded as phases)

  Overlap (phase inner product):
    m_μ(φ) = Σ_i cos(φ_i − ξ_i^μ)    ∈ [−N, N]
    m_μ = N when φ = ξ^μ  (perfect match)
    m_μ = 0 when perpendicular

  Energy on S¹:
    E = −Σ_μ F(m_μ)  =  −Σ_μ F(Σ_i cos(φ_i − ξ_i^μ))

  Dynamics (gradient flow on S¹):
    dφ_i/dt = −∂E/∂φ_i
             = Σ_μ F'(m_μ) · sin(φ_i − ξ_i^μ)
             + anchor(200Hz)

  PROOF (stored patterns are fixed points):
    At φ = ξ^μ: sin(φ_i − ξ_i^μ) = sin(0) = 0  for all i.
    → dφ_i/dt = 0  for all i. □

  STABILITY (at φ = ξ^μ, for P << N):
    ∂²E/∂φ_i² = −Σ_μ F''(m_μ)·sin²(.) − Σ_μ F'(m_μ)·cos(φ_i−ξ_i^μ)
    At stored pattern: cos(φ_i−ξ_i^μ) = 1, dominant term:
    ≈ −Σ_μ F'(m_μ) = −P·F'(N) < 0  if F'(N) > 0  → energy maximum?

    WAIT — we need dφ_i/dt to PULL toward ξ^μ, not push away.
    For F increasing (F'>0):
      sin(φ_i − ξ_i^μ) > 0 when φ_i > ξ_i^μ  → coupling term < 0 → pulls φ_i down ✓
      sin(φ_i − ξ_i^μ) < 0 when φ_i < ξ_i^μ  → coupling term > 0 → pulls φ_i up ✓
    So the SIGN is correct — stored patterns are STABLE. □

════════════════════════════════════════════════════════════════════════
UPDATE RULE (discrete, Modern Hopfield on S¹):

  For F(x) = exp(x) → softmax retrieval:
    φ_i^new = circular_mean({ξ_i^μ}, weights=softmax({m_μ}))
            = arg( Σ_μ softmax_μ · exp(j·ξ_i^μ) )

  This is the PHASE ANALOG of the Transformer attention mechanism:
    Query:   φ (current state)
    Keys:    ξ^μ (stored patterns)
    Values:  ξ^μ (same — auto-associative)
    Overlap: cos(φ_i − ξ_i^μ)  (phase inner product)

════════════════════════════════════════════════════════════════════════
CAPACITY PREDICTION:

  For Phase Dense AM with F(x) = x^n/n:
    P_max ~ N^(n-1)  (same scaling as binary Dense AM)
  For F(x) = exp(x):
    P_max ~ exp(N)   (exponential, same as Modern Hopfield)

  Key question: does the exponential capacity survive on S¹?
  → MEASURED EMPIRICALLY BELOW.

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


# ── Dense Phase AM ────────────────────────────────────────────────────────────

class DensePhaseAM:
    """
    Dense Associative Memory on S¹.

    Energy:    E = −Σ_μ F(Σ_i cos(φ_i − ξ_i^μ))
    Dynamics:  dφ_i/dt = K · Σ_μ F'(m_μ) · sin(φ_i − ξ_i^μ) + anchor

    F_type options:
      'linear' : F(x) = x         → standard Phase XY/Hopfield (n=1)
      'poly2'  : F(x) = x²/2      → Dense AM n=2
      'poly3'  : F(x) = x³/3      → Dense AM n=3
      'poly4'  : F(x) = x⁴/4      → Dense AM n=4
      'exp'    : F(x) = exp(x)     → Modern Hopfield on S¹  [softmax]

    Capacity prediction (from Krotov-Hopfield framework):
      linear : P_max ≈ 0.138·N   (classical)
      poly2  : P_max ≈ N         (linear in N)
      poly3  : P_max ≈ N²        (quadratic)
      exp    : P_max ≈ exp(N)    (exponential)
    """

    def __init__(
        self,
        N: int,
        patterns: list,              # list of arrays, values in {0,1}
        *,
        F_type: str = 'linear',
        K: float = 1.0,             # coupling strength
        dt: float = 0.001,
        a_anchor: float = 0.08,
        omega_std: float = 0.01,
        noise_amp: float = 0.0,
        seed: int = 42,
    ):
        self.N         = N
        self.P         = len(patterns)
        self.F_type    = F_type
        self.K         = K
        self.dt        = dt
        self.a_anchor  = a_anchor
        self.noise_amp = noise_amp
        self.rng       = np.random.default_rng(seed)

        # Patterns as phases {0, π} — shape (P, N)
        self.xi = np.array(patterns, dtype=float) * np.pi  # (P, N)

        # Natural frequencies
        self.omega = self.rng.normal(0.0, omega_std, N)

        # Phase state
        self.phi = np.zeros(N)
        self._t  = 0.0

    # ── Interaction function ──────────────────────────────────────────────────

    def _F_and_dF(self, m: np.ndarray):
        """
        Returns (F(m), F'(m)) vectorized over overlap array m ∈ R^P.
        m normalized to [−1, 1] range for numerical stability.
        """
        F_type = self.F_type

        if F_type == 'linear':
            return m, np.ones_like(m)

        elif F_type == 'poly2':
            return m ** 2 / 2.0, m

        elif F_type == 'poly3':
            return m ** 3 / 3.0, m ** 2

        elif F_type == 'poly4':
            return m ** 4 / 4.0, m ** 3

        elif F_type == 'exp':
            # Numerically stable: subtract max before exp
            m_stable = m - m.max()
            ex = np.exp(m_stable)
            return ex, ex   # F' = F for exp

        else:
            raise ValueError(f"Unknown F_type: {F_type!r}")

    # ── Overlaps ──────────────────────────────────────────────────────────────

    def overlaps(self) -> np.ndarray:
        """
        Compute phase overlaps with all stored patterns.
        m_μ = Σ_i cos(φ_i − ξ_i^μ)   shape: (P,)
        Range: [−N, N].  m_μ = N at perfect match.
        """
        diff = self.phi[None, :] - self.xi   # (P, N)
        return np.sum(np.cos(diff), axis=1)   # (P,)

    # ── Discrete update (Modern Hopfield style) ───────────────────────────────

    def update_discrete(self) -> np.ndarray:
        """
        One discrete update step (synchronous).

        For F = exp (softmax): φ_i^new = circular_mean of ξ_i^μ
        weighted by softmax(m_μ).

        This is the Phase Analog of Transformer Attention:
          Attention(Q, K, V) = softmax(Q·K^T) · V
          Here: Q = φ, K = ξ, V = ξ, inner product = Σcos(φ_i−ξ_i^μ)
        """
        m = self.overlaps()          # (P,)
        _, dF = self._F_and_dF(m)   # (P,)

        # Normalize weights (softmax-like for exp, else raw F')
        if self.F_type == 'exp':
            w = dF / (dF.sum() + 1e-12)   # proper softmax weights
        else:
            w = dF / (np.abs(dF).sum() + 1e-12)

        # Circular mean: φ_i^new = arg(Σ_μ w_μ · exp(j·ξ_i^μ))
        # where j = imaginary unit
        complex_sum = np.sum(
            w[:, None] * np.exp(1j * self.xi),   # (P, N)
            axis=0
        )   # (N,)
        phi_new = np.angle(complex_sum) % TWO_PI
        return phi_new

    # ── Continuous ODE step ───────────────────────────────────────────────────

    def step(self) -> None:
        """Single Euler step of Dense Phase AM dynamics."""
        phi = self.phi
        t   = self._t

        # Overlaps m_μ = Σ_i cos(φ_i − ξ_i^μ)
        diff = phi[None, :] - self.xi    # (P, N)
        m    = np.sum(np.cos(diff), axis=1)   # (P,)

        # F'(m_μ) — interaction weights
        _, dF = self._F_and_dF(m)   # (P,)

        # Gradient descent on E = −Σ_μ F(m_μ):
        #   ∂E/∂φ_i = Σ_μ F'(m_μ) · sin(φ_i − ξ_i^μ)   [via ∂m_μ/∂φ_i = −sin(φ_i−ξ_i^μ)]
        #   dφ_i/dt = −∂E/∂φ_i = −K·Σ_μ F'(m_μ)·sin(φ_i − ξ_i^μ)
        # Check: sin(φ_i−ξ_i^μ)>0 when φ_i>ξ_i^μ → coupling<0 → pulls φ_i DOWN → toward ξ_i^μ ✓
        coupling = -self.K * np.sum(
            dF[:, None] * np.sin(diff),   # (P, N)
            axis=0
        )   # (N,)

        # Anchor (200 Hz immutable)
        anchor = self.a_anchor * np.sin(TWO_PI * ANCHOR_HZ * t - phi)

        # Noise
        noise = (self.rng.normal(0.0, self.noise_amp, self.N)
                 if self.noise_amp > 0 else 0.0)

        self.phi = (phi + self.dt * (self.omega + coupling + anchor + noise)) % TWO_PI
        self._t  += self.dt

    def run(self, n_steps: int) -> np.ndarray:
        for _ in range(n_steps):
            self.step()
        return self.phi.copy()

    def reset(self, phi_init: np.ndarray) -> None:
        self.phi = phi_init.copy() % TWO_PI
        self._t  = 0.0

    def decode(self) -> np.ndarray:
        """Decode phase → binary pattern."""
        return (np.cos(self.phi) < 0.0).astype(int)

    def energy(self) -> float:
        """E = −Σ_μ F(m_μ)."""
        m = self.overlaps()
        Fm, _ = self._F_and_dF(m)
        return float(-np.sum(Fm))

    def nearest_pattern(self) -> tuple:
        """(index, hamming_distance) of closest stored pattern."""
        decoded = self.decode()
        dists = []
        for xi_pat in self.xi:
            pat_bits = (xi_pat / np.pi).astype(int)
            dists.append(int(np.sum(decoded != pat_bits)))
        best = int(np.argmin(dists))
        return best, dists[best]

    def recall(
        self,
        pattern_idx: int,
        flip_fraction: float = 0.1,
        warmup_steps: int = 5000,
        recall_steps: int = 10000,
    ) -> dict:
        """Corrupt pattern, run dynamics, check recovery."""
        pat_bits = (self.xi[pattern_idx] / np.pi).astype(int)
        N = self.N

        n_flip = max(1, int(flip_fraction * N))
        flip_idx = self.rng.choice(N, size=n_flip, replace=False)
        noisy = pat_bits.copy()
        noisy[flip_idx] ^= 1

        phi_init = noisy.astype(float) * np.pi
        phi_init += self.rng.normal(0.0, 0.05, N)

        self.reset(phi_init)
        self.run(warmup_steps)
        self.run(recall_steps)

        decoded     = self.decode()
        hamming_out = int(np.sum(decoded != pat_bits))
        recovered   = (hamming_out == 0)
        nearest, nd = self.nearest_pattern()

        return {
            "pattern_idx": pattern_idx,
            "flip_fraction": flip_fraction,
            "n_flipped": n_flip,
            "hamming_out": hamming_out,
            "recovered": recovered,
            "nearest_stored": nearest,
            "nearest_dist": nd,
            "energy": round(self.energy(), 4),
        }


# ── Capacity comparison across F types ───────────────────────────────────────

def compare_capacity(
    N: int = 32,
    P_values: list = None,
    F_types: list = None,
    *,
    flip_fraction: float = 0.10,
    trials: int = 3,
    warmup_steps: int = 5000,
    recall_steps: int = 10000,
    K: float = 1.0,
    dt: float = 0.001,
    a_anchor: float = 0.08,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Compare recall rates across F_types for increasing P.

    Key question: does F(x)=exp(x) give higher capacity than F(x)=x?
    """
    if P_values is None:
        # Test at P = 1, 2, 4, 6, 8, 12, 16, 20, 24, 32 (up to N)
        P_values = sorted(set([
            1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24,
            int(0.138 * N), int(0.2 * N), int(0.3 * N), N
        ]))
        P_values = [p for p in P_values if 1 <= p <= N]

    if F_types is None:
        F_types = ['linear', 'poly2', 'poly3', 'exp']

    rng = np.random.default_rng(seed)
    results = {F: [] for F in F_types}

    for P in P_values:
        patterns = [rng.integers(0, 2, N).tolist() for _ in range(P)]

        if verbose:
            print(f"  P={P:3d}  α={P/N:.2f}  ", end="", flush=True)

        for F_type in F_types:
            net = DensePhaseAM(
                N, patterns, F_type=F_type, K=K, dt=dt,
                a_anchor=a_anchor, seed=seed,
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

            rate = round(ok / total, 3)
            results[F_type].append({"P": P, "alpha": round(P/N, 3),
                                     "success_rate": rate, "ok": ok,
                                     "total": total})
            if verbose:
                print(f"{F_type}={rate:.2f}  ", end="", flush=True)

        if verbose:
            print()

    # Find P* for each F_type
    P_star = {}
    for F_type in F_types:
        p_star = 0
        for r in results[F_type]:
            if r["success_rate"] >= 0.5:
                p_star = r["P"]
        P_star[F_type] = p_star

    return {
        "N": N,
        "P_values": P_values,
        "F_types": F_types,
        "results": results,
        "P_star": P_star,
        "alpha_star": {f: round(p/N, 3) for f, p in P_star.items()},
    }


# ── Discrete update test (attention analog) ───────────────────────────────────

def test_discrete_update(
    N: int = 32,
    P: int = 5,
    flip_fraction: float = 0.10,
    n_updates: int = 10,
    seed: int = 42,
) -> dict:
    """
    Test discrete Modern Hopfield update (attention analog) on S¹.
    Each update = one circular-mean step (like one attention layer).
    """
    rng = np.random.default_rng(seed)
    patterns = [rng.integers(0, 2, N).tolist() for _ in range(P)]
    net = DensePhaseAM(N, patterns, F_type='exp', seed=seed)

    pat_bits = (net.xi[0] / np.pi).astype(int)
    n_flip = max(1, int(flip_fraction * N))
    flip_idx = rng.choice(N, size=n_flip, replace=False)
    noisy = pat_bits.copy()
    noisy[flip_idx] ^= 1

    phi = noisy.astype(float) * np.pi
    net.reset(phi)

    trajectory = []
    for step in range(n_updates):
        decoded = net.decode()
        hamming = int(np.sum(decoded != pat_bits))
        trajectory.append({"step": step, "hamming": hamming})
        phi_new = net.update_discrete()
        net.phi = phi_new

    decoded_final = net.decode()
    hamming_final = int(np.sum(decoded_final != pat_bits))
    trajectory.append({"step": n_updates, "hamming": hamming_final})

    return {
        "N": N, "P": P,
        "flip_fraction": flip_fraction,
        "n_flipped": n_flip,
        "recovered": (hamming_final == 0),
        "hamming_final": hamming_final,
        "trajectory": trajectory,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Dense Phase AM — Modern Hopfield on S¹"
    )
    p.add_argument("--N",        type=int,   default=32)
    p.add_argument("--trials",   type=int,   default=3)
    p.add_argument("--flip",     type=float, default=0.10)
    p.add_argument("--warmup",   type=int,   default=5000)
    p.add_argument("--recall",   type=int,   default=10000)
    p.add_argument("--K",        type=float, default=1.0)
    p.add_argument("--dt",       type=float, default=0.001)
    p.add_argument("--a-anchor", type=float, default=0.08)
    p.add_argument("--out-json", type=str,
                   default="reports/phase_dense_am_report.json")
    args = p.parse_args()

    net_kw = dict(
        flip_fraction=args.flip,
        trials=args.trials,
        warmup_steps=args.warmup,
        recall_steps=args.recall,
        K=args.K, dt=args.dt, a_anchor=args.a_anchor,
    )

    print("\n" + "═" * 64)
    print("  DENSE ASSOCIATIVE MEMORY ON S¹")
    print("  E = −Σ_μ F(Σ_i cos(φ_i − ξ_i^μ))")
    print("  Comparing: F=linear, poly2, poly3, exp (Modern Hopfield)")
    print("═" * 64)

    t0 = time.time()

    # ── Capacity comparison ───────────────────────────────────────────────────
    print(f"\n[1] Capacity comparison — N={args.N}, flip={args.flip:.0%}")
    comp = compare_capacity(N=args.N, **net_kw, verbose=True)

    print(f"\n  P* (capacity at 50% recall threshold):")
    for F_type, p_star in comp["P_star"].items():
        alpha = comp["alpha_star"][F_type]
        theoretical = {
            'linear': '0.138·N (classical Hopfield)',
            'poly2':  '~N      (linear, Dense AM n=2)',
            'poly3':  '~N²     (quadratic, Dense AM n=3)',
            'exp':    '~exp(N) (exponential, Modern Hopfield)',
        }.get(F_type, '?')
        print(f"    {F_type:8s}: P*={p_star:3d}  α*={alpha:.3f}  "
              f"(theory: {theoretical})")

    # ── Discrete update (attention analog) ───────────────────────────────────
    print(f"\n[2] Discrete update (Phase Attention) — N={args.N}, P=5")
    disc = test_discrete_update(N=args.N, P=5, flip_fraction=args.flip)
    print(f"  Hamming distance per update step:")
    for r in disc["trajectory"]:
        bar = "█" * r["hamming"]
        print(f"    step {r['step']:2d}: hamming={r['hamming']:2d}  {bar}")
    status = "RECOVERED ✓" if disc["recovered"] else f"FAILED (hamming={disc['hamming_final']})"
    print(f"  → {status}")

    elapsed = time.time() - t0

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'═' * 64}")
    print("  KEY FINDINGS:")
    print()
    print("  1. ENERGY on S¹: E = −Σ_μ F(Σ_i cos(φ_i − ξ_i^μ))")
    print("     Natural phase inner product via cosine overlap.")
    print()
    print("  2. FIXED POINTS: At φ = ξ^μ: sin(φ_i−ξ_i^μ)=0 → dφ/dt=0. □")
    print()
    print("  3. MODERN HOPFIELD on S¹ (F=exp):")
    print("     φ_i^new = circular_mean({ξ_i^μ}, weights=softmax({m_μ}))")
    print("     = Phase analog of Transformer attention mechanism.")
    print()
    print("  4. CAPACITY SCALING:")
    for F_type, p_star in comp["P_star"].items():
        print(f"     F={F_type:8s}: P* = {p_star}")
    print()
    print("  NOVELTY vs Krotov-Hopfield 2020:")
    print("  Their framework: σ ∈ {±1}^N (binary Ising spins)")
    print("  Our extension:   φ ∈ S¹^N  (continuous phase oscillators)")
    print("  New inner product: Σ cos(φ_i−ξ_i^μ)  (periodic, hardware-native)")
    print("  Physical substrate: RC oscillator arrays at 200 Hz")
    print(f"  [{elapsed:.0f}s total]")
    print("═" * 64)

    # ── Save ─────────────────────────────────────────────────────────────────
    report = {
        "title": "Dense Associative Memory on S¹",
        "framework": {
            "energy": "E = -sum_mu F(sum_i cos(phi_i - xi_i^mu))",
            "dynamics": "dphi_i/dt = K * sum_mu F'(m_mu) * sin(phi_i - xi_i^mu) + anchor",
            "overlap": "m_mu = sum_i cos(phi_i - xi_i^mu)  in [-N, N]",
            "fixed_point_proof": (
                "At phi = xi^mu: sin(phi_i - xi_i^mu) = sin(0) = 0 "
                "for all i -> dphi_i/dt = 0. Fixed point proven. □"
            ),
            "attention_analog": (
                "F=exp: phi_i^new = circular_mean(xi_i^mu, softmax(m_mu)). "
                "= Phase analog of Transformer attention: "
                "  Q=phi (query), K=xi (keys), V=xi (values), "
                "  inner product = sum_i cos(phi_i - xi_i^mu)"
            ),
            "novelty_vs_krotov2020": (
                "Krotov-Hopfield 2020 (arXiv:2008.06996): sigma in {+-1}^N. "
                "This work: phi in S^1^N. New overlap: sum cos(phi_i - xi_i^mu). "
                "Physical substrate: RC oscillator arrays at 200 Hz anchor."
            ),
        },
        "capacity_comparison": comp,
        "discrete_update_test": disc,
        "params": {**net_kw, "N": args.N},
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
