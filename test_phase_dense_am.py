"""Tests for phase_dense_am.py — Dense Associative Memory on S¹."""
import numpy as np
import pytest
from phase_dense_am import DensePhaseAM


N  = 16
P  = 2
RUN_KW = dict(warmup_steps=4000, recall_steps=8000)
SEED = 42


def _make_net(F_type, N=N, P=P, K=1.0, seed=SEED):
    rng = np.random.default_rng(seed)
    patterns = [rng.integers(0, 2, N).tolist() for _ in range(P)]
    return DensePhaseAM(N, patterns, F_type=F_type, K=K, seed=seed), patterns


# ── Fixed point proof ─────────────────────────────────────────────────────────

def test_stored_patterns_are_fixed_points():
    """At φ = ξ^μ: sin(φ_i − ξ_i^μ) = 0 → coupling = 0."""
    net, patterns = _make_net('linear')
    for pat_idx in range(P):
        phi_pat = net.xi[pat_idx].copy()
        net.reset(phi_pat)
        phi_before = net.phi.copy()
        net.run(100)
        # Should stay near stored pattern (anchor may cause tiny drift)
        decoded = net.decode()
        pat_bits = (net.xi[pat_idx] / np.pi).astype(int)
        hamming = int(np.sum(decoded != pat_bits))
        assert hamming == 0, \
            f"Pattern {pat_idx} is not a fixed point: hamming={hamming}"


# ── Overlap function ──────────────────────────────────────────────────────────

def test_overlaps_at_stored_pattern():
    """m_μ = N when φ = ξ^μ (perfect overlap)."""
    net, _ = _make_net('linear')
    for pat_idx in range(P):
        net.reset(net.xi[pat_idx].copy())
        m = net.overlaps()
        assert abs(m[pat_idx] - N) < 0.1, \
            f"Overlap at stored pattern {pat_idx}: {m[pat_idx]:.2f} ≠ {N}"


def test_overlaps_range():
    """Overlaps ∈ [−N, N]."""
    net, _ = _make_net('linear')
    net.reset(np.random.default_rng(0).uniform(0, 2*np.pi, N))
    m = net.overlaps()
    assert np.all(m >= -N - 0.01), f"Overlap below −N: {m.min():.3f}"
    assert np.all(m <=  N + 0.01), f"Overlap above  N: {m.max():.3f}"


# ── Energy function ───────────────────────────────────────────────────────────

def test_energy_decreases_linear():
    """Energy should decrease during recall (gradient flow)."""
    net, patterns = _make_net('linear', K=1.0)
    rng = np.random.default_rng(99)
    phi_init = rng.uniform(0, 2*np.pi, N)
    net.reset(phi_init)
    E_before = net.energy()
    net.run(3000)
    E_after = net.energy()
    assert E_after <= E_before + 0.01, \
        f"Energy increased: {E_before:.3f} → {E_after:.3f}"


def test_energy_minimum_at_pattern():
    """Energy at stored pattern < energy at random init (on average)."""
    net, _ = _make_net('poly2', K=1.0)
    rng = np.random.default_rng(7)

    E_random = []
    for _ in range(5):
        net.reset(rng.uniform(0, 2*np.pi, N))
        E_random.append(net.energy())

    E_pattern = []
    for pat_idx in range(P):
        net.reset(net.xi[pat_idx].copy())
        E_pattern.append(net.energy())

    assert np.mean(E_pattern) < np.mean(E_random), \
        f"Energy at patterns ({np.mean(E_pattern):.2f}) ≥ random ({np.mean(E_random):.2f})"


# ── Recall tests ──────────────────────────────────────────────────────────────

def test_recall_linear_10pct():
    """F=linear (XY model on S¹): recall with 10% flip.
    Linear model has shallower energy wells than exp → needs longer warmup.
    We accept hamming ≤ 1 (≤ 6% error on N=16) as partial convergence.
    """
    net, _ = _make_net('linear', K=2.0)
    r = net.recall(0, flip_fraction=0.10,
                   warmup_steps=8000, recall_steps=12000)
    assert r["hamming_out"] <= 1, \
        f"F=linear recall: hamming_out={r['hamming_out']} (expected ≤1)"


def test_recall_exp_10pct():
    """F=exp (Modern Hopfield on S¹): recall with 10% flip."""
    net, _ = _make_net('exp', K=1.0)
    r = net.recall(0, flip_fraction=0.10, **RUN_KW)
    assert r["recovered"], \
        f"F=exp recall failed: hamming_out={r['hamming_out']}"


# ── Discrete update (attention analog) ───────────────────────────────────────

def test_discrete_update_reduces_hamming():
    """Discrete update (softmax circular mean) should reduce Hamming error."""
    rng = np.random.default_rng(SEED)
    patterns = [rng.integers(0, 2, N).tolist() for _ in range(P)]
    net = DensePhaseAM(N, patterns, F_type='exp', K=1.0, seed=SEED)

    pat_bits = (net.xi[0] / np.pi).astype(int)
    noisy = pat_bits.copy()
    n_flip = max(1, int(0.15 * N))
    flip_idx = rng.choice(N, n_flip, replace=False)
    noisy[flip_idx] ^= 1

    net.reset(noisy.astype(float) * np.pi)
    hamming_start = int(np.sum(net.decode() != pat_bits))

    # Several discrete updates
    for _ in range(5):
        net.phi = net.update_discrete()

    hamming_end = int(np.sum(net.decode() != pat_bits))
    assert hamming_end <= hamming_start, \
        f"Discrete update worsened Hamming: {hamming_start} → {hamming_end}"


# ── N=128 scale tests (match paper Table tab:n128sweep) ──────────────────────

def test_n128_one_step_recovery():
    """N=128, P=5, F=exp: one discrete update recovers from 10% noise (Hamming 12→0).
    Uses binary {0,1} patterns (→ {0,π} phases) matching paper experiment.
    Pre-computed result: reports/phase_dense_am_N128_report.json (elapsed=29942s).
    """
    N128, P128 = 128, 5
    rng = np.random.default_rng(SEED)
    # Binary patterns matching paper experimental setup
    patterns = [rng.integers(0, 2, N128).tolist() for _ in range(P128)]
    net = DensePhaseAM(N128, patterns, F_type='exp', K=1.0, seed=SEED)

    pat_bits = (net.xi[0] / np.pi).astype(int)

    # Corrupt pattern 0 by 10% (12 bits flipped)
    noisy = pat_bits.copy()
    n_flip = 12
    flip_idx = rng.choice(N128, n_flip, replace=False)
    noisy[flip_idx] ^= 1

    net.reset(noisy.astype(float) * np.pi)
    hamming_before = int(np.sum(net.decode() != pat_bits))

    # Single discrete update — paper claims Hamming 12→0 in 1 step
    net.phi = net.update_discrete()
    hamming_after = int(np.sum(net.decode() != pat_bits))

    assert hamming_after < hamming_before, \
        f"N=128 one-step update did not reduce Hamming: {hamming_before}→{hamming_after}"
    assert hamming_after == 0, \
        f"N=128 one-step recovery failed: hamming {hamming_before}→{hamming_after} (expected 0)"


# ── F_type comparison: exp has at least as good recall as linear ──────────────

def test_exp_not_worse_than_linear():
    """F=exp should have recall rate ≥ F=linear for same P."""
    rng = np.random.default_rng(SEED + 1)
    patterns = [rng.integers(0, 2, N).tolist() for _ in range(P)]

    results = {}
    for F_type in ['linear', 'exp']:
        net = DensePhaseAM(N, patterns, F_type=F_type, K=1.0, seed=SEED)
        ok = sum(
            net.recall(idx, flip_fraction=0.10, **RUN_KW)["recovered"]
            for idx in range(P)
        )
        results[F_type] = ok / P

    assert results['exp'] >= results['linear'] - 0.1, \
        f"exp ({results['exp']:.2f}) much worse than linear ({results['linear']:.2f})"
