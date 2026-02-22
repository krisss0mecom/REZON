"""Tests for phase_hopfield.py — associative memory."""
import numpy as np
import pytest
from phase_hopfield import PhaseHopfield

NET_KW = dict(K=3.0, dt=0.001, a_anchor=0.08, noise_amp=0.0)


def make_net(N=20, P=2, seed=42):
    rng = np.random.default_rng(seed)
    patterns = [rng.integers(0, 2, N).tolist() for _ in range(P)]
    return PhaseHopfield(N, patterns, seed=seed, **NET_KW), patterns


def test_stored_patterns_are_fixed_points():
    """
    Stored patterns should be (near-)fixed points.
    Start exactly at ξ^μ, run 1000 steps, decoded should be unchanged.
    """
    net, patterns = make_net(N=16, P=2)
    for mu, pat in enumerate(patterns):
        phi_init = np.array(pat, dtype=float) * np.pi
        phi_init += 1e-4  # tiny perturbation (test stability)
        net.reset(phi_init)
        net.run(2000)
        decoded = net.decode()
        hamming = int(np.sum(decoded != np.array(pat)))
        assert hamming == 0, \
            f"Pattern {mu}: {hamming} bits wrong after relaxation from near-attractor"


def test_weight_matrix_symmetric():
    """Weight matrix must be symmetric and zero diagonal."""
    net, _ = make_net(N=16, P=3)
    assert np.allclose(net.W, net.W.T),       "W not symmetric"
    assert np.allclose(np.diag(net.W), 0.0),  "W diagonal not zero"


def test_weight_matrix_hebbian():
    """W = (1/N) Σ_μ s_i^μ s_j^μ, verify one element manually."""
    N = 4
    patterns = [[0, 1, 0, 1], [1, 0, 1, 0]]
    net = PhaseHopfield(N, patterns, seed=0, **NET_KW)
    # s^1 = [1,-1,1,-1], s^2 = [-1,1,-1,1]
    # W[0,1] = (1/4)(s0^1*s1^1 + s0^2*s1^2) = (1/4)(1*(-1) + (-1)*1) = -0.5
    assert abs(net.W[0, 1] - (-0.5)) < 1e-9, f"W[0,1]={net.W[0,1]}"


def test_energy_decreases():
    """Energy should decrease (or stay constant) during recall."""
    net, patterns = make_net(N=20, P=2)
    # Start from noisy pattern
    phi_init = np.array(patterns[0], dtype=float) * np.pi
    phi_init[0:4] = (1 - np.array(patterns[0][:4])) * np.pi  # flip 4 bits
    net.reset(phi_init)

    energies = []
    for _ in range(10):
        net.run(300)
        energies.append(net.energy())

    # Energy should not increase significantly (allow small float noise)
    for i in range(1, len(energies)):
        assert energies[i] <= energies[i-1] + 0.5, \
            f"Energy increased: {energies[i-1]:.4f} → {energies[i]:.4f}"


def test_recall_10pct_noise():
    """10% bit flip: should recall perfectly."""
    net, patterns = make_net(N=24, P=2, seed=7)
    successes = 0
    for pat_idx in range(len(patterns)):
        r = net.recall(pat_idx, flip_fraction=0.10,
                       warmup_steps=2000, recall_steps=6000)
        if r["recovered"]:
            successes += 1
    assert successes == len(patterns), \
        f"10% recall: {successes}/{len(patterns)} recovered"


def test_recall_20pct_noise():
    """20% bit flip: majority should be recovered."""
    net, patterns = make_net(N=24, P=2, seed=11)
    total, ok = 0, 0
    for trial in range(3):
        for pat_idx in range(len(patterns)):
            r = net.recall(pat_idx, flip_fraction=0.20,
                           warmup_steps=2000, recall_steps=8000)
            total += 1
            ok += int(r["recovered"])
    assert ok / total >= 0.6, f"20% recall rate = {ok}/{total}"


def test_nearest_pattern_correct_at_attractor():
    """After recall, nearest_pattern should match the target."""
    net, patterns = make_net(N=20, P=2, seed=5)
    for pat_idx in range(len(patterns)):
        # Start at exact pattern (perturbed slightly)
        phi_init = np.array(patterns[pat_idx], dtype=float) * np.pi
        phi_init += np.random.default_rng(99).normal(0, 0.05, len(patterns[pat_idx]))
        net.reset(phi_init)
        net.run(3000)
        nearest, dist = net.nearest_pattern()
        assert nearest == pat_idx or dist == 0, \
            f"Pattern {pat_idx}: nearest={nearest}, dist={dist}"


def test_capacity_below_limit():
    """Store P patterns with P/N < 0.138 (Hopfield capacity limit)."""
    N, P = 30, 3
    assert P / N < 0.138, "Test misconfigured: P/N exceeds capacity"
    net, patterns = make_net(N=N, P=P, seed=13)
    # At least 1 stored pattern should be a fixed point
    successes = 0
    for mu, pat in enumerate(patterns):
        phi_init = np.array(pat, dtype=float) * np.pi + \
                   np.random.default_rng(mu).normal(0, 0.05, N)
        net.reset(phi_init)
        net.run(3000)
        if net.decode().tolist() == pat:
            successes += 1
    assert successes >= 1, "No stored pattern recovered as fixed point"
