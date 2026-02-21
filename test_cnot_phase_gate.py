#!/usr/bin/env python3
"""
Test suite for CNOTPhaseGate — pure phase CNOT without RLS.
Verifies 4/4 truth table across multiple seeds and under noise.
"""
import numpy as np
import pytest
from cnot_phase_gate import CNOTPhaseGate, run_truth_table, evaluate_multi_seed

BASE_CFG = dict(dt=0.001, K_inj=8.0, K_cnot=5.0, a_anchor=0.08, leak=0.0, noise_amp=0.0)


def test_single_seed_4of4():
    gate = CNOTPhaseGate(**BASE_CFG, seed=42)
    score, rows = run_truth_table(gate, warmup=2000, collect=400, run_seed=42)
    assert score == 4, f"Expected 4/4, got {score}/4: {rows}"


def test_multi_seed_100pct():
    multi = evaluate_multi_seed(BASE_CFG, warmup=2000, collect=400, n_seeds=50, gate_seed=42)
    assert multi["pass4_rate"] == 1.0, (
        f"Expected pass4=1.0, got {multi['pass4_rate']:.3f}  dist={multi['distribution']}"
    )


def test_noise_robust():
    """Gate must achieve 4/4 even with heavy noise (noise_amp=1.0)."""
    cfg_noisy = {**BASE_CFG, "noise_amp": 1.0}
    multi = evaluate_multi_seed(cfg_noisy, warmup=2000, collect=400, n_seeds=50, gate_seed=42)
    assert multi["pass4_rate"] == 1.0, (
        f"Noisy gate: expected pass4=1.0, got {multi['pass4_rate']:.3f}"
    )


def test_cos_values_near_unity():
    """mean_cos_out should be close to ±1 (strong attractor convergence)."""
    gate = CNOTPhaseGate(**BASE_CFG, seed=42)
    _, rows = run_truth_table(gate, warmup=2000, collect=400, run_seed=42)
    for r in rows:
        assert abs(r["mean_cos_out"]) > 0.9, (
            f"Weak convergence C={r['control']} T={r['target']}: mean_cos={r['mean_cos_out']:.4f}"
        )


def test_anchor_immutable():
    import pytest
    with pytest.raises(ValueError, match="200"):
        from cnot_phase_gate import ensure_anchor_immutable
        ensure_anchor_immutable(100.0)
