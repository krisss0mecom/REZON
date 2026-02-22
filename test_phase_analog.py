"""Tests for phase_analog.py — analog (fuzzy-like) phase computing."""
import numpy as np
import pytest
from phase_analog import analog_gate, analog_sweep

GATE_KW = dict(warmup=2000, collect=400, dt=0.001,
               a_anchor=0.08, K_inj=8.0, K_main=5.0, K_bias=1.5)
TOL = 0.20


# ── NOT gate (exact complement) ───────────────────────────────────────────────

def test_not_binary_endpoints():
    """NOT(0)=1, NOT(1)=0 — exact for binary inputs."""
    r0 = analog_gate("NOT", 0.0, 0.0, run_seed=2, **GATE_KW)
    r1 = analog_gate("NOT", 1.0, 0.0, run_seed=3, **GATE_KW)
    assert r0["out_value"] > 1.0 - TOL, f"NOT(0)={r0['out_value']:.3f}, expected ~1"
    assert r1["out_value"] < TOL,        f"NOT(1)={r1['out_value']:.3f}, expected ~0"


def test_not_complement_analog():
    """NOT(x) ≈ 1-x (analytical result from anti-sync attractor)."""
    for seed, x in enumerate([0.25, 0.5, 0.75]):
        r = analog_gate("NOT", x, 0.0, run_seed=seed * 3, **GATE_KW)
        expected = 1.0 - x
        assert abs(r["out_value"] - expected) < TOL, \
            f"NOT({x}): out={r['out_value']:.3f}, expected≈{expected:.3f}"


# ── AND gate (threshold / conditional wire) ───────────────────────────────────

def test_and_binary_endpoints():
    """AND(0,0)=0, AND(1,1)=1."""
    r00 = analog_gate("AND", 0.0, 0.0, run_seed=0, **GATE_KW)
    r11 = analog_gate("AND", 1.0, 1.0, run_seed=1, **GATE_KW)
    assert r00["out_value"] < TOL,        f"AND(0,0)={r00['out_value']:.3f}"
    assert r11["out_value"] > 1.0 - TOL, f"AND(1,1)={r11['out_value']:.3f}"


def test_and_binary_mixed():
    """AND(1,0)=0, AND(0,1)=0."""
    r10 = analog_gate("AND", 1.0, 0.0, run_seed=4, **GATE_KW)
    r01 = analog_gate("AND", 0.0, 1.0, run_seed=5, **GATE_KW)
    assert r10["out_value"] < TOL, f"AND(1,0)={r10['out_value']:.3f}"
    assert r01["out_value"] < TOL, f"AND(0,1)={r01['out_value']:.3f}"


def test_and_threshold_behavior():
    """AND output transitions from 0 to target as control x increases past threshold."""
    # With y=1, as x grows from 0 to 1, output should increase
    outs = []
    for seed, x in enumerate(np.linspace(0.0, 1.0, 5)):
        r = analog_gate("AND", float(x), 1.0, run_seed=seed, **GATE_KW)
        outs.append(r["out_value"])
    # First should be near 0, last should be near 1
    assert outs[0] < 0.3,  f"AND(0, 1) should be ~0, got {outs[0]:.3f}"
    assert outs[-1] > 0.7, f"AND(1, 1) should be ~1, got {outs[-1]:.3f}"


# ── OR gate ───────────────────────────────────────────────────────────────────

def test_or_binary_endpoints():
    """OR(0,0)=0, OR(1,0)=1, OR(0,1)=1, OR(1,1)=1."""
    cases = [(0.0, 0.0, 0.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0), (1.0, 1.0, 1.0)]
    for seed, (x, y, exp) in enumerate(cases):
        r = analog_gate("OR", x, y, run_seed=seed + 10, **GATE_KW)
        if exp == 0.0:
            assert r["out_value"] < TOL, f"OR({x},{y})={r['out_value']:.3f}"
        else:
            assert r["out_value"] > 1.0 - TOL, f"OR({x},{y})={r['out_value']:.3f}"


# ── XOR gate ──────────────────────────────────────────────────────────────────

def test_xor_binary_endpoints():
    """XOR binary truth table: 0⊕0=0, 0⊕1=1, 1⊕0=1, 1⊕1=0."""
    cases = [(0.0, 0.0, 0.0), (0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 0.0)]
    for seed, (x, y, exp) in enumerate(cases):
        r = analog_gate("XOR", x, y, run_seed=seed * 11, **GATE_KW)
        assert abs(r["out_value"] - exp) < TOL, \
            f"XOR({x},{y})={r['out_value']:.3f}, expected {exp}"


def test_xor_conditional_flip():
    """XOR(0.9, y): output ≈ 1-y (control≈1 flips target)."""
    for seed, y in enumerate([0.0, 1.0]):
        r = analog_gate("XOR", 0.9, y, run_seed=seed + 20, **GATE_KW)
        expected = 1.0 - y
        assert abs(r["out_value"] - expected) < TOL, \
            f"XOR(0.9, {y})={r['out_value']:.3f}, expected {expected}"


# ── Sweep and monotonicity ────────────────────────────────────────────────────

def test_and_monotone():
    """AND output should be monotone: AND(high,high) > AND(low,low)."""
    r_low  = analog_gate("AND", 0.1, 0.1, run_seed=0, **GATE_KW)
    r_high = analog_gate("AND", 0.9, 0.9, run_seed=1, **GATE_KW)
    assert r_high["out_value"] > r_low["out_value"], \
        f"AND not monotone: low={r_low['out_value']:.3f} high={r_high['out_value']:.3f}"


def test_not_sweep_error():
    """NOT sweep: mean error vs 1-x should be < 0.10 (analytically exact)."""
    res = analog_sweep("NOT", grid_points=4, **GATE_KW)
    assert res["mean_error"] < 0.10, \
        f"NOT sweep mean_error={res['mean_error']:.4f}, expected < 0.10"
