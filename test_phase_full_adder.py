"""Tests for phase_full_adder.py"""
import numpy as np
import pytest
from phase_full_adder import (
    run_gate_phase, phase_to_bit, full_adder_1bit, test_full_adder_1bit,
    ripple_carry_adder_4bit,
)

GATE_KW = dict(warmup=2000, collect=400, dt=0.001,
               a_anchor=0.08, K_inj=8.0, K_main=5.0, K_bias=1.5)


def test_phase_to_bit():
    assert phase_to_bit(0.0)       == 0   # cos(0)=1 > 0
    assert phase_to_bit(np.pi)     == 1   # cos(π)=-1 < 0
    assert phase_to_bit(np.pi/4)   == 0   # cos(π/4)>0
    assert phase_to_bit(3*np.pi/4) == 1   # cos(3π/4)<0


def test_run_gate_phase_xor_binary():
    """XOR gate with binary inputs should recover standard truth table."""
    cases = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]
    for a, b, expected in cases:
        phi = run_gate_phase("XOR", float(a)*np.pi, float(b)*np.pi,
                             run_seed=42, **GATE_KW)
        assert phase_to_bit(phi) == expected, \
            f"XOR({a},{b}) → bit={phase_to_bit(phi)}, expected {expected}"


def test_run_gate_phase_and_binary():
    """AND gate with binary inputs."""
    cases = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)]
    for a, b, expected in cases:
        phi = run_gate_phase("AND", float(a)*np.pi, float(b)*np.pi,
                             run_seed=7, **GATE_KW)
        assert phase_to_bit(phi) == expected, \
            f"AND({a},{b}) → bit={phase_to_bit(phi)}, expected {expected}"


def test_full_adder_1bit_all_cases():
    """All 8 combinations of (a,b,cin) must be correct."""
    res = test_full_adder_1bit(**GATE_KW)
    assert res["score"] == res["total"], \
        f"1-bit FA: {res['score']}/{res['total']} correct\n" + \
        "\n".join(str(r) for r in res["rows"] if not r["ok"])


def test_full_adder_specific_carry():
    """1+1+1=3: SUM=1, CARRY=1 (hardest case)."""
    r = full_adder_1bit(1, 1, 1, run_seed=0, **GATE_KW)
    assert r["sum"]   == 1, f"SUM=1 expected, got {r['sum']}"
    assert r["carry"] == 1, f"CARRY=1 expected, got {r['carry']}"


def test_full_adder_zero_carry():
    """0+0+0=0: SUM=0, CARRY=0."""
    r = full_adder_1bit(0, 0, 0, run_seed=1, **GATE_KW)
    assert r["sum"]   == 0
    assert r["carry"] == 0


def test_phase_carry_propagation():
    """Carry phase from one full adder feeds as cin_phi to next."""
    # 1+1=2, so first bit: SUM=0, CARRY=1 → carry_phi ≈ π
    r = full_adder_1bit(1, 1, 0, run_seed=5, **GATE_KW)
    assert r["carry"] == 1
    carry_phi = r["phi_carry"]
    assert abs(np.cos(carry_phi) + 1.0) < 0.5, \
        f"carry_phi={carry_phi:.3f} should be near π, cos={np.cos(carry_phi):.3f}"


def test_ripple_carry_4bit_corners():
    """Test corner cases: 0+0, 15+0, 7+8, 15+15."""
    for a, b in [(0, 0), (15, 0), (0, 15), (7, 8), (15, 15)]:
        r = ripple_carry_adder_4bit(a, b, run_seed=42, **GATE_KW)
        assert r["ok"], \
            f"{a}+{b}={a+b}: got {r['pred_total']}"


def test_ripple_carry_4bit_random():
    """10 random pairs must be correct."""
    rng = np.random.default_rng(99)
    errors = []
    for trial in range(10):
        a, b = int(rng.integers(0, 16)), int(rng.integers(0, 16))
        r = ripple_carry_adder_4bit(a, b, run_seed=trial * 7, **GATE_KW)
        if not r["ok"]:
            errors.append(f"{a}+{b}={a+b}, got {r['pred_total']}")
    assert not errors, "Errors: " + "; ".join(errors)
