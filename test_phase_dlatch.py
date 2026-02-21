#!/usr/bin/env python3
"""Tests for PhaseDLatch and PhaseRegister."""
import pytest
from phase_dlatch import PhaseDLatch, PhaseRegister


def test_write_0_and_hold():
    latch = PhaseDLatch(seed=42, noise_amp=0.05)
    latch.write(0, warmup=3000)
    assert latch.hold(5000) == 0


def test_write_1_and_hold():
    latch = PhaseDLatch(seed=7, noise_amp=0.05)
    latch.write(1, warmup=3000)
    assert latch.hold(5000) == 1


def test_alternating_write():
    """Write 0→1→0→1→0, each should hold 5000 steps."""
    latch = PhaseDLatch(seed=99, noise_amp=0.05)
    for bit in [0, 1, 0, 1, 0]:
        latch.write(bit, warmup=3000)
        held = latch.hold(n_steps=5000)
        assert held == bit, f"Expected {bit}, got {held}"


def test_hold_under_heavy_noise():
    """State must hold even under noise_amp=1.0."""
    latch = PhaseDLatch(seed=42, noise_amp=1.0)
    latch.write(1, warmup=4000)
    assert latch.hold(n_steps=8000) == 1


def test_hold_long_duration():
    """State must hold for 20,000 steps (noise=0.05)."""
    latch = PhaseDLatch(seed=42, noise_amp=0.05)
    latch.write(0, warmup=3000)
    assert latch.hold(n_steps=20_000) == 0


def test_phase_register_8bit():
    """PhaseRegister: write 8 bits, read back all correct."""
    reg = PhaseRegister(8, seed=42, noise_amp=0.05)
    pattern = [1, 0, 1, 1, 0, 1, 0, 0]
    reg.write_all(pattern)
    read_back = reg.read_all(hold_steps=3000)
    assert read_back == pattern, f"Expected {pattern}, got {read_back}"


def test_phase_register_independent():
    """Writing to one address doesn't affect others."""
    reg = PhaseRegister(4, seed=0, noise_amp=0.0)
    reg.write_all([0, 0, 0, 0])
    reg.write(0, 1, warmup=3000)
    results = reg.read_all(hold_steps=2000)
    assert results[0] == 1
    assert results[1] == 0
    assert results[2] == 0
    assert results[3] == 0


def test_anchor_immutable():
    from phase_gate_universal import ensure_anchor_immutable
    with pytest.raises(ValueError, match="200"):
        ensure_anchor_immutable(100.0)
