#!/usr/bin/env python3
"""Tests for PhaseAutomaton (3-state FSM / mod-3 counter)."""
import pytest
from phase_automaton import PhaseAutomaton


def test_mod3_clean_10():
    """10 transitions, no noise → 10/10 correct."""
    fsm = PhaseAutomaton(seed=42, noise_amp=0.0)
    for i in range(10):
        s = fsm.tick()
        assert s == (i + 1) % 3, f"step {i+1}: expected {(i+1)%3}, got {s}"


def test_mod3_clean_50():
    """50 transitions, no noise → 50/50 correct."""
    fsm = PhaseAutomaton(seed=42, noise_amp=0.0)
    for i in range(50):
        s = fsm.tick()
        assert s == (i + 1) % 3, f"step {i+1}: expected {(i+1)%3}, got {s}"


def test_mod3_noisy_50():
    """50 transitions, noise=0.3 → all correct."""
    fsm = PhaseAutomaton(seed=42, noise_amp=0.3)
    for i in range(50):
        s = fsm.tick()
        assert s == (i + 1) % 3, f"noisy step {i+1}: expected {(i+1)%3}, got {s}"


def test_idle_holds_state():
    """After advancing to S2, idle 5000 steps — still S2."""
    fsm = PhaseAutomaton(seed=42, noise_amp=0.05)
    fsm.tick()  # S0 → S1
    fsm.tick()  # S1 → S2
    assert fsm.read_state() == 2
    assert fsm.idle(n_steps=5000) == 2


def test_all_three_states_visited():
    """In 6 transitions, all 3 states must appear."""
    fsm = PhaseAutomaton(seed=0, noise_amp=0.0)
    visited = set()
    for _ in range(6):
        visited.add(fsm.tick())
    assert visited == {0, 1, 2}, f"Not all states visited: {visited}"


def test_full_cycle_returns_to_start():
    """After 3k transitions, state == k % 3 == 0 → back at S0."""
    fsm = PhaseAutomaton(seed=42, noise_amp=0.0)
    for _ in range(3):
        fsm.tick()
    assert fsm.read_state() == 0, f"Expected S0 after 3 ticks, got {fsm.read_state()}"
