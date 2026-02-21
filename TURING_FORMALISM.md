# Formal Claim Scope: Turing Completeness

This document specifies the exact claim used in this repository and its assumptions.

## Model

We define a phase-computing machine `M_phase` with:

1. Gate primitives implemented by ODE dynamics (no trained readout):
   - `NOT`, `AND`, `OR`, `XOR`, `NAND`, `NOR`
2. Addressable phase memory:
   - `PhaseDLatch` and `PhaseRegister` (`phase_dlatch.py`)
3. Feedback/composition:
   - outputs can be written back to memory and reused as later inputs
4. Discrete observation rule:
   - bit readout by sign of `cos(phi)` (or nearest attractor for 3-state automaton)

## Theorem (Constructive Scope)

Given:

- G1: `NOT` and `AND` are correctly implemented (truth-table complete),
- G2: memory cell preserves one bit state under hold dynamics,
- G3: memory is addressable and writable/readable in finite time,
- G4: outputs can be fed back as inputs,
- G5: available memory is unbounded (standard TM assumption),

then `M_phase` can simulate arbitrary Boolean circuits with state across arbitrary
time horizons, hence is Turing complete in the standard constructive sense.

## Why this follows

1. By Shannon completeness, `{NOT, AND}` is functionally complete.
2. With addressable state, we can realize sequential logic.
3. Sequential logic + unbounded memory simulates a universal machine.

## What is empirically verified here

- Gate truth tables and compositions:
  - `phase_gate_universal.py`
- CNOT/XOR phase gate robustness:
  - `cnot_phase_gate.py`
- Memory hold/write/read:
  - `phase_dlatch.py`, `test_phase_dlatch.py`
- Finite-state iterative dynamics:
  - `phase_automaton.py`, `test_phase_automaton.py`
- End-to-end memory + NAND + feedback loop:
  - `phase_turing_demo.py`

## Explicit non-claims

- Not a proof of quantum entanglement.
- Not a complexity-theory advantage claim over CMOS/GPUs.
- Not a claim that finite hardware memory is literally unbounded.

The repository claim is constructive universality under the same unbounded-memory
assumption used in classical computability theory.
