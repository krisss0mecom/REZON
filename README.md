# REZON: Phase-Gate Computing

Classical phase-oscillator logic gates — **no RLS, no learned weights, no machine learning.**

Logic emerges from pure oscillator dynamics.

---

## What This Is

A **phase-gate framework** where every logic gate is a single differential equation:

```
dφ_out = K · f(φ_control) · sin(φ_target − φ_out) + bias(φ_out)
```

By choosing `f(φ_c)`, you get different gates:

| f(φ_c)              | Gate      | Effect                                     |
|---------------------|-----------|--------------------------------------------|
| `cos(φ_c)`          | XOR/CNOT  | sync when c=0, anti-sync when c=1          |
| `+1`                | WIRE      | always synchronize (copy)                  |
| `−1`                | NOT       | always anti-synchronize (invert)           |
| `(1−cos(φ_c))/2`   | AND-like  | conditional coupling (quadratic)           |
| `(1+cos(φ_c))/2`   | OR-like   | threshold coupling                         |

Full set implemented and verified: **NOT, AND, OR, XOR, NAND, NOR, Half-Adder**.

---

## Key Result: Constructive Universality (Turing-Complete Under Standard Assumptions)

```
1. {NOT, AND} ⊆ framework  →  functional completeness (any boolean function) ✓
2. Bistable oscillator holds state ∈ {0, π} without external support          ✓
3. φ_out feeds back as φ_in of next gate                                       ✓
∴  Framework is computationally universal (Turing complete) under standard
   unbounded-memory assumptions                                                 □
```

This is a **classical phase computer** in the mathematical sense — not quantum, not quantum-inspired. Logic as attractor dynamics, not CMOS boolean algebra.

Formal claim scope and assumptions: `TURING_FORMALISM.md`.
Repro protocol: `REPRODUCIBILITY.md`.
Submission gap checklist: `PAPER_READINESS_CHECKLIST.md`.

---

## Core Files

| File | Description |
|------|-------------|
| `cnot_phase_gate.py` | CNOTPhaseGate — 3-oscillator CNOT, 200/200 seeds pass4=100%, noise-robust |
| `phase_gate_universal.py` | All 6 gates + Half-Adder + Turing completeness proof |
| `phase_dlatch.py` | Addressable D-latch/PhaseRegister memory from pure ODE dynamics |
| `phase_automaton.py` | 3-state phase automaton (mod-3 FSM) |
| `phase_turing_demo.py` | End-to-end memory + NAND + loop demonstration |
| `test_cnot_phase_gate.py` | Test suite (5/5 passing) |
| `test_phase_dlatch.py` | D-latch/register tests |
| `test_phase_automaton.py` | FSM tests |
| `reports/cnot_phase_gate_report.json` | CNOT benchmark: 200 seeds, noise sweep |
| `reports/phase_gate_universal_report.json` | All gates benchmark |
| `legacy/` | Old RLS/pure-mode experiments (kept for reference) |

---

## CNOT Gate: The Core Equation

```
dφ_out/dt = ω + anchor + K_cnot · cos(φ_c) · sin(φ_t − φ_out)
```

**Analytical proof:**

Fixed points: `φ_out ∈ {φ_t, φ_t + π}`

Stability: `d/dφ_out[cos(φ_c) · sin(φ_t − φ_out)] = −cos(φ_c) · cos(φ_t − φ_out)`

- `control=0`: `cos(φ_c) ≈ +1` → φ_out=φ_t STABLE, φ_out=φ_t+π UNSTABLE → **preserve target**
- `control=1`: `cos(φ_c) ≈ −1` → φ_out=φ_t UNSTABLE, φ_out=φ_t+π STABLE → **flip target**

Architecture: 3 oscillators — φ_c (control, strong injection), φ_t (target, strong injection), φ_out (free, CNOT-coupled).
Readout: `mean(cos(φ_out)) > 0 → bit=0`, else `bit=1`. **No RLS. No learned weights.**

---

## Results

### CNOT (200 seeds, noise sweep)

| Method       | pass4/100 seeds | Note                         |
|--------------|-----------------|------------------------------|
| Old pure mode | 0%             | unstable, seed-dependent     |
| cnot_rls.py  | 100%            | has RLS readout (legacy)     |
| **CNOTPhaseGate** | **100%**   | **pure, no RLS**            |
| noise=1.0    | **100%**        | robust under heavy noise     |

### All Gates (phase_gate_universal.py)

| Gate | Truth table | Result |
|------|-------------|--------|
| NOT  | 2 rows      | 2/2 ✓  |
| AND  | 4 rows      | 4/4 ✓  |
| OR   | 4 rows      | 4/4 ✓  |
| XOR  | 4 rows      | 4/4 ✓  |
| NAND | 4 rows      | 4/4 ✓  |
| NOR  | 4 rows      | 4/4 ✓  |
| Half-Adder | 4 rows | 4/4 ✓ |

---

## Quick Start

```bash
pip install -r requirements.txt
```

### Run CNOT gate

```bash
python3 cnot_phase_gate.py
```

### Run all gates

```bash
python3 phase_gate_universal.py
```

### Run tests

```bash
pytest -q
```

---

## Constraints

- Anchor frequency: **200.0 Hz** (immutable — hardware constraint)
- No machine learning, no RLS, no learned weights
- Classical physics only (Kuramoto-type coupling)
- Not a quantum computer — classical phase computer

---

## Applications

This framework opens paths in:

1. **Neuromorphic computing** — oscillator chips replacing CMOS transistors; computes by reaching dynamic equilibrium, not clock edges
2. **Conditional Ising Machines** — `K·cos(φ_k)·sin(φ_j−φ_i)` enables constraint encoding for SAT/CSP/QUBO
3. **Neuroscience models** — theta-gamma coupling in hippocampus matches `K·f(φ_theta)·sin(φ_gamma_in−φ_gamma_out)` exactly
4. **Photonic logic** — `cos(φ_c)` maps to polarization modulation; all-optical logic without electronics
5. **Phase-stream ciphers** — CNOT is its own inverse; oscillator chains as stream ciphers
6. **RC substrate** — phase gates as logic layer in physical Reservoir Computing architectures

---

## Legacy

Files in `legacy/` are old RLS-based and pure-mode experiments kept for historical reference:
- `cnot_rls.py` — RLS readout baseline (4/4 but requires trained weights)
- `reservoir_phase_cnot_pure.py` — original pure attempt (pass4=0%, unstable)
- `cnot_variant_audit.py`, `sweep_pure_cnot.py`, etc.

Results from those experiments are in `legacy/` as well.

---

## Hardware

Physical setup (target): `Jetson Orin Nano → AD9850 (anchor 200 Hz) → PCB RC → AD7606 → Jetson`

Current limit: 8 analog inputs → max 80 oscillators simultaneously (10/board, 8 boards).
Time-multiplexing possible for more boards.

Details: `hardware/HARDWARE_PROTOCOL.md`
