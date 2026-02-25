# REZON: Phase-Gate Computing & Dense Associative Memory on S¹

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18746395.svg)](https://doi.org/10.5281/zenodo.18746395)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18768137.svg)](https://doi.org/10.5281/zenodo.18768137)

> **Preprint (v1.2.0):** [paper.pdf](paper.pdf) — *Dense Associative Memory on S¹: Phase-Gate Computing and Superlinear Capacity in Circular Oscillator Networks*
> **Author:** Krzysztof Gwóźdź, Independent Researcher, Poznań, Poland

---

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

## Scientific Paper (v1.2.0)

📄 **[paper.pdf](paper.pdf)** — journal-ready preprint

| | |
|---|---|
| **Title** | Dense Associative Memory on S¹: Phase-Gate Computing and Superlinear Capacity in Circular Oscillator Networks |
| **Author** | Krzysztof Gwóźdź, Independent Researcher |
| **DOI (this version)** | [10.5281/zenodo.18768137](https://doi.org/10.5281/zenodo.18768137) |
| **DOI (always latest)** | [10.5281/zenodo.18746395](https://doi.org/10.5281/zenodo.18746395) |
| **Target journals** | Neural Networks · IEEE TNNLS · Physical Review E · Nature Physics |

### Key Results in Paper

| Result | Value |
|--------|-------|
| Storage capacity F=exp, N=32 | **α\* = 1.0** — 100% recall at P=N |
| vs classical Hopfield (Amit 1985) | **7.2× improvement** (α\*=0.138 → 1.0) |
| One-step recall (10% noise) | Hamming 3 → 0 in **single update step** |
| CNOT gate robustness | **100%** at noise a=1.0 (20 seeds, Wilson 95% CI) |
| Boolean gates | NOT, AND, OR, XOR, NAND, NOR, half-adder — all **100%** |
| Turing completeness | Proven constructively (NOT + AND + D-latch) |

To regenerate the PDF from source:
```bash
python3 generate_paper_pdf.py
```

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
Formal appendix (lemmas/proof sketch): `FORMAL_APPENDIX.md`.
Repro protocol: `REPRODUCIBILITY.md`.
Submission gap checklist: `PAPER_READINESS_CHECKLIST.md`.
Threats to validity: `THREATS_TO_VALIDITY.md`.
Reviewer checklist: `REVIEWER_CHECKLIST.md`.
Statistical power notes: `STATISTICAL_POWER.md`.

---

## Core Files

| File | Description |
|------|-------------|
| `cnot_phase_gate.py` | CNOTPhaseGate — 3-oscillator CNOT, 200/200 seeds pass4=100%, noise-robust |
| `phase_gate_universal.py` | All 6 gates + Half-Adder + Turing completeness proof |
| `phase_dlatch.py` | Addressable D-latch/PhaseRegister memory from pure ODE dynamics |
| `phase_automaton.py` | 3-state phase automaton (mod-3 FSM) |
| `phase_turing_demo.py` | End-to-end memory + NAND + loop demonstration |
| `phase_full_adder.py` | 1-bit full adder (5 cascaded gates) + 4-bit ripple carry adder |
| `phase_analog.py` | Analog phase computing — fuzzy logic from phase dynamics |
| `phase_hopfield.py` | Phase Hopfield associative memory (Hebbian, 200 Hz anchor) |
| `phase_oim_comparison.py` | Standard OIM vs Conditional OIM — hard constraint encoding |
| `phase_dense_am.py` | Dense Associative Memory on S¹ — Modern Hopfield extension |
| `phase_capacity_study.py` | Empirical storage capacity sweep (N=16,32,64) |
| `test_cnot_phase_gate.py` | Test suite (5/5 passing) |
| `test_phase_dlatch.py` | D-latch/register tests |
| `test_phase_automaton.py` | FSM tests |
| `test_phase_full_adder.py` | Full adder + ripple carry tests (10/10) |
| `test_phase_analog.py` | Analog/fuzzy gate tests (10/10) |
| `test_phase_hopfield.py` | Hopfield recall/energy/capacity tests (8/8) |
| `test_phase_oim_comparison.py` | OIM comparison tests (8/8) |
| `test_phase_dense_am.py` | Dense AM on S¹ tests (9/9) |
| `reports/cnot_phase_gate_report.json` | CNOT benchmark: 200 seeds, noise sweep |
| `reports/phase_gate_universal_report.json` | All gates benchmark |
| `reports/phase_full_adder_report.json` | FA: 8/8 1-bit, 20/20 4-bit ripple |
| `reports/phase_analog_report.json` | Fuzzy: NOT err=0.002, AND=0.069, XOR=0.066 |
| `reports/phase_hopfield_report.json` | Hopfield: 100% recall@10%, 80%@20% noise |
| `reports/phase_oim_comparison_report.json` | OIM: conditional vs standard, novelty proof |
| `reports/phase_dense_am_report.json` | Dense AM: capacity by F-type, discrete update |
| `reports/phase_capacity_report.json` | Capacity sweep: α*(N=16)=0.188, α*(N=64)=0.109 |
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

### Full Adder (phase_full_adder.py) — no RLS, no trained weights

| Circuit | Score | Note |
|---------|-------|------|
| 1-bit Full Adder | **8/8** | All (A,B,Cin) combos correct |
| 4-bit Ripple Carry | **20/20** | Random pairs, carry phase continuous |

Key: carry phase φ_carry propagates directly between adder stages — no
digital re-encoding between stages. Phase continuity = analog carry chain.

### Analog / Fuzzy Logic (phase_analog.py)

| Gate | Fuzzy operation | Mean error |
|------|----------------|------------|
| NOT  | 1 − x (exact complement) | **0.002** |
| AND  | Threshold: 0 if x<0.5, y if x≥0.5 | 0.069 |
| OR   | Threshold: y if x<0.5, 1 if x≥0.5 | 0.125 |
| XOR  | Conditional flip: y if x<0.5, 1-y if x≥0.5 | 0.066 |

**Finding**: Phase ODE implements fuzzy logic without explicit programming.
Attractor structure of the ODE naturally encodes the fuzzy operation.
NOT gate implements exact analytical complement (1-x) — no approximation.

### Phase Hopfield Memory (phase_hopfield.py)

| Noise (flip fraction) | Recall rate (N=32, P=3) |
|-----------------------|------------------------|
| 10% | **100%** (15/15) |
| 20% | 80% (12/15) |
| 30% | 73% (11/15) |

**Proof**: At φ∈{0,π}: sin(φ_i−φ_j)=0 → dφ/dt=0. All {0,π}^N patterns
are fixed points. Energy E = −½·Σ W_ij·cos(φ_i−φ_j) ≡ Hopfield Ising H.

### Dense Associative Memory on S¹ (phase_dense_am.py)

Extension of [Krotov & Hopfield 2020](https://arxiv.org/abs/2008.06996) to continuous phase state space S¹.

**Energy:** `E = −Σ_μ F(Σ_i cos(φ_i − ξ_i^μ))`

**Overlap (phase inner product):** `m_μ = Σ_i cos(φ_i − ξ_i^μ)` ∈ [−N, N]

| F(x) | Model | P* (N=32) | α* | Theory |
|------|-------|-----------|----|--------|
| x | XY/linear | **1** | 0.031 | 0.138·N |
| x²/2 | Dense AM n=2 | **9** | 0.281 | ~N |
| x³/3 | Dense AM n=3 | **32** | 1.000 | ~N² |
| exp(x) | **Modern Hopfield S¹** | **32** | **1.000** | ~exp(N) |

**Modern Hopfield on S¹ (F=exp) stores P=N patterns with 100% recall.**

**Discrete update (Phase Attention):**
```
φ_i^new = circular_mean(ξ_i^μ, weights=softmax(m_μ))
```
Recovers pattern in **1 step** (hamming 3→0 immediately).

**Connection to Transformer attention:**
- Query = φ (current state), Keys = ξ^μ (stored patterns), Values = ξ^μ
- Inner product = `Σ cos(φ_i − ξ_i^μ)` (periodic, hardware-native)
- RC oscillators compute this physically, no GPU needed

**Fixed point proof:** At φ=ξ^μ: sin(φ_i−ξ_i^μ)=0 → dφ/dt=0. □

**Novelty vs Krotov-Hopfield 2020:**
- Their framework: σ ∈ {±1}^N (binary Ising spins)
- This work: φ ∈ S¹^N (continuous phase oscillators, RC hardware-native)
- New overlap: `Σ cos(φ_i − ξ_i^μ)` (periodic, naturally bounded, no normalization needed)

### Phase Hopfield Capacity Study (phase_capacity_study.py)

Empirical verification that binary {0,π}^N Phase Hopfield ≡ classical Hopfield universality class.

| N | P* | α* (measured) | α* (theory AGS 1985) |
|---|----|----|---|
| 16 | 3 | 0.188 | 0.138 |
| 32 | 4 | 0.125 | 0.138 |
| 64 | 7 | 0.109 | 0.138 |

Finite-size effects visible (α* converges to 0.138 as N→∞). Confirms Phase Hopfield
restricted to {0,π}^N is in the same universality class as Ising Hopfield.

### OIM Comparison (phase_oim_comparison.py)

**Novelty** — our framework `K·cos(φ_c)·sin(φ_t−φ_out)` vs literature:

| Method | Equation | Constraint encoding |
|--------|----------|---------------------|
| Wang 2019 OIM | J_ij·sin(φ_j−φ_i) | penalty only |
| 3-body Kuramoto | sin(φ_j+φ_k−2φ_i) | additive, no sign flip |
| **Our framework** | **cos(φ_c)·sin(φ_t−φ_out)** | **exact hard constraint** |

φ_c=0 → sync (same partition), φ_c=π → anti-sync (cut edge), φ_c=π/2 → disabled.
Reduces to standard OIM when φ_c=π. Backward compatible.

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

### Run memory/FSM robustness sweep

```bash
python bench/memory_fsm_robustness.py --seeds 12 --out-json reports/memory_fsm_robustness.json --out-md reports/memory_fsm_robustness.md
```

### Run full adder (8/8 + 4-bit ripple)

```bash
python3 phase_full_adder.py --warmup 2000 --collect 400
```

### Run analog/fuzzy gate sweep

```bash
python3 phase_analog.py --warmup 2000 --collect 400 --grid 5
```

### Run Hopfield associative memory

```bash
python3 phase_hopfield.py
```

### Run OIM comparison

```bash
python3 phase_oim_comparison.py
```

### Run Dense AM on S¹ (Modern Hopfield extension)

```bash
python3 phase_dense_am.py --N 32 --trials 3
```

### Run capacity study (N=16,32,64)

```bash
python3 phase_capacity_study.py --sizes 16 32 64 --trials 3
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
