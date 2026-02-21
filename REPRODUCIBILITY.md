# Reproducibility Protocol

This protocol defines the minimum set of commands needed to reproduce the core
claims in this repository.

## Environment

- Python 3.10+
- `pip install -r requirements.txt`
- CPU-only is sufficient

## Determinism

- All evaluation scripts expose fixed seeds.
- Reports include parameters and seed counts used for each run.

## Core Claim Reproduction

1. Run unit tests:

```bash
pytest -q
```

Expected: all tests pass.

2. Reproduce CNOT report:

```bash
python cnot_phase_gate.py --eval-seeds 200 --noise-sweep --out-json reports/cnot_phase_gate_report.json
```

Expected:
- `single_seed_score = 4`
- `multi_seed_clean.pass4_rate = 1.0`

3. Reproduce universal gates report:

```bash
python phase_gate_universal.py --out-json reports/phase_gate_universal_report.json
```

Expected:
- `NOT, AND, OR, XOR, NAND, NOR` all full-score
- `half_adder` full-score

4. Reproduce memory+logic+loop demonstration:

```bash
python phase_turing_demo.py
```

Expected:
- all demo cases correct
- output report at `reports/phase_turing_demo_report.json`

5. Reproduce memory/FSM robustness with confidence intervals:

```bash
python bench/memory_fsm_robustness.py --seeds 12 --out-json reports/memory_fsm_robustness.json --out-md reports/memory_fsm_robustness.md
```

Expected:
- pass-rate summaries with 95% Wilson intervals
- failure envelope section for latch/register/automaton/turing_demo

## Claim Scope

Interpret results under assumptions and non-claims in `TURING_FORMALISM.md`.
In particular, the claim is constructive universality under standard unbounded
memory assumptions, not a quantum or complexity-advantage claim.
