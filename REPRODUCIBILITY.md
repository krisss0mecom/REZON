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

## Claim Scope

Interpret results under assumptions and non-claims in `TURING_FORMALISM.md`.
In particular, the claim is constructive universality under standard unbounded
memory assumptions, not a quantum or complexity-advantage claim.
