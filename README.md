# Phase Entanglement RC

Phase-reservoir computing experiments focused on coherent phase states and CNOT-like mapping.

## Scope

This repo is intentionally minimal and public-safe. It contains:
- `reservoir_phase_cnot_pure.py` - **pure emergent PoC** (unstable, typically ~1-2/4)
- `cnot_rls.py` - **RC + RLS readout** (stable CNOT truth-table mapping)
- tests and reporting utilities

## Physics Constraints

- Anchor frequency is immutable: `200.0 Hz`
- Core mechanism is physical phase dynamics (coupling + synchronization), not transformer attention

## Install

```bash
python -m pip install -r requirements.txt
```

## Run

### 1) Pure phase CNOT PoC (unstable)

```bash
python3 reservoir_phase_cnot_pure.py --seed 42 --eval-seeds 20 --out-json results/cnot_pure_report.json
```

Interpretation:
- proof-of-concept only
- usually `~1-2/4` cases correct across seeds (current reference run: mean 1.5/4)

### 2) RC + RLS CNOT (stable)

```bash
python3 cnot_rls.py --seed 42 --out-json results/cnot_rls_report.json
python3 cnot_report_table.py --in-json results/cnot_rls_report.json --out-md results/cnot_rls_table.md
```

Expected:
- `score=4/4`
- markdown table in `results/cnot_rls_table.md`

## Tests

```bash
pytest -q test_cnot_rls.py
```

## Notes

This is a quantum-inspired classical system with continuous phase states, not a quantum computer.

Latest pure-sweep result (`results/pure_cnot_sweep.json`):
- 120 parameter trials, 60 seeds per trial, 600 steps
- best pure configuration reached `pass_rate_4of4 = 0.083` (8.3%), far below 95%
- conclusion: stable full CNOT currently requires readout learning (RLS) in this setup
