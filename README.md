# REZON: Coherent-Phase Computing

Lightweight experiments in phase-reservoir dynamics for CNOT-like behavior.

## What This Repo Contains

- `reservoir_phase_cnot_pure.py`
  - pure emergent phase dynamics
  - research proof-of-concept
  - currently unstable for strict 4/4 CNOT
- `cnot_rls.py`
  - phase reservoir + RLS readout
  - stable CNOT truth-table mapping (4/4 in current setup)
- `cnot_variant_audit.py`
  - side-by-side audit: `pure` vs `xor_driven`
- `sweep_pure_cnot.py`
  - parameter sweep for pure mode (multi-seed)
- `cnot_report_table.py`
  - JSON -> markdown table for report output

## Core Constraints

- Anchor frequency is immutable: `200.0 Hz`
- Model is classical phase dynamics (coupling, synchronization, leakage)
- This is quantum-inspired, not a quantum computer

## Install

```bash
python -m pip install -r requirements.txt
```

## Quick Start

### 1) Pure phase mode (research, unstable)

```bash
python3 reservoir_phase_cnot_pure.py \
  --seed 42 \
  --eval-seeds 20 \
  --out-json results/cnot_pure_report.json
```

### 2) Phase reservoir + RLS (stable CNOT mapping)

```bash
python3 cnot_rls.py --seed 42 --out-json results/cnot_rls_report.json
python3 cnot_report_table.py --in-json results/cnot_rls_report.json --out-md results/cnot_rls_table.md
```

### 3) Audit both variants on the same parameters

```bash
python3 cnot_variant_audit.py --eval-seeds 50 --out-json results/cnot_variant_audit.json
```

## Tests

```bash
pytest -q test_reservoir_phase_cnot.py test_cnot_rls.py
```

## Current Findings

- Pure mode can show phase-coherent conditional behavior, but is not reliably 4/4 across seeds.
- RLS readout on top of phase reservoir gives stable, reproducible 4/4 CNOT truth-table output.
- Practical takeaway: use pure mode for dynamics research, RLS mode for stable execution.

## Output Files

- `results/cnot_pure_report.json`
- `results/cnot_rls_report.json`
- `results/cnot_rls_table.md`
- `results/cnot_variant_audit.json`
- `results/pure_cnot_sweep.json`
