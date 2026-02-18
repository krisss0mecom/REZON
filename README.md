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

### 4) Proof benchmark pack (multi-seed)

```bash
python3 benchmark_rezon_proof.py \
  --eval-seeds 8 \
  --warmup 1200 \
  --steps 600 \
  --train-steps 700 \
  --out-json reports/rezon_proof_benchmark.json \
  --out-md reports/rezon_proof_benchmark.md
```

### 5) QC-like decryption benchmark (key search)

```bash
python3 benchmark_qc_like_decrypt.py \
  --bits 8,10,12,14,16,18,20,22,24 \
  --known-len 10 \
  --out-json reports/qc_like_decrypt_benchmark.json \
  --out-md reports/qc_like_decrypt_benchmark.md
```

### 6) Phase-guided candidate ordering vs brute-force

```bash
python3 benchmark_qc_like_phase_guided.py \
  --bits 8,10,12,14,16 \
  --known-len 10 \
  --precheck-len 4 \
  --out-json reports/qc_like_phase_guided_benchmark.json \
  --out-md reports/qc_like_phase_guided_benchmark.md
```

## Tests

```bash
pytest -q test_reservoir_phase_cnot.py test_cnot_rls.py
```

## Current Findings

- Pure mode can show phase-coherent conditional behavior, but is not reliably 4/4 across seeds.
- RLS readout on top of phase reservoir gives stable, reproducible 4/4 CNOT truth-table output.
- Practical takeaway: use pure mode for dynamics research, RLS mode for stable execution.
- Latest proof benchmark confirms this trend:
  - `pure`: low 4/4 reliability across seeds,
  - `xor_driven` and `RLS`: substantially higher stability on the same truth-table task.
- QC-like decryption benchmark shows classical brute-force scaling with keyspace size (`O(N)`)
  and provides a Grover-query reference (`sqrt(N)`) as a theoretical lower-bound comparator.
- Phase-guided ordering can reduce average query count, but may still be slower wall-clock
  due to ranking overhead (important distinction: attempts vs time).

## Output Files

- `results/cnot_pure_report.json`
- `results/cnot_rls_report.json`
- `results/cnot_rls_table.md`
- `results/cnot_variant_audit.json`
- `results/pure_cnot_sweep.json`
- `reports/rezon_proof_benchmark.json`
- `reports/rezon_proof_benchmark.md`
- `reports/qc_like_decrypt_benchmark.json`
- `reports/qc_like_decrypt_benchmark.md`
- `reports/qc_like_phase_guided_benchmark.json`
- `reports/qc_like_phase_guided_benchmark.md`
