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
- Claim scope: **quantum-inspired analog search heuristic**, not QC replacement

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

### 7) C++ phase-guided benchmark (faster runtime path)

```bash
g++ -O3 -march=native -std=c++17 -fopenmp benchmark_qc_like_phase_guided.cpp -o benchmark_qc_like_phase_guided_cpp
./benchmark_qc_like_phase_guided_cpp \
  --known-len 10 \
  --precheck-len 4 \
  --out-json reports/qc_like_phase_guided_benchmark_cpp.json \
  --out-md reports/qc_like_phase_guided_benchmark_cpp.md
```

### 8) Synthetic encrypted-folder recovery benchmark

```bash
python3 benchmark_encrypted_folder_sim.py \
  --bits 8,10,12,14,16 \
  --reps 3 \
  --top-k 2048 \
  --out-json reports/encrypted_folder_sim_benchmark.json \
  --out-md reports/encrypted_folder_sim_benchmark.md
```

### 9) Autonomous RC key ordering (no user candidate list)

```bash
python3 benchmark_encrypted_folder_autorc.py \
  --bits 8,10,12,14,16 \
  --reps 3 \
  --out-json reports/encrypted_folder_autorc_benchmark.json \
  --out-md reports/encrypted_folder_autorc_benchmark.md
```

### 10) Standard instances suite (QUBO / Max-Cut / SAT + ablation + stats)

```bash
python3 bench/standard_suite.py \
  --instances 6 \
  --seeds 6 \
  --rc-nodes 1000 \
  --rc-warmup-steps 40 \
  --out-json reports/standard_suite_report.json \
  --out-md reports/standard_suite_report.md
```

### 11) Top-K C++ ranking (no full sort)

```bash
python3 bench/topk_ranker_wrapper.py --keys 65536 --k 1024 --out reports/topk_ranker_smoke.txt
```

### 11b) Top-K CUDA (CUB/Thrust fallback)

```bash
python3 bench/topk_cuda_wrapper.py --keys 65536 --k 1024 --out reports/topk_cuda_smoke.txt
```

### 13) Public dataset runner (canonical benchmark sources)

```bash
python3 bench/public_dataset_runner.py --out-json reports/public_dataset_benchmark.json
```

### 14) Two-stage phase-guided ranking (coarse-to-fine)

```bash
python3 bench/phase_guided_two_stage.py \
  --bits 8,10,12,14,16 \
  --reps 3 \
  --shortlist-ratio 0.02 \
  --out-json reports/two_stage_phase_guided.json \
  --out-md reports/two_stage_phase_guided.md
```

### 12) Hardware protocol (80 oscillators)

```bash
# Read protocol:
cat hardware/HARDWARE_PROTOCOL.md

# Fill hardware/hw_runs.csv from template and compare:
python3 hardware/compare_hw_sw.py \
  --sw reports/standard_suite_report.json \
  --hw hardware/hw_runs.csv \
  --out reports/hw_sw_comparison.json
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
- C++ implementation of phase-guided ranking significantly reduces overhead vs Python in this repo
  (measured speedups in generated report).
- Synthetic encrypted-folder benchmark confirms the same tradeoff on archive recovery:
  phase-guided ordering can reduce attempts strongly, while full ranking overhead can still dominate wall-clock time.
- Autonomous RC variant (no manual candidate list) also reduces attempts strongly in this synthetic setup,
  but runtime remains slower than brute-force due to full-space scoring overhead.
- Standard suite now supports `1000` oscillators with explicit warmup (`--rc-nodes`, `--rc-warmup-steps`).
- Two-stage ranking (coarse-to-fine) is included to reduce full-ranking overhead.

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
- `reports/qc_like_phase_guided_benchmark_cpp.json`
- `reports/qc_like_phase_guided_benchmark_cpp.md`
- `reports/qc_like_phase_guided_cpp_vs_python.md`
- `reports/encrypted_folder_sim_benchmark.json`
- `reports/encrypted_folder_sim_benchmark.md`
- `reports/encrypted_samples/`
- `reports/encrypted_folder_autorc_benchmark.json`
- `reports/encrypted_folder_autorc_benchmark.md`
- `reports/standard_suite_report.json`
- `reports/standard_suite_report.md`
- `reports/hw_sw_comparison.json`
- `reports/topk_cuda_smoke.txt`
- `reports/cuda_topk_research.md`
- `reports/public_dataset_benchmark.json`
- `reports/two_stage_phase_guided.json`
- `reports/two_stage_phase_guided.md`
