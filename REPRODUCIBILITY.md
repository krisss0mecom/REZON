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

6. Reproduce Dense AM capacity sweep — N=32 (fast, ~10 min):

```bash
python phase_dense_am.py --N 32 --trials 3 --out-json reports/phase_dense_am_N32_report.json
```

Expected:
- `F=exp: P*=32, α*=1.000`
- `F=poly3: P*=32, α*=1.000` (94.8% trials)
- `F=poly2: P*=9, α*=0.281`
- `F=linear: P*=1, α*=0.031`

7. Reproduce Dense AM capacity sweep — N=64 (~97 min):

```bash
python phase_dense_am.py --N 64 --trials 3 --out-json reports/phase_dense_am_N64_report.json
```

Expected:
- `F=exp: P*=64, α*=1.000`
- `F=poly2: P*=12, α*=0.188`
- `F=poly3: P*=6, α*=0.094` (Euler-unstable at higher P)
- `F=linear: P*=1, α*=0.016`

8. Reproduce Dense AM capacity sweep — N=128 (long, ~8.3 h):

```bash
python phase_dense_am.py --N 128 --trials 3 --out-json reports/phase_dense_am_N128_report.json
```

Expected (verified, elapsed=29942s):
- `F=exp: P*=128, α*=1.000` (384/384 trials, 100% success at every P)
- `F=poly2: P*=20, α*=0.156` (graceful degradation: 0.97 at P=12, 0.57 at P=20)
- `F=poly3: P*=0, α*=0.000` (Euler-unstable: Δt·N²=16.4 >> 1)
- `F=linear: P*=1, α*=0.008`
- Discrete update (Phase Attention): Hamming 12→0 in 1 step at N=128, P=5

Pre-computed results are stored in `reports/phase_dense_am_N128_report.json`.

## Claim Scope

Interpret results under assumptions and non-claims in `TURING_FORMALISM.md`.
In particular, the claim is constructive universality under standard unbounded
memory assumptions, not a quantum or complexity-advantage claim.
