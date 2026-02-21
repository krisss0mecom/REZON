# Paper Readiness Checklist

Use this before submission.

## Theory

- [x] Explicit model definition (`TURING_FORMALISM.md`)
- [x] Constructive theorem statement with assumptions
- [x] Explicit non-claims section
- [x] Full formal appendix with step-by-step lemmas and proof details
- [ ] Proof review by independent co-author

## Experiments

- [x] Unit tests for gate correctness and memory/FSM behavior
- [x] End-to-end memory+logic+loop demonstration
- [x] Fixed-seed reproducibility protocol (`REPRODUCIBILITY.md`)
- [ ] Multi-platform replication (at least 2 hardware/OS setups)
- [x] Confidence intervals and sensitivity sweeps for new memory/FSM modules
- [x] Adversarial/noise stress tests with failure envelopes

## Claims and Writing

- [x] Claim language constrained to constructive universality
- [x] Separation of empirical evidence vs theorem assumptions
- [x] Submission-ready manuscript structure (Intro/Related/Methods/Theory/Results/Limitations)
- [x] Statistical power justification for all key experiments
- [x] Reviewer checklist (threats to validity, ablations, negative results)

## Artifact Quality

- [x] CI executes current test suite and smoke repro
- [x] Reports stored under `reports/`
- [ ] Tagged release with immutable artifact snapshot
- [ ] DOI archive (Zenodo/OSF) with exact commit hash and reports
