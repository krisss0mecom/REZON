# Threats to Validity

## Internal Validity

- Numerical integration artifacts:
  - Results depend on `dt`, warmup, and collection horizons.
  - Mitigation: sensitivity sweeps and fixed defaults in reports.
- Readout threshold dependence:
  - Binary decision uses sign of `cos(phi)`.
  - Mitigation: report attractor margins (`|mean_cos|`) where possible.

## Construct Validity

- "Turing complete" is constructive under assumptions, not a claim of practical
  infinite memory hardware.
- "Entanglement-like" metrics are classical proxies, not quantum entanglement.

## Statistical Conclusion Validity

- Some results are finite-seed estimates.
- Mitigation: Wilson CI for pass rates and explicit failure envelopes in
  `reports/memory_fsm_robustness.*`.

## External Validity

- Current evidence is software simulation first.
- Hardware transfer (noise, drift, ADC quantization, coupling mismatch) is not
  fully characterized yet.

## Reproducibility Risk

- Risk: stale scripts or outdated CI.
- Mitigation: CI uses current tests and smoke reproductions; all outputs target
  `reports/`.
