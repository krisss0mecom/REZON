# Phase Entanglement-Like Ablation (Classical Proxy)

Claim boundary: proxy metrics on classical phase dynamics (not quantum entanglement).

## Aggregate Stats

| metric | mean | std | ci95 | p-value vs baseline |
|---|---:|---:|---:|---:|
| pairwise_correlation | 0.568690 | 0.251857 | [0.483955, 0.647147] | 1.638e-08 |
| mutual_info | 1.810263 | 0.564330 | [1.629990, 2.001231] | 9.744e-13 |
| bipartition_entropy | 1.717860 | 0.536632 | [1.547266, 1.900808] | 3.455e-11 |
| chsh_proxy | 0.640107 | 0.218628 | [0.564924, 0.714964] | 1.623e-06 |
| quality | 1.292344 | 0.012442 | [1.288267, 1.296496] | 0.7146 |
| quality_vs_ls_baseline | 0.994386 | 0.004211 | [0.993139, 0.995793] | 1.93e-11 |
| phase_gap | 49.785542 | 1.678433 | [49.231726, 50.328959] | 3.027e-288 |

## Notes
- `chsh_proxy` is CHSH-like classical proxy, not QM Bell test.
- `quality` is phase+local-search cut divided by best simple baseline cut (random-8 vs degree).
- `quality_vs_ls_baseline` compares phase+local-search to local-search baselines (random-LS / degree-LS).
- `phase_gap` is phase cut minus best baseline cut.
