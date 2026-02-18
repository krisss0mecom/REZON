# Phase Entanglement-Like Ablation (Classical Proxy)

Claim boundary: proxy metrics on classical phase dynamics (not quantum entanglement).

## Aggregate Stats

| metric | mean | std | ci95 | p-value vs baseline |
|---|---:|---:|---:|---:|
| pairwise_correlation | 0.568690 | 0.251857 | [0.483955, 0.647147] | 1.638e-08 |
| mutual_info | 1.810263 | 0.564330 | [1.629990, 2.001231] | 9.744e-13 |
| bipartition_entropy | 1.717860 | 0.536632 | [1.547266, 1.900808] | 3.455e-11 |
| chsh_proxy | 0.640107 | 0.218628 | [0.564924, 0.714964] | 1.623e-06 |
| quality | 1.298983 | 0.013195 | [1.294455, 1.302990] | 0.3178 |
| quality_vs_ls_baseline | 0.999430 | 0.003977 | [0.998152, 1.000843] | 0.4361 |
| phase_gap | 50.874621 | 1.814744 | [50.261162, 51.427168] | 6.909e-291 |

## Notes
- `chsh_proxy` is CHSH-like classical proxy, not QM Bell test.
- `quality` is phase+local-search cut divided by best simple baseline cut (random-8 vs degree).
- `quality_vs_ls_baseline` compares phase+local-search to local-search baselines (random-LS / degree-LS).
- `phase_gap` is phase cut minus best baseline cut.
