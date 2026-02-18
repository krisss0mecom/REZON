# Phase Entanglement-Like Ablation (Classical Proxy)

Claim boundary: proxy metrics on classical phase dynamics (not quantum entanglement).

## Aggregate Stats

| metric | mean | std | ci95 | p-value vs baseline |
|---|---:|---:|---:|---:|
| pairwise_correlation | 0.334023 | 0.276111 | [0.243634, 0.428154] | 1.766e-173 |
| mutual_info | 2.332882 | 0.535159 | [2.156213, 2.504721] | 4.093e-161 |
| bipartition_entropy | 2.316820 | 0.535851 | [2.139919, 2.488824] | 6.25e-161 |
| chsh_proxy | 0.655159 | 0.290912 | [0.558920, 0.747943] | 2.49e-69 |
| quality | 1.124307 | 0.000729 | [1.124081, 1.124537] | 0.8701 |
| quality_vs_ls_baseline | 1.001538 | 0.000517 | [1.001359, 1.001714] | 8.906e-63 |
| phase_gap | 1281.865954 | 7.012797 | [1279.719205, 1284.128542] | 0 |

## Notes
- `chsh_proxy` is CHSH-like classical proxy, not QM Bell test.
- `quality` is phase+local-search cut divided by best simple baseline cut (random-8 vs degree).
- `quality_vs_ls_baseline` compares phase+local-search to local-search baselines (random-LS / degree-LS).
- `phase_gap` is phase cut minus best baseline cut.
