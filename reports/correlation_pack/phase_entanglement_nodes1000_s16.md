# Phase Entanglement-Like Ablation (Classical Proxy)

Claim boundary: proxy metrics on classical phase dynamics (not quantum entanglement).

## Aggregate Stats

| metric | mean | std | ci95 | p-value vs baseline |
|---|---:|---:|---:|---:|
| pairwise_correlation | 0.332605 | 0.276675 | [0.241083, 0.427067] | 1.893e-95 |
| mutual_info | 2.342686 | 0.539968 | [2.164457, 2.517378] | 9.44e-79 |
| bipartition_entropy | 2.332683 | 0.540384 | [2.154289, 2.507445] | 9.802e-79 |
| chsh_proxy | 0.645165 | 0.311423 | [0.540013, 0.745097] | 2.94e-35 |
| quality | 1.101113 | 0.000763 | [1.100879, 1.101369] | 0.2621 |
| quality_vs_ls_baseline | 1.001390 | 0.000474 | [1.001247, 1.001546] | 2.848e-57 |
| phase_gap | 2538.462074 | 17.933679 | [2532.886020, 2544.510082] | 0 |

## Notes
- `chsh_proxy` is CHSH-like classical proxy, not QM Bell test.
- `quality` is phase+local-search cut divided by best simple baseline cut (random-8 vs degree).
- `quality_vs_ls_baseline` compares phase+local-search to local-search baselines (random-LS / degree-LS).
- `phase_gap` is phase cut minus best baseline cut.
