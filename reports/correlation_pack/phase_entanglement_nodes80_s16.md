# Phase Entanglement-Like Ablation (Classical Proxy)

Claim boundary: proxy metrics on classical phase dynamics (not quantum entanglement).

## Aggregate Stats

| metric | mean | std | ci95 | p-value vs baseline |
|---|---:|---:|---:|---:|
| pairwise_correlation | 0.478199 | 0.238233 | [0.397043, 0.552197] | 2.352e-16 |
| mutual_info | 1.989518 | 0.500951 | [1.835976, 2.154734] | 1.702e-24 |
| bipartition_entropy | 1.883982 | 0.476995 | [1.738715, 2.041596] | 2.926e-23 |
| chsh_proxy | 0.592818 | 0.202026 | [0.525519, 0.661441] | 1.212e-10 |
| quality | 1.296614 | 0.008379 | [1.293959, 1.299190] | 0.631 |
| quality_vs_ls_baseline | 1.001114 | 0.003605 | [0.999885, 1.002211] | 0.06661 |
| phase_gap | 50.014103 | 1.177452 | [49.634797, 50.364815] | 0 |

## Notes
- `chsh_proxy` is CHSH-like classical proxy, not QM Bell test.
- `quality` is phase+local-search cut divided by best simple baseline cut (random-8 vs degree).
- `quality_vs_ls_baseline` compares phase+local-search to local-search baselines (random-LS / degree-LS).
- `phase_gap` is phase cut minus best baseline cut.
