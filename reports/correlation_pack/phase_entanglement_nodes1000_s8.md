# Phase Entanglement-Like Ablation (Classical Proxy)

Claim boundary: proxy metrics on classical phase dynamics (not quantum entanglement).

## Aggregate Stats

| metric | mean | std | ci95 | p-value vs baseline |
|---|---:|---:|---:|---:|
| pairwise_correlation | 0.337252 | 0.278675 | [0.244702, 0.432956] | 1.619e-45 |
| mutual_info | 2.331576 | 0.550378 | [2.148510, 2.509674] | 6.621e-38 |
| bipartition_entropy | 2.321211 | 0.550744 | [2.138140, 2.499276] | 1.444e-37 |
| chsh_proxy | 0.650081 | 0.333017 | [0.537786, 0.756130] | 2.314e-15 |
| quality | 1.101188 | 0.000846 | [1.100928, 1.101469] | 0.122 |
| quality_vs_ls_baseline | 1.001377 | 0.000630 | [1.001175, 1.001570] | 1.961e-27 |
| phase_gap | 2539.008958 | 20.033066 | [2532.864030, 2545.597562] | 0 |

## Notes
- `chsh_proxy` is CHSH-like classical proxy, not QM Bell test.
- `quality` is phase+local-search cut divided by best simple baseline cut (random-8 vs degree).
- `quality_vs_ls_baseline` compares phase+local-search to local-search baselines (random-LS / degree-LS).
- `phase_gap` is phase cut minus best baseline cut.
