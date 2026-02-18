# Phase Entanglement-Like Ablation (Classical Proxy)

Claim boundary: proxy metrics on classical phase dynamics (not quantum entanglement).

## Aggregate Stats

| metric | mean | std | ci95 | p-value vs baseline |
|---|---:|---:|---:|---:|
| pairwise_correlation | 0.337252 | 0.278675 | [0.244702, 0.432956] | 1.619e-45 |
| mutual_info | 2.331576 | 0.550378 | [2.148510, 2.509674] | 6.621e-38 |
| bipartition_entropy | 2.321211 | 0.550744 | [2.138140, 2.499276] | 1.444e-37 |
| chsh_proxy | 0.650081 | 0.333017 | [0.537786, 0.756130] | 2.314e-15 |
| quality | 1.100864 | 0.000989 | [1.100532, 1.101170] | 0.5388 |
| quality_vs_ls_baseline | 1.001081 | 0.000704 | [1.000850, 1.001332] | 1.007e-17 |
| phase_gap | 2530.850739 | 23.576313 | [2522.813076, 2538.054449] | 0 |

## Notes
- `chsh_proxy` is CHSH-like classical proxy, not QM Bell test.
- `quality` is phase+local-search cut divided by best simple baseline cut (random-8 vs degree).
- `quality_vs_ls_baseline` compares phase+local-search to local-search baselines (random-LS / degree-LS).
- `phase_gap` is phase cut minus best baseline cut.
