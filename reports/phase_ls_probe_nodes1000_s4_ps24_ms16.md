# Phase Entanglement-Like Ablation (Classical Proxy)

Claim boundary: proxy metrics on classical phase dynamics (not quantum entanglement).

## Aggregate Stats

| metric | mean | std | ci95 | p-value vs baseline |
|---|---:|---:|---:|---:|
| pairwise_correlation | 0.341546 | 0.259583 | [0.254109, 0.430508] | 1.233e-19 |
| mutual_info | 2.334375 | 0.517250 | [2.161978, 2.501217] | 2.728e-16 |
| bipartition_entropy | 2.324081 | 0.517295 | [2.151520, 2.490873] | 7.422e-16 |
| chsh_proxy | 0.629291 | 0.341679 | [0.512453, 0.740512] | 4.921e-06 |
| quality | 1.100975 | 0.001716 | [1.100384, 1.101513] | 0.6946 |
| quality_vs_ls_baseline | 1.001262 | 0.000912 | [1.000968, 1.001588] | 3.802e-11 |
| phase_gap | 2538.122174 | 40.391189 | [2524.114575, 2550.838474] | 3.331e-227 |

## Notes
- `chsh_proxy` is CHSH-like classical proxy, not QM Bell test.
- `quality` is phase+local-search cut divided by best simple baseline cut (random-8 vs degree).
- `quality_vs_ls_baseline` compares phase+local-search to local-search baselines (random-LS / degree-LS).
- `phase_gap` is phase cut minus best baseline cut.
