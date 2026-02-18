# Phase Entanglement-Like Ablation (Classical Proxy)

Claim boundary: proxy metrics on classical phase dynamics (not quantum entanglement).

## Aggregate Stats

| metric | mean | std | ci95 | p-value vs baseline |
|---|---:|---:|---:|---:|
| pairwise_correlation | 0.485276 | 0.238680 | [0.403753, 0.560854] | 5.319e-39 |
| mutual_info | 1.978569 | 0.498179 | [1.823663, 2.144193] | 5.99e-66 |
| bipartition_entropy | 1.875505 | 0.474592 | [1.728020, 2.033446] | 5.257e-65 |
| chsh_proxy | 0.598332 | 0.220872 | [0.521608, 0.671358] | 1.293e-15 |
| quality | 1.292849 | 0.006824 | [1.290705, 1.295190] | 0.438 |
| quality_vs_ls_baseline | 1.001882 | 0.002640 | [1.000989, 1.002650] | 1.475e-05 |
| phase_gap | 49.208362 | 0.959843 | [48.912082, 49.535603] | 0 |

## Notes
- `chsh_proxy` is CHSH-like classical proxy, not QM Bell test.
- `quality` is phase+local-search cut divided by best simple baseline cut (random-8 vs degree).
- `quality_vs_ls_baseline` compares phase+local-search to local-search baselines (random-LS / degree-LS).
- `phase_gap` is phase cut minus best baseline cut.
