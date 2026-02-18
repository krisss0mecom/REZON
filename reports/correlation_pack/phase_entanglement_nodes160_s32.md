# Phase Entanglement-Like Ablation (Classical Proxy)

Claim boundary: proxy metrics on classical phase dynamics (not quantum entanglement).

## Aggregate Stats

| metric | mean | std | ci95 | p-value vs baseline |
|---|---:|---:|---:|---:|
| pairwise_correlation | 0.395001 | 0.242358 | [0.313737, 0.474370] | 2.913e-73 |
| mutual_info | 2.179527 | 0.509836 | [2.013441, 2.343320] | 2.324e-110 |
| bipartition_entropy | 2.119640 | 0.504710 | [1.954837, 2.281652] | 4.005e-113 |
| chsh_proxy | 0.658917 | 0.240071 | [0.580992, 0.737464] | 9.269e-17 |
| quality | 1.228198 | 0.003022 | [1.227085, 1.229236] | 0.9758 |
| quality_vs_ls_baseline | 1.004023 | 0.001930 | [1.003356, 1.004610] | 1.432e-47 |
| phase_gap | 149.685097 | 1.740923 | [149.030206, 150.280644] | 0 |

## Notes
- `chsh_proxy` is CHSH-like classical proxy, not QM Bell test.
- `quality` is phase+local-search cut divided by best simple baseline cut (random-8 vs degree).
- `quality_vs_ls_baseline` compares phase+local-search to local-search baselines (random-LS / degree-LS).
- `phase_gap` is phase cut minus best baseline cut.
