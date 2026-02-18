# Phase Entanglement-Like Ablation (Classical Proxy)

Claim boundary: proxy metrics on classical phase dynamics (not quantum entanglement).

## Aggregate Stats

| metric | mean | std | ci95 | p-value vs baseline |
|---|---:|---:|---:|---:|
| pairwise_correlation | 0.334523 | 0.276055 | [0.244366, 0.428603] | 2.704e-77 |
| mutual_info | 2.334422 | 0.534768 | [2.158730, 2.506551] | 4.726e-81 |
| bipartition_entropy | 2.317861 | 0.536123 | [2.141596, 2.490277] | 2.581e-81 |
| chsh_proxy | 0.647645 | 0.298142 | [0.548215, 0.742558] | 4.573e-33 |
| quality | 1.124215 | 0.000943 | [1.123885, 1.124492] | 0.9 |
| quality_vs_ls_baseline | 1.001696 | 0.000687 | [1.001453, 1.001921] | 1.947e-41 |
| phase_gap | 1283.133034 | 9.025492 | [1279.998481, 1285.783492] | 0 |

## Notes
- `chsh_proxy` is CHSH-like classical proxy, not QM Bell test.
- `quality` is phase+local-search cut divided by best simple baseline cut (random-8 vs degree).
- `quality_vs_ls_baseline` compares phase+local-search to local-search baselines (random-LS / degree-LS).
- `phase_gap` is phase cut minus best baseline cut.
