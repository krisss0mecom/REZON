# Phase Entanglement-Like Ablation (Classical Proxy)

Claim boundary: proxy metrics on classical phase dynamics (not quantum entanglement).

## Aggregate Stats

| metric | mean | std | ci95 | p-value vs baseline |
|---|---:|---:|---:|---:|
| pairwise_correlation | 0.340544 | 0.271907 | [0.250981, 0.434179] | 2.366e-36 |
| mutual_info | 2.330171 | 0.523206 | [2.157891, 2.499993] | 8.517e-42 |
| bipartition_entropy | 2.312705 | 0.524361 | [2.139897, 2.482522] | 3.391e-42 |
| chsh_proxy | 0.641282 | 0.290292 | [0.543770, 0.733880] | 2.291e-13 |
| quality | 1.122638 | 0.001511 | [1.122137, 1.123089] | 0.08286 |
| quality_vs_ls_baseline | 1.000536 | 0.000884 | [1.000223, 1.000795] | 0.000721 |
| phase_gap | 1268.975547 | 14.623206 | [1264.133734, 1273.337137] | 0 |

## Notes
- `chsh_proxy` is CHSH-like classical proxy, not QM Bell test.
- `quality` is phase+local-search cut divided by best simple baseline cut (random-8 vs degree).
- `quality_vs_ls_baseline` compares phase+local-search to local-search baselines (random-LS / degree-LS).
- `phase_gap` is phase cut minus best baseline cut.
