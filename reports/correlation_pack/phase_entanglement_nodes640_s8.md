# Phase Entanglement-Like Ablation (Classical Proxy)

Claim boundary: proxy metrics on classical phase dynamics (not quantum entanglement).

## Aggregate Stats

| metric | mean | std | ci95 | p-value vs baseline |
|---|---:|---:|---:|---:|
| pairwise_correlation | 0.340544 | 0.271907 | [0.250981, 0.434179] | 2.366e-36 |
| mutual_info | 2.330171 | 0.523206 | [2.157891, 2.499993] | 8.517e-42 |
| bipartition_entropy | 2.312705 | 0.524361 | [2.139897, 2.482522] | 3.391e-42 |
| chsh_proxy | 0.641282 | 0.290292 | [0.543770, 0.733880] | 2.291e-13 |
| quality | 1.123925 | 0.001426 | [1.123470, 1.124364] | 0.517 |
| quality_vs_ls_baseline | 1.001683 | 0.001047 | [1.001298, 1.002001] | 1.889e-19 |
| phase_gap | 1282.296620 | 13.719186 | [1277.948343, 1286.580206] | 0 |

## Notes
- `chsh_proxy` is CHSH-like classical proxy, not QM Bell test.
- `quality` is phase+local-search cut divided by best simple baseline cut (random-8 vs degree).
- `quality_vs_ls_baseline` compares phase+local-search to local-search baselines (random-LS / degree-LS).
- `phase_gap` is phase cut minus best baseline cut.
