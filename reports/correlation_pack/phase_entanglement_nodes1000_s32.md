# Phase Entanglement-Like Ablation (Classical Proxy)

Claim boundary: proxy metrics on classical phase dynamics (not quantum entanglement).

## Aggregate Stats

| metric | mean | std | ci95 | p-value vs baseline |
|---|---:|---:|---:|---:|
| pairwise_correlation | 0.323353 | 0.273032 | [0.233835, 0.415770] | 6.629e-178 |
| mutual_info | 2.355238 | 0.526182 | [2.182504, 2.524486] | 7.124e-154 |
| bipartition_entropy | 2.345274 | 0.526276 | [2.172479, 2.514506] | 2.486e-153 |
| chsh_proxy | 0.657215 | 0.292146 | [0.560661, 0.751560] | 1.742e-71 |
| quality | 1.101128 | 0.000405 | [1.101006, 1.101266] | 0.5919 |
| quality_vs_ls_baseline | 1.001318 | 0.000381 | [1.001187, 1.001448] | 1.786e-93 |
| phase_gap | 2537.727845 | 9.706339 | [2534.815727, 2541.088130] | 0 |

## Notes
- `chsh_proxy` is CHSH-like classical proxy, not QM Bell test.
- `quality` is phase+local-search cut divided by best simple baseline cut (random-8 vs degree).
- `quality_vs_ls_baseline` compares phase+local-search to local-search baselines (random-LS / degree-LS).
- `phase_gap` is phase cut minus best baseline cut.
