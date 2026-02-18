# Phase Entanglement-Like Ablation (Classical Proxy)

Claim boundary: proxy metrics on classical phase dynamics (not quantum entanglement).

## Aggregate Stats

| metric | mean | std | ci95 | p-value vs baseline |
|---|---:|---:|---:|---:|
| pairwise_correlation | 0.645832 | 0.346559 | [0.518625, 0.765835] | 2.942e-27 |
| mutual_info | 1.633920 | 0.890475 | [1.309638, 1.948517] | 2.557e-63 |
| bipartition_entropy | 1.625289 | 0.888259 | [1.301339, 1.939230] | 2.29e-63 |
| chsh_proxy | 0.778938 | 0.532097 | [0.591597, 0.971779] | 1.862e-32 |
| quality | 0.359841 | 0.363388 | [0.232629, 0.491592] | 3.055e-29 |
| phase_gap | -16066.084629 | 9115.793020 | [-19258.045494, -12760.453503] | 6.143e-83 |

## Notes
- `chsh_proxy` is CHSH-like classical proxy, not QM Bell test.
- `quality` is phase cut divided by best baseline cut (random-8 vs degree baseline).
- `phase_gap` is phase cut minus best baseline cut.
