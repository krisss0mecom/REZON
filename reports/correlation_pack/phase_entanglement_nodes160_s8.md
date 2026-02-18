# Phase Entanglement-Like Ablation (Classical Proxy)

Claim boundary: proxy metrics on classical phase dynamics (not quantum entanglement).

## Aggregate Stats

| metric | mean | std | ci95 | p-value vs baseline |
|---|---:|---:|---:|---:|
| pairwise_correlation | 0.775960 | 0.267128 | [0.679751, 0.859641] | 4.13e-08 |
| mutual_info | 1.236620 | 0.796653 | [0.988748, 1.516556] | 8.549e-20 |
| bipartition_entropy | 1.205681 | 0.779992 | [0.963167, 1.478703] | 1.472e-19 |
| chsh_proxy | 0.881151 | 0.347739 | [0.754140, 0.998531] | 3.202e-06 |
| quality | 0.259907 | 0.275920 | [0.164220, 0.360150] | 3.012e-06 |
| phase_gap | -490.590403 | 183.118362 | [-553.868426, -424.301662] | 3.796e-113 |

## Notes
- `chsh_proxy` is CHSH-like classical proxy, not QM Bell test.
- `quality` is phase cut divided by best baseline cut (random-8 vs degree baseline).
- `phase_gap` is phase cut minus best baseline cut.
