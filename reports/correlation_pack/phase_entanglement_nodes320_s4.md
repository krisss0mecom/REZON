# Phase Entanglement-Like Ablation (Classical Proxy)

Claim boundary: proxy metrics on classical phase dynamics (not quantum entanglement).

## Aggregate Stats

| metric | mean | std | ci95 | p-value vs baseline |
|---|---:|---:|---:|---:|
| pairwise_correlation | 0.734339 | 0.302550 | [0.623654, 0.828790] | 2.284e-06 |
| mutual_info | 1.374566 | 0.869480 | [1.087956, 1.683503] | 1.727e-20 |
| bipartition_entropy | 1.352381 | 0.860214 | [1.069546, 1.657510] | 7.844e-24 |
| chsh_proxy | 0.776022 | 0.440903 | [0.615918, 0.932563] | 4.82e-39 |
| quality | 2.610572 | 9.943025 | [-0.420675, 6.133105] | 0.2648 |

## Notes
- `chsh_proxy` is CHSH-like classical proxy, not QM Bell test.
- `quality` is Max-Cut objective from phase-derived partition.
