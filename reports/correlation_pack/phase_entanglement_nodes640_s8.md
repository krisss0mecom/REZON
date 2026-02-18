# Phase Entanglement-Like Ablation (Classical Proxy)

Claim boundary: proxy metrics on classical phase dynamics (not quantum entanglement).

## Aggregate Stats

| metric | mean | std | ci95 | p-value vs baseline |
|---|---:|---:|---:|---:|
| pairwise_correlation | 0.650365 | 0.341162 | [0.525507, 0.769353] | 4.346e-14 |
| mutual_info | 1.612732 | 0.890614 | [1.288941, 1.923776] | 1.121e-44 |
| bipartition_entropy | 1.600400 | 0.886954 | [1.277798, 1.910080] | 8.475e-45 |
| chsh_proxy | 0.744307 | 0.537281 | [0.562625, 0.935348] | 6.364e-36 |
| quality | 0.333326 | 0.354460 | [0.207284, 0.465142] | 3.433e-13 |
| phase_gap | -6897.223073 | 3668.622517 | [-8202.209738, -5533.358062] | 1.151e-89 |

## Notes
- `chsh_proxy` is CHSH-like classical proxy, not QM Bell test.
- `quality` is phase cut divided by best baseline cut (random-8 vs degree baseline).
- `phase_gap` is phase cut minus best baseline cut.
