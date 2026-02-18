# Phase Entanglement-Like Ablation (Classical Proxy)

Claim boundary: proxy metrics on classical phase dynamics (not quantum entanglement).

## Aggregate Stats

| metric | mean | std | ci95 | p-value vs baseline |
|---|---:|---:|---:|---:|
| pairwise_correlation | 0.762600 | 0.272075 | [0.665474, 0.848887] | 0.0006869 |
| mutual_info | 1.267173 | 0.808708 | [1.018697, 1.547885] | 3.625e-10 |
| bipartition_entropy | 1.235474 | 0.791221 | [0.991872, 1.509937] | 1.937e-08 |
| chsh_proxy | 0.791068 | 0.378920 | [0.652946, 0.923484] | 6.93e-05 |
| quality | 0.455227 | 3.319490 | [-0.605493, 1.411587] | 0.7259 |

## Notes
- `chsh_proxy` is CHSH-like classical proxy, not QM Bell test.
- `quality` is Max-Cut objective from phase-derived partition.
