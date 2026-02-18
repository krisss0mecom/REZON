# Phase Entanglement-Like Ablation (Classical Proxy)

Claim boundary: proxy metrics on classical phase dynamics (not quantum entanglement).

## Aggregate Stats

| metric | mean | std | ci95 | p-value vs baseline |
|---|---:|---:|---:|---:|
| pairwise_correlation | 0.662168 | 0.332421 | [0.539597, 0.774061] | 1.412e-05 |
| mutual_info | 1.567719 | 0.911833 | [1.242049, 1.883696] | 6.536e-19 |
| bipartition_entropy | 1.553997 | 0.908047 | [1.229645, 1.869005] | 1.905e-19 |
| chsh_proxy | 0.691216 | 0.545346 | [0.504876, 0.884794] | 6.093e-29 |
| quality | -2.137254 | 18.199941 | [-7.820122, 3.937679] | 0.112 |

## Notes
- `chsh_proxy` is CHSH-like classical proxy, not QM Bell test.
- `quality` is Max-Cut objective from phase-derived partition.
