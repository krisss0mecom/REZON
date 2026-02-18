# Phase Entanglement-Like Ablation (Classical Proxy)

Claim boundary: proxy metrics on classical phase dynamics (not quantum entanglement).

## Aggregate Stats

| metric | mean | std | ci95 | p-value vs baseline |
|---|---:|---:|---:|---:|
| pairwise_correlation | 0.827409 | 0.223941 | [0.746620, 0.895551] | 6.75e-05 |
| mutual_info | 1.074262 | 0.723464 | [0.849368, 1.309098] | 2.625e-09 |
| bipartition_entropy | 1.027715 | 0.685300 | [0.814994, 1.250781] | 6.955e-09 |
| chsh_proxy | 0.873933 | 0.300523 | [0.762061, 0.977081] | 1.045e-06 |
| quality | 0.183187 | 0.246182 | [0.109255, 0.273900] | 9.302e-06 |
| phase_gap | -139.141379 | 42.158211 | [-151.872009, -123.406179] | 3.043e-142 |

## Notes
- `chsh_proxy` is CHSH-like classical proxy, not QM Bell test.
- `quality` is phase cut divided by best baseline cut (random-8 vs degree baseline).
- `phase_gap` is phase cut minus best baseline cut.
