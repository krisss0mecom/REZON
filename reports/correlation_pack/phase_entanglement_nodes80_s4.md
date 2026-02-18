# Phase Entanglement-Like Ablation (Classical Proxy)

Claim boundary: proxy metrics on classical phase dynamics (not quantum entanglement).

## Aggregate Stats

| metric | mean | std | ci95 | p-value vs baseline |
|---|---:|---:|---:|---:|
| pairwise_correlation | 0.821427 | 0.246991 | [0.730924, 0.894686] | 0.005442 |
| mutual_info | 1.056932 | 0.773454 | [0.814504, 1.327893] | 5.971e-07 |
| bipartition_entropy | 1.013871 | 0.737076 | [0.783108, 1.271465] | 2.055e-07 |
| chsh_proxy | 0.924996 | 0.357809 | [0.796836, 1.053738] | 1.163e-05 |
| quality | 0.597432 | 1.743521 | [-0.003086, 1.235225] | 0.4758 |

## Notes
- `chsh_proxy` is CHSH-like classical proxy, not QM Bell test.
- `quality` is Max-Cut objective from phase-derived partition.
