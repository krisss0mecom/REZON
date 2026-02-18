# Phase Entanglement-Like Ablation (Classical Proxy)

Claim boundary: proxy metrics on classical phase dynamics (not quantum entanglement).

## Aggregate Stats

| metric | mean | std | ci95 | p-value vs baseline |
|---|---:|---:|---:|---:|
| pairwise_correlation | 0.663542 | 0.333410 | [0.543222, 0.775876] | 1.436e-08 |
| mutual_info | 1.587542 | 0.871891 | [1.280536, 1.894152] | 1.594e-28 |
| bipartition_entropy | 1.578609 | 0.869695 | [1.271810, 1.884574] | 1.746e-28 |
| chsh_proxy | 0.756138 | 0.537488 | [0.569165, 0.948366] | 2.189e-15 |
| quality | 4.411219 | 12.041753 | [0.324824, 8.192697] | 0.9457 |

## Notes
- `chsh_proxy` is CHSH-like classical proxy, not QM Bell test.
- `quality` is Max-Cut objective from phase-derived partition.
