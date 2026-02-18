# Phase Entanglement-Like Ablation (Classical Proxy)

Claim boundary: proxy metrics on classical phase dynamics (not quantum entanglement).

## Aggregate Stats

| metric | mean | std | ci95 | p-value vs baseline |
|---|---:|---:|---:|---:|
| pairwise_correlation | 0.695739 | 0.316549 | [0.577726, 0.799330] | 3.814e-15 |
| mutual_info | 1.479957 | 0.874795 | [1.178382, 1.788858] | 1.271e-46 |
| bipartition_entropy | 1.456216 | 0.864788 | [1.158249, 1.761879] | 1.746e-48 |
| chsh_proxy | 0.796864 | 0.455278 | [0.633334, 0.960462] | 7.178e-49 |
| quality | 0.324614 | 0.325538 | [0.214092, 0.444367] | 6.745e-10 |
| phase_gap | -1755.309952 | 847.881104 | [-2043.285370, -1443.351878] | 9.295e-92 |

## Notes
- `chsh_proxy` is CHSH-like classical proxy, not QM Bell test.
- `quality` is phase cut divided by best baseline cut (random-8 vs degree baseline).
- `phase_gap` is phase cut minus best baseline cut.
