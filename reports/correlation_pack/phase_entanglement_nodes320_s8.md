# Phase Entanglement-Like Ablation (Classical Proxy)

Claim boundary: proxy metrics on classical phase dynamics (not quantum entanglement).

## Aggregate Stats

| metric | mean | std | ci95 | p-value vs baseline |
|---|---:|---:|---:|---:|
| pairwise_correlation | 0.345692 | 0.242404 | [0.264304, 0.425711] | 6.489e-43 |
| mutual_info | 2.304130 | 0.480760 | [2.144507, 2.461150] | 1.521e-34 |
| bipartition_entropy | 2.270117 | 0.479887 | [2.110702, 2.427214] | 2.194e-35 |
| chsh_proxy | 0.618637 | 0.241308 | [0.538273, 0.696358] | 6.733e-10 |
| quality | 1.168442 | 0.002990 | [1.167423, 1.169432] | 0.3121 |
| quality_vs_ls_baseline | 1.001690 | 0.001712 | [1.001120, 1.002275] | 3.433e-08 |
| phase_gap | 437.896013 | 6.941652 | [435.469896, 440.207077] | 0 |

## Notes
- `chsh_proxy` is CHSH-like classical proxy, not QM Bell test.
- `quality` is phase+local-search cut divided by best simple baseline cut (random-8 vs degree).
- `quality_vs_ls_baseline` compares phase+local-search to local-search baselines (random-LS / degree-LS).
- `phase_gap` is phase cut minus best baseline cut.
