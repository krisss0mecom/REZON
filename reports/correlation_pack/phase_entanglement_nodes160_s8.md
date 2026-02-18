# Phase Entanglement-Like Ablation (Classical Proxy)

Claim boundary: proxy metrics on classical phase dynamics (not quantum entanglement).

## Aggregate Stats

| metric | mean | std | ci95 | p-value vs baseline |
|---|---:|---:|---:|---:|
| pairwise_correlation | 0.454431 | 0.236010 | [0.373532, 0.528653] | 1.98e-17 |
| mutual_info | 2.093015 | 0.497846 | [1.937879, 2.258432] | 6.567e-28 |
| bipartition_entropy | 2.040570 | 0.491343 | [1.887128, 2.203782] | 3.904e-31 |
| chsh_proxy | 0.621904 | 0.209531 | [0.551136, 0.690973] | 0.001858 |
| quality | 1.227209 | 0.004713 | [1.225673, 1.228823] | 0.6958 |
| quality_vs_ls_baseline | 1.004827 | 0.002426 | [1.004004, 1.005684] | 8.584e-17 |
| phase_gap | 150.382655 | 2.806302 | [149.471643, 151.315453] | 0 |

## Notes
- `chsh_proxy` is CHSH-like classical proxy, not QM Bell test.
- `quality` is phase+local-search cut divided by best simple baseline cut (random-8 vs degree).
- `quality_vs_ls_baseline` compares phase+local-search to local-search baselines (random-LS / degree-LS).
- `phase_gap` is phase cut minus best baseline cut.
