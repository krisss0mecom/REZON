# Phase Entanglement-Like Ablation (Classical Proxy)

Claim boundary: proxy metrics on classical phase dynamics (not quantum entanglement).

## Aggregate Stats

| metric | mean | std | ci95 | p-value vs baseline |
|---|---:|---:|---:|---:|
| pairwise_correlation | 0.397518 | 0.239263 | [0.317230, 0.474505] | 1.726e-32 |
| mutual_info | 2.173363 | 0.499875 | [2.009709, 2.333295] | 2.079e-55 |
| bipartition_entropy | 2.115379 | 0.496166 | [1.952070, 2.274209] | 2.852e-57 |
| chsh_proxy | 0.660876 | 0.224164 | [0.587850, 0.736401] | 5.975e-07 |
| quality | 1.227321 | 0.003264 | [1.226205, 1.228477] | 0.5877 |
| quality_vs_ls_baseline | 1.003885 | 0.002144 | [1.003221, 1.004558] | 4.611e-22 |
| phase_gap | 149.593567 | 1.935641 | [148.950075, 150.246956] | 0 |

## Notes
- `chsh_proxy` is CHSH-like classical proxy, not QM Bell test.
- `quality` is phase+local-search cut divided by best simple baseline cut (random-8 vs degree).
- `quality_vs_ls_baseline` compares phase+local-search to local-search baselines (random-LS / degree-LS).
- `phase_gap` is phase cut minus best baseline cut.
