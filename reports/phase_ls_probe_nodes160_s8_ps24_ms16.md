# Phase Entanglement-Like Ablation (Classical Proxy)

Claim boundary: proxy metrics on classical phase dynamics (not quantum entanglement).

## Aggregate Stats

| metric | mean | std | ci95 | p-value vs baseline |
|---|---:|---:|---:|---:|
| pairwise_correlation | 0.454431 | 0.236010 | [0.373532, 0.528653] | 1.98e-17 |
| mutual_info | 2.093015 | 0.497846 | [1.937879, 2.258432] | 6.567e-28 |
| bipartition_entropy | 2.040570 | 0.491343 | [1.887128, 2.203782] | 3.904e-31 |
| chsh_proxy | 0.621904 | 0.209531 | [0.551136, 0.690973] | 0.001858 |
| quality | 1.225562 | 0.004269 | [1.224148, 1.226900] | 0.7113 |
| quality_vs_ls_baseline | 1.003474 | 0.003080 | [1.002526, 1.004446] | 6.55e-09 |
| phase_gap | 149.291481 | 2.546508 | [148.476372, 150.088111] | 0 |

## Notes
- `chsh_proxy` is CHSH-like classical proxy, not QM Bell test.
- `quality` is phase+local-search cut divided by best simple baseline cut (random-8 vs degree).
- `quality_vs_ls_baseline` compares phase+local-search to local-search baselines (random-LS / degree-LS).
- `phase_gap` is phase cut minus best baseline cut.
