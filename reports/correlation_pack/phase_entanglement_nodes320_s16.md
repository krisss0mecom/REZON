# Phase Entanglement-Like Ablation (Classical Proxy)

Claim boundary: proxy metrics on classical phase dynamics (not quantum entanglement).

## Aggregate Stats

| metric | mean | std | ci95 | p-value vs baseline |
|---|---:|---:|---:|---:|
| pairwise_correlation | 0.358903 | 0.255207 | [0.273020, 0.445187] | 2.798e-94 |
| mutual_info | 2.282334 | 0.520997 | [2.108219, 2.453143] | 1.808e-75 |
| bipartition_entropy | 2.250832 | 0.520930 | [2.076088, 2.421446] | 1.107e-74 |
| chsh_proxy | 0.631495 | 0.285129 | [0.536173, 0.723473] | 4.004e-21 |
| quality | 1.169476 | 0.002852 | [1.168560, 1.170445] | 0.3945 |
| quality_vs_ls_baseline | 1.002132 | 0.001154 | [1.001721, 1.002547] | 5.63e-22 |
| phase_gap | 440.455092 | 6.565053 | [438.322874, 442.722151] | 0 |

## Notes
- `chsh_proxy` is CHSH-like classical proxy, not QM Bell test.
- `quality` is phase+local-search cut divided by best simple baseline cut (random-8 vs degree).
- `quality_vs_ls_baseline` compares phase+local-search to local-search baselines (random-LS / degree-LS).
- `phase_gap` is phase cut minus best baseline cut.
