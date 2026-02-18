# Phase Entanglement-Like Ablation (Classical Proxy)

Claim boundary: proxy metrics on classical phase dynamics (not quantum entanglement).

## Aggregate Stats

| metric | mean | std | ci95 | p-value vs baseline |
|---|---:|---:|---:|---:|
| pairwise_correlation | 0.357577 | 0.258989 | [0.270770, 0.445774] | 1.035e-148 |
| mutual_info | 2.283435 | 0.517727 | [2.110940, 2.452852] | 3.615e-156 |
| bipartition_entropy | 2.251780 | 0.517578 | [2.079014, 2.420943] | 8.216e-150 |
| chsh_proxy | 0.641801 | 0.284099 | [0.547031, 0.733658] | 1.574e-47 |
| quality | 1.169855 | 0.001506 | [1.169361, 1.170323] | 0.983 |
| quality_vs_ls_baseline | 1.002584 | 0.000736 | [1.002319, 1.002834] | 1.097e-58 |
| phase_gap | 440.973777 | 3.485281 | [439.866137, 442.071671] | 0 |

## Notes
- `chsh_proxy` is CHSH-like classical proxy, not QM Bell test.
- `quality` is phase+local-search cut divided by best simple baseline cut (random-8 vs degree).
- `quality_vs_ls_baseline` compares phase+local-search to local-search baselines (random-LS / degree-LS).
- `phase_gap` is phase cut minus best baseline cut.
