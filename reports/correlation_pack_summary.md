# Phase Metrics vs Quality Correlation Pack

Scope: classical proxy metrics vs Max-Cut quality proxy.

## phase_entanglement_nodes1000_s8.json

| metric | pearson r | pearson p | spearman r | spearman p |
|---|---:|---:|---:|---:|
| pairwise_correlation | 0.2292 | 0.1787 | 0.2136 | 0.2109 |
| mutual_info | -0.2539 | 0.1351 | -0.1786 | 0.2972 |
| bipartition_entropy | -0.2540 | 0.135 | -0.1956 | 0.2529 |
| chsh_proxy | 0.2217 | 0.1939 | 0.2301 | 0.177 |

## phase_entanglement_nodes160_s8.json

| metric | pearson r | pearson p | spearman r | spearman p |
|---|---:|---:|---:|---:|
| pairwise_correlation | 0.0031 | 0.9857 | 0.0046 | 0.9786 |
| mutual_info | -0.0033 | 0.9846 | 0.0528 | 0.7599 |
| bipartition_entropy | -0.0014 | 0.9934 | 0.0468 | 0.7862 |
| chsh_proxy | -0.1386 | 0.4201 | -0.1408 | 0.4127 |

## phase_entanglement_nodes320_s8.json

| metric | pearson r | pearson p | spearman r | spearman p |
|---|---:|---:|---:|---:|
| pairwise_correlation | -0.2728 | 0.1075 | -0.2654 | 0.1178 |
| mutual_info | 0.3159 | 0.06055 | 0.2801 | 0.09807 |
| bipartition_entropy | 0.3175 | 0.0592 | 0.3233 | 0.05444 |
| chsh_proxy | -0.2937 | 0.08216 | -0.3272 | 0.05146 |

## phase_entanglement_nodes640_s8.json

| metric | pearson r | pearson p | spearman r | spearman p |
|---|---:|---:|---:|---:|
| pairwise_correlation | 0.2677 | 0.1144 | 0.2999 | 0.07559 |
| mutual_info | -0.1866 | 0.276 | -0.3019 | 0.07351 |
| bipartition_entropy | -0.1879 | 0.2725 | -0.2996 | 0.07585 |
| chsh_proxy | 0.0545 | 0.7521 | -0.0613 | 0.7226 |

## phase_entanglement_nodes80_s8.json

| metric | pearson r | pearson p | spearman r | spearman p |
|---|---:|---:|---:|---:|
| pairwise_correlation | 0.4325 | 0.008424 | 0.4757 | 0.003366 |
| mutual_info | -0.4224 | 0.01028 | -0.3839 | 0.02079 |
| bipartition_entropy | -0.4170 | 0.0114 | -0.3871 | 0.01967 |
| chsh_proxy | 0.0989 | 0.566 | 0.1936 | 0.258 |

## Aggregate

| metric | pearson r | spearman r |
|---|---:|---:|
| pairwise_correlation | 0.3321 | 0.3248 |
| mutual_info | -0.3572 | -0.3424 |
| bipartition_entropy | -0.4045 | -0.3849 |
| chsh_proxy | -0.0146 | 0.0079 |
