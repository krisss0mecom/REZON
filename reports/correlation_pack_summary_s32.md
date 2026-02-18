# Phase Metrics vs Quality Correlation Pack

Scope: classical proxy metrics vs Max-Cut quality proxy.

## phase_entanglement_nodes1000_s32.json

| metric | pearson r | pearson p | spearman r | spearman p |
|---|---:|---:|---:|---:|
| pairwise_correlation | -0.0434 | 0.8016 | -0.0973 | 0.5724 |
| mutual_info | 0.0113 | 0.948 | 0.0721 | 0.6762 |
| bipartition_entropy | 0.0112 | 0.9482 | 0.0795 | 0.6447 |
| chsh_proxy | 0.0135 | 0.9377 | 0.0916 | 0.5951 |

## phase_entanglement_nodes160_s32.json

| metric | pearson r | pearson p | spearman r | spearman p |
|---|---:|---:|---:|---:|
| pairwise_correlation | 0.1465 | 0.3938 | 0.1967 | 0.2503 |
| mutual_info | -0.1254 | 0.4661 | -0.1696 | 0.3226 |
| bipartition_entropy | -0.1261 | 0.4637 | -0.1954 | 0.2535 |
| chsh_proxy | 0.1073 | 0.5332 | 0.1526 | 0.3741 |

## phase_entanglement_nodes320_s32.json

| metric | pearson r | pearson p | spearman r | spearman p |
|---|---:|---:|---:|---:|
| pairwise_correlation | -0.0836 | 0.6279 | -0.0994 | 0.5643 |
| mutual_info | 0.1045 | 0.5443 | 0.0754 | 0.662 |
| bipartition_entropy | 0.1039 | 0.5464 | 0.0674 | 0.6959 |
| chsh_proxy | -0.0498 | 0.7732 | -0.0281 | 0.871 |

## phase_entanglement_nodes640_s32.json

| metric | pearson r | pearson p | spearman r | spearman p |
|---|---:|---:|---:|---:|
| pairwise_correlation | -0.3292 | 0.04995 | -0.2911 | 0.08496 |
| mutual_info | 0.3565 | 0.03282 | 0.2896 | 0.0867 |
| bipartition_entropy | 0.3566 | 0.03277 | 0.2677 | 0.1145 |
| chsh_proxy | -0.3532 | 0.0346 | -0.3506 | 0.03606 |

## phase_entanglement_nodes80_s32.json

| metric | pearson r | pearson p | spearman r | spearman p |
|---|---:|---:|---:|---:|
| pairwise_correlation | 0.3432 | 0.04042 | 0.2893 | 0.087 |
| mutual_info | -0.3137 | 0.06245 | -0.1876 | 0.2731 |
| bipartition_entropy | -0.3067 | 0.06889 | -0.1704 | 0.3204 |
| chsh_proxy | -0.1002 | 0.5608 | -0.1426 | 0.4067 |

## Aggregate

| metric | pearson r | spearman r |
|---|---:|---:|
| pairwise_correlation | 0.2205 | 0.2665 |
| mutual_info | -0.2557 | -0.3006 |
| bipartition_entropy | -0.3155 | -0.3347 |
| chsh_proxy | -0.0662 | -0.0667 |
