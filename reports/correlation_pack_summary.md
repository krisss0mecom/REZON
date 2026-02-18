# Phase Metrics vs Quality Correlation Pack

Scope: classical proxy metrics vs Max-Cut quality proxy.

## phase_entanglement_nodes1000_s8.json

| metric | pearson r | pearson p | spearman r | spearman p |
|---|---:|---:|---:|---:|
| pairwise_correlation | 0.0310 | 0.8575 | -0.0337 | 0.8452 |
| mutual_info | -0.0724 | 0.6749 | 0.0878 | 0.6107 |
| bipartition_entropy | -0.0723 | 0.6754 | 0.0754 | 0.662 |
| chsh_proxy | 0.1391 | 0.4186 | 0.1864 | 0.2765 |

## phase_entanglement_nodes160_s8.json

| metric | pearson r | pearson p | spearman r | spearman p |
|---|---:|---:|---:|---:|
| pairwise_correlation | -0.1543 | 0.3689 | -0.2018 | 0.2379 |
| mutual_info | 0.1468 | 0.393 | 0.1946 | 0.2554 |
| bipartition_entropy | 0.1450 | 0.3989 | 0.1838 | 0.2833 |
| chsh_proxy | -0.1169 | 0.497 | -0.1521 | 0.3758 |

## phase_entanglement_nodes320_s8.json

| metric | pearson r | pearson p | spearman r | spearman p |
|---|---:|---:|---:|---:|
| pairwise_correlation | -0.3373 | 0.04423 | -0.3403 | 0.04229 |
| mutual_info | 0.3514 | 0.03559 | 0.3037 | 0.07172 |
| bipartition_entropy | 0.3517 | 0.03544 | 0.3310 | 0.04861 |
| chsh_proxy | -0.2832 | 0.09419 | -0.2728 | 0.1074 |

## phase_entanglement_nodes640_s8.json

| metric | pearson r | pearson p | spearman r | spearman p |
|---|---:|---:|---:|---:|
| pairwise_correlation | -0.0931 | 0.5893 | -0.0147 | 0.9323 |
| mutual_info | 0.1502 | 0.382 | 0.0674 | 0.6959 |
| bipartition_entropy | 0.1503 | 0.3815 | 0.0597 | 0.7294 |
| chsh_proxy | -0.1776 | 0.3001 | -0.2054 | 0.2294 |

## phase_entanglement_nodes80_s8.json

| metric | pearson r | pearson p | spearman r | spearman p |
|---|---:|---:|---:|---:|
| pairwise_correlation | 0.5388 | 0.000698 | 0.5588 | 0.0003965 |
| mutual_info | -0.5394 | 0.0006862 | -0.4531 | 0.005524 |
| bipartition_entropy | -0.5338 | 0.0008009 | -0.4592 | 0.004844 |
| chsh_proxy | 0.1912 | 0.264 | 0.2378 | 0.1625 |

## Aggregate

| metric | pearson r | spearman r |
|---|---:|---:|
| pairwise_correlation | 0.3322 | 0.2963 |
| mutual_info | -0.3587 | -0.3153 |
| bipartition_entropy | -0.4059 | -0.3580 |
| chsh_proxy | -0.0125 | -0.0009 |
