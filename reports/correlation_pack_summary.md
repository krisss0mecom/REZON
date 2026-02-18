# Phase Metrics vs Quality Correlation Pack

Scope: classical proxy metrics vs Max-Cut quality proxy.

## phase_entanglement_nodes1000_s16.json

| metric | pearson r | pearson p | spearman r | spearman p |
|---|---:|---:|---:|---:|
| pairwise_correlation | 0.0016 | 0.9928 | -0.1099 | 0.5234 |
| mutual_info | -0.0529 | 0.7595 | 0.0940 | 0.5857 |
| bipartition_entropy | -0.0528 | 0.7597 | 0.0875 | 0.6118 |
| chsh_proxy | 0.1312 | 0.4456 | 0.1951 | 0.2541 |

## phase_entanglement_nodes160_s16.json

| metric | pearson r | pearson p | spearman r | spearman p |
|---|---:|---:|---:|---:|
| pairwise_correlation | -0.0708 | 0.6814 | -0.0517 | 0.7644 |
| mutual_info | 0.0714 | 0.6791 | 0.1248 | 0.4682 |
| bipartition_entropy | 0.0694 | 0.6878 | 0.0680 | 0.6937 |
| chsh_proxy | -0.0184 | 0.9152 | -0.0546 | 0.7519 |

## phase_entanglement_nodes320_s16.json

| metric | pearson r | pearson p | spearman r | spearman p |
|---|---:|---:|---:|---:|
| pairwise_correlation | 0.0728 | 0.6733 | 0.0489 | 0.777 |
| mutual_info | -0.0329 | 0.8491 | -0.1187 | 0.4906 |
| bipartition_entropy | -0.0341 | 0.8436 | -0.1148 | 0.505 |
| chsh_proxy | 0.0823 | 0.6332 | 0.1024 | 0.5521 |

## phase_entanglement_nodes640_s16.json

| metric | pearson r | pearson p | spearman r | spearman p |
|---|---:|---:|---:|---:|
| pairwise_correlation | 0.1225 | 0.4766 | 0.1910 | 0.2645 |
| mutual_info | -0.1115 | 0.5172 | -0.1732 | 0.3123 |
| bipartition_entropy | -0.1111 | 0.5188 | -0.1884 | 0.2711 |
| chsh_proxy | 0.0289 | 0.8669 | 0.0744 | 0.6663 |

## phase_entanglement_nodes80_s16.json

| metric | pearson r | pearson p | spearman r | spearman p |
|---|---:|---:|---:|---:|
| pairwise_correlation | 0.4478 | 0.006169 | 0.3115 | 0.06445 |
| mutual_info | -0.3990 | 0.01592 | -0.3042 | 0.07122 |
| bipartition_entropy | -0.3924 | 0.01794 | -0.2898 | 0.08641 |
| chsh_proxy | 0.0313 | 0.8561 | 0.0770 | 0.6555 |

## Aggregate

| metric | pearson r | spearman r |
|---|---:|---:|
| pairwise_correlation | 0.2103 | 0.2489 |
| mutual_info | -0.2495 | -0.3041 |
| bipartition_entropy | -0.3099 | -0.3531 |
| chsh_proxy | -0.0526 | 0.0071 |
