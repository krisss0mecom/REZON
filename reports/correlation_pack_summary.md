# Phase Metrics vs Quality Correlation Pack

Scope: classical proxy metrics vs Max-Cut quality proxy.

## phase_entanglement_nodes1000_s4.json

| metric | pearson r | pearson p | spearman r | spearman p |
|---|---:|---:|---:|---:|
| pairwise_correlation | -0.0906 | 0.5991 | -0.0067 | 0.9691 |
| mutual_info | 0.1624 | 0.3439 | 0.1023 | 0.5526 |
| bipartition_entropy | 0.1625 | 0.3438 | 0.1116 | 0.517 |
| chsh_proxy | 0.0692 | 0.6884 | 0.0685 | 0.6915 |

## phase_entanglement_nodes160_s4.json

| metric | pearson r | pearson p | spearman r | spearman p |
|---|---:|---:|---:|---:|
| pairwise_correlation | -0.1483 | 0.388 | -0.0577 | 0.738 |
| mutual_info | 0.1725 | 0.3143 | 0.0322 | 0.852 |
| bipartition_entropy | 0.1710 | 0.3187 | 0.0330 | 0.8485 |
| chsh_proxy | 0.2712 | 0.1096 | 0.3673 | 0.02756 |

## phase_entanglement_nodes320_s4.json

| metric | pearson r | pearson p | spearman r | spearman p |
|---|---:|---:|---:|---:|
| pairwise_correlation | -0.7138 | 1.021e-06 | -0.3583 | 0.0319 |
| mutual_info | 0.5336 | 0.0008035 | 0.3580 | 0.03203 |
| bipartition_entropy | 0.5358 | 0.0007586 | 0.3580 | 0.03203 |
| chsh_proxy | -0.5701 | 0.0002833 | -0.4245 | 0.00986 |

## phase_entanglement_nodes640_s4.json

| metric | pearson r | pearson p | spearman r | spearman p |
|---|---:|---:|---:|---:|
| pairwise_correlation | -0.5358 | 0.000757 | -0.1075 | 0.5327 |
| mutual_info | 0.3408 | 0.04195 | 0.1525 | 0.3745 |
| bipartition_entropy | 0.3439 | 0.04 | 0.1566 | 0.3616 |
| chsh_proxy | -0.1238 | 0.4721 | -0.0506 | 0.7695 |

## phase_entanglement_nodes80_s4.json

| metric | pearson r | pearson p | spearman r | spearman p |
|---|---:|---:|---:|---:|
| pairwise_correlation | -0.7841 | 1.536e-08 | -0.5427 | 0.0006263 |
| mutual_info | 0.6913 | 3.042e-06 | 0.5556 | 0.0004353 |
| bipartition_entropy | 0.6918 | 2.969e-06 | 0.5600 | 0.0003831 |
| chsh_proxy | -0.8249 | 6.152e-10 | -0.6895 | 3.296e-06 |

## Aggregate

| metric | pearson r | spearman r |
|---|---:|---:|
| pairwise_correlation | -0.3714 | -0.1118 |
| mutual_info | 0.2835 | 0.1623 |
| bipartition_entropy | 0.2861 | 0.1540 |
| chsh_proxy | -0.1294 | -0.0481 |
