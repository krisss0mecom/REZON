# Phase Metrics vs Quality Correlation Pack

Scope: classical proxy metrics vs Max-Cut quality proxy.

## phase_entanglement_nodes1000_s8.json

| metric | pearson r | pearson p | spearman r | spearman p |
|---|---:|---:|---:|---:|
| pairwise_correlation | -0.9902 | 1.249e-30 | -0.9825 | 2.118e-26 |
| mutual_info | 0.9582 | 4.705e-20 | 0.9671 | 8.801e-22 |
| bipartition_entropy | 0.9588 | 3.713e-20 | 0.9689 | 3.43e-22 |
| chsh_proxy | -0.9648 | 2.661e-21 | -0.9786 | 6.092e-25 |

## phase_entanglement_nodes160_s8.json

| metric | pearson r | pearson p | spearman r | spearman p |
|---|---:|---:|---:|---:|
| pairwise_correlation | -0.9623 | 8.488e-21 | -0.9166 | 4.278e-15 |
| mutual_info | 0.9266 | 5.269e-16 | 0.8113 | 1.953e-09 |
| bipartition_entropy | 0.9277 | 4.148e-16 | 0.8172 | 1.193e-09 |
| chsh_proxy | -0.9844 | 2.971e-27 | -0.9511 | 6.418e-19 |

## phase_entanglement_nodes320_s8.json

| metric | pearson r | pearson p | spearman r | spearman p |
|---|---:|---:|---:|---:|
| pairwise_correlation | -0.9633 | 5.276e-21 | -0.9308 | 2.022e-16 |
| mutual_info | 0.9309 | 1.974e-16 | 0.9398 | 2.028e-17 |
| bipartition_entropy | 0.9315 | 1.677e-16 | 0.9382 | 3.082e-17 |
| chsh_proxy | -0.9672 | 8.356e-22 | -0.9302 | 2.285e-16 |

## phase_entanglement_nodes640_s8.json

| metric | pearson r | pearson p | spearman r | spearman p |
|---|---:|---:|---:|---:|
| pairwise_correlation | -0.9967 | 9.029e-39 | -0.9964 | 5.049e-38 |
| mutual_info | 0.9500 | 9.345e-19 | 0.9768 | 2.38e-24 |
| bipartition_entropy | 0.9511 | 6.472e-19 | 0.9771 | 1.972e-24 |
| chsh_proxy | -0.9266 | 5.333e-16 | -0.9786 | 6.092e-25 |

## phase_entanglement_nodes80_s8.json

| metric | pearson r | pearson p | spearman r | spearman p |
|---|---:|---:|---:|---:|
| pairwise_correlation | -0.9830 | 1.333e-26 | -0.8773 | 2.229e-12 |
| mutual_info | 0.9366 | 4.779e-17 | 0.8895 | 4.126e-13 |
| bipartition_entropy | 0.9368 | 4.513e-17 | 0.8882 | 4.973e-13 |
| chsh_proxy | -0.9382 | 3.118e-17 | -0.8156 | 1.368e-09 |

## Aggregate

| metric | pearson r | spearman r |
|---|---:|---:|
| pairwise_correlation | -0.9799 | -0.9547 |
| mutual_info | 0.9402 | 0.9201 |
| bipartition_entropy | 0.9408 | 0.9190 |
| chsh_proxy | -0.9467 | -0.9411 |
