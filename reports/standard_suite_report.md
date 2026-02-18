# Standard Suite Benchmark

Claims scope: quantum-inspired analog search heuristic (not QC replacement).

## Aggregate Stats vs no_rc

| problem | compare | delta quality mean | p-value |
|---|---|---:|---:|
| maxcut | rc_no_anchor vs no_rc | 0.594411 | nan |
| maxcut | rc_anchor vs no_rc | 0.683247 | nan |
| maxcut | rc_anchor_rls vs no_rc | -3.000731 | nan |
| qubo | rc_no_anchor vs no_rc | -6.229224 | nan |
| qubo | rc_anchor vs no_rc | -4.149246 | nan |
| qubo | rc_anchor_rls vs no_rc | -9.221821 | nan |
| sat | rc_no_anchor vs no_rc | -1.000000 | nan |
| sat | rc_anchor vs no_rc | -2.000000 | nan |
| sat | rc_anchor_rls vs no_rc | -4.000000 | nan |

## Notes
- Quality/time/energy/stability measured across many seeds.
- Ablations include no RC, RC without anchor, RC with anchor, RC with anchor+RLS.
