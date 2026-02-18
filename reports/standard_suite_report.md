# Standard Suite Benchmark

Claims scope: quantum-inspired analog search heuristic (not QC replacement).

## Aggregate Stats vs no_rc

| problem | compare | delta quality mean | p-value |
|---|---|---:|---:|
| maxcut | rc_no_anchor vs no_rc | -1.438339 | 0.1467 |
| maxcut | rc_anchor vs no_rc | -1.428426 | 0.4099 |
| maxcut | rc_anchor_rls vs no_rc | -3.807832 | 0.1315 |
| qubo | rc_no_anchor vs no_rc | -4.857135 | 0.2131 |
| qubo | rc_anchor vs no_rc | -4.874239 | 0.1317 |
| qubo | rc_anchor_rls vs no_rc | -6.445864 | 0.1017 |
| sat | rc_no_anchor vs no_rc | -1.250000 | 0.1257 |
| sat | rc_anchor vs no_rc | -0.750000 | 0.2048 |
| sat | rc_anchor_rls vs no_rc | -3.000000 | 0.2048 |

## Notes
- Quality/time/energy/stability measured across many seeds.
- Ablations include no RC, RC without anchor, RC with anchor, RC with anchor+RLS.
