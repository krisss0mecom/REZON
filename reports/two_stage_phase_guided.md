# Two-Stage Phase-Guided Benchmark

| bits | bf attempts | two-stage attempts | attempt gain | bf time (s) | two-stage time (s) | time gain | shortlist hit rate |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 128.50 | 1.00 | 128.500 | 0.000241 | 0.039277 | 0.006 | 1.000 |
| 10 | 512.50 | 1.00 | 512.500 | 0.000838 | 0.038847 | 0.022 | 1.000 |
| 12 | 2048.50 | 2.00 | 1024.250 | 0.003696 | 0.052611 | 0.070 | 1.000 |
| 14 | 8192.50 | 2.50 | 3277.000 | 0.013975 | 0.151570 | 0.092 | 1.000 |

Two-stage = cheap coarse score on all keys, expensive RC score only on shortlist.
