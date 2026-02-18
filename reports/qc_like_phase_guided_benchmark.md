# Phase-Guided Decryption Benchmark

| bits | keyspace | bf avg attempts | phase-guided avg attempts | attempts gain (bf/pg) | bf avg time (s) | pg avg time (s) | time gain (bf/pg) | Grover ref queries |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 256 | 128.50 | 1.67 | 77.100 | 0.000196 | 0.018272 | 0.011 | 16.00 |
| 10 | 1024 | 512.50 | 3.67 | 139.773 | 0.000738 | 0.072043 | 0.010 | 32.00 |
| 12 | 4096 | 2048.50 | 11.25 | 182.089 | 0.002945 | 0.287583 | 0.010 | 64.00 |
| 14 | 16384 | 8192.67 | 67.67 | 121.074 | 0.011787 | 1.149411 | 0.010 | 128.00 |
| 16 | 65536 | 32768.50 | 596.00 | 54.981 | 0.046793 | 4.603483 | 0.010 | 256.00 |

Notes:
- Phase-guided search reorders candidates using RC-inspired analog scoring.
- Gains in attempts do not guarantee runtime gains due to ranking overhead.