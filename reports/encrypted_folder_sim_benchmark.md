# Synthetic Encrypted Folder Benchmark

| bits | bf avg attempts | phase-guided avg attempts | attempt gain | bf avg time (s) | phase avg time (s) | time gain | top-k hit rate |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 128.67 | 1.00 | 128.667 | 0.001529 | 0.019428 | 0.079 | 1.000 |
| 10 | 512.67 | 3.67 | 139.818 | 0.003044 | 0.073355 | 0.042 | 1.000 |
| 12 | 2048.67 | 7.00 | 292.667 | 0.010263 | 0.291982 | 0.035 | 1.000 |
| 14 | 8192.67 | 19.67 | 416.576 | 0.038851 | 1.144135 | 0.034 | 1.000 |
| 16 | 32768.67 | 197.33 | 166.057 | 0.152450 | 4.664200 | 0.033 | 1.000 |

Scope: synthetic benchmark only (controlled toy cipher + synthetic archive).