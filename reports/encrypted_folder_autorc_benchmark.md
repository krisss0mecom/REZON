# Autonomous RC Encrypted-Folder Benchmark

| bits | bf avg attempts | rc avg attempts | attempt gain | bf avg time (s) | rc avg time (s) | time gain |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 128.67 | 1.00 | 128.667 | 0.001415 | 0.108828 | 0.013 |
| 10 | 512.67 | 1.00 | 512.667 | 0.003174 | 0.431435 | 0.007 |
| 12 | 2048.67 | 1.67 | 1229.200 | 0.010290 | 1.725340 | 0.006 |
| 14 | 8192.67 | 5.00 | 1638.533 | 0.039251 | 6.798585 | 0.006 |
| 16 | 32768.67 | 31.33 | 1045.809 | 0.155829 | 27.421574 | 0.006 |

Autonomous means: RC generates ordering itself; no user-provided candidate list.
Scope: synthetic benchmark only.