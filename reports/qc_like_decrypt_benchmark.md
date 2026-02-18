# QC-like Decryption Benchmark

| key bits | keyspace | classical avg attempts | classical avg time (s) | Grover ideal queries | classical/Grover |
|---:|---:|---:|---:|---:|---:|
| 8 | 256 | 128.50 | 0.000197 | 16.00 | 8.03 |
| 10 | 1024 | 512.50 | 0.000738 | 32.00 | 16.02 |
| 12 | 4096 | 2048.50 | 0.002908 | 64.00 | 32.01 |
| 14 | 16384 | 8192.50 | 0.011630 | 128.00 | 64.00 |
| 16 | 65536 | 32768.50 | 0.046578 | 256.00 | 128.00 |
| 18 | 262144 | 131072.50 | 0.186665 | 512.00 | 256.00 |
| 20 | 1048576 | 524288.50 | 0.760276 | 1024.00 | 512.00 |
| 22 | 4194304 | 2097152.50 | 3.033818 | 2048.00 | 1024.00 |
| 24 | 16777216 | 8388608.50 | 12.009578 | 4096.00 | 2048.00 |

Notes:
- Classical column is real brute-force runtime/query count on this machine.
- Grover column is a theoretical query lower bound reference (not measured hardware QC runtime).
- Keys are chosen at fixed quantiles of keyspace for reproducible scaling.