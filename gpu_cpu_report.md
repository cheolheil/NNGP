# CPU versus GPU simulation timings

{'python': '3.10.20', 'torch': '2.12.0+cu132', 'cpu': 'Intel64 Family 6 Model 151 Stepping 2, GenuineIntel', 'gpu': 'NVIDIA GeForce RTX 3080', 'cpu_threads': 1, 'dtype': 'float64'}

Seed 42; identical CPU-generated data and initial parameters; 3 paired restarts; full fit includes prediction cache and history; transfers, model construction and warmup excluded; prediction median of 5 calls per restart; CUDA synchronized.

| Dimension | Method | CPU fit total (s) | GPU fit total (s) | CPU/GPU speedup | CPU predict (s) | GPU predict (s) |
|---|---|---:|---:|---:|---:|---:|
| 1D | alternating | 4.956 | 14.564 | 0.340x | 0.006765 | 0.014591 |
| 1D | joint | 2.168 | 6.199 | 0.350x | 0.006430 | 0.013634 |
| 4D | alternating | 32.416 | 29.025 | 1.117x | 0.117387 | 0.034774 |
| 4D | joint | 18.975 | 15.787 | 1.202x | 0.135531 | 0.032583 |

Fit totals cover three restarts. Speedup below 1 means GPU is slower. CPU uses one thread to match the existing simulation protocol. These are small-data workloads; results do not establish scaling at larger N. See JSON for individual times, losses, RMSE, failures and settings.
