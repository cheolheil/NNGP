# 20D NNGP-EIV: CPU versus GPU

## Problem and protocol

100 training observations; 200 independent exact-input test points; 20 dimensions; seed 42.
Nominal inputs are uniform on [-3,3]^20. Physical inputs W=X+u, with independent Gaussian Berkson errors.
True coordinate variances increase linearly from 0.02 to 0.08. Observation standard deviation is 0.1.
The latent response is f(z)=2-0.25 sum_d(a_d z_d^2)/sqrt(20)+0.5 sin(z_1)+0.5 cos(z_2), with a_d increasing linearly from 0.5 to 1.5.

- Alternating minimizes negative approximate ELBO: second-order expected log likelihood plus input KL. Taylor truncation does not guarantee a bound.
- Joint minimizes moment Gaussian NLL based on the second-order expected kernel.
- Three matched initializations per method/device. Initial S_u=0.01, V=0.001; later starts use matched CPU-generated lognormal perturbations. Restart selection uses final training loss only.
- Alternating: 15 outer iterations, 10 kernel/noise and 10 S_u/V steps (300 updates), learning rates 0.03. Joint: 25 warmup and 75 joint steps (100 updates), learning rates 0.05 and 0.025. Objective values are not comparable between methods.
- Float64 on both devices; one CPU thread. Warmup excluded. CUDA synchronized around timings. Fit includes history and prediction-cache construction. Model construction and data transfers excluded.
- Prediction timing is one call on the selected restart, including latent covariance/interval computation. Total measured time is all three fits plus selected-model prediction; diagnostics and report generation are excluded.
- This is one synthetic dataset with optimization restarts, not a repeated-dataset Monte Carlo study. Test intervals exclude observation and test-input noise.

Environment: {'python': '3.10.20', 'torch': '2.12.0+cu132', 'cpu': 'Intel64 Family 6 Model 151 Stepping 2, GenuineIntel', 'gpu': 'NVIDIA GeForce RTX 3080', 'dtype': 'float64', 'cpu_threads': 1}

## Selected-restart results

| Device | Method | Successful fits | Selected restart | Total fit (s) | Predict (s) | Total measured (s) | RMSE | MAE | R2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cpu | alternating | 3/3 | 1 | 258.274 | 1.395 | 259.668 | 0.62151 | 0.47487 | 0.53524 |
| cuda | alternating | 3/3 | 1 | 132.742 | 0.366 | 133.107 | 0.62151 | 0.47487 | 0.53524 |
| cpu | joint | 3/3 | 1 | 164.610 | 1.166 | 165.775 | 0.62177 | 0.47507 | 0.53484 |
| cuda | joint | 3/3 | 1 | 70.788 | 0.155 | 70.942 | 0.62177 | 0.47507 | 0.53484 |

## Input-noise inference and intervals

| Device | Method | S_u RMSE | Mean relative S_u error | Observation std. | 95% latent coverage | Mean width | Min training eigenvalue |
|---|---|---:|---:|---:|---:|---:|---:|
| cpu | alternating | 0.05025 | 92.56% | 0.02944 | 95.0% | 2.29887 | 0.0973842 |
| cuda | alternating | 0.05025 | 92.56% | 0.02944 | 95.0% | 2.29887 | 0.0973842 |
| cpu | joint | 0.05108 | 94.63% | 0.02790 | 95.0% | 2.28900 | 0.0960679 |
| cuda | joint | 0.05108 | 94.63% | 0.02790 | 95.0% | 2.28900 | 0.0960679 |

Coverage is descriptive for this fixed test set, not evidence of repeated-sample calibration.

## Coordinate-wise input variance estimates

| Coordinate | True S_u | CPU alternating | GPU alternating | CPU joint | GPU joint |
|---|---:|---:|---:|---:|---:|
| 1 | 0.020000 | 0.003159 | 0.003159 | 0.002282 | 0.002282 |
| 2 | 0.023158 | 0.003161 | 0.003161 | 0.002281 | 0.002281 |
| 3 | 0.026316 | 0.003161 | 0.003161 | 0.002270 | 0.002270 |
| 4 | 0.029474 | 0.003162 | 0.003162 | 0.002272 | 0.002272 |
| 5 | 0.032632 | 0.003161 | 0.003161 | 0.002299 | 0.002299 |
| 6 | 0.035789 | 0.003161 | 0.003161 | 0.002277 | 0.002277 |
| 7 | 0.038947 | 0.003161 | 0.003161 | 0.002287 | 0.002287 |
| 8 | 0.042105 | 0.003161 | 0.003161 | 0.002299 | 0.002299 |
| 9 | 0.045263 | 0.003162 | 0.003162 | 0.002288 | 0.002288 |
| 10 | 0.048421 | 0.003162 | 0.003162 | 0.002288 | 0.002288 |
| 11 | 0.051579 | 0.003161 | 0.003161 | 0.002255 | 0.002255 |
| 12 | 0.054737 | 0.003161 | 0.003161 | 0.002286 | 0.002286 |
| 13 | 0.057895 | 0.003162 | 0.003162 | 0.002284 | 0.002284 |
| 14 | 0.061053 | 0.003161 | 0.003161 | 0.002281 | 0.002281 |
| 15 | 0.064211 | 0.003161 | 0.003161 | 0.002295 | 0.002295 |
| 16 | 0.067368 | 0.003160 | 0.003160 | 0.002286 | 0.002286 |
| 17 | 0.070526 | 0.003159 | 0.003159 | 0.002275 | 0.002275 |
| 18 | 0.073684 | 0.003160 | 0.003160 | 0.002269 | 0.002269 |
| 19 | 0.076842 | 0.003161 | 0.003161 | 0.002276 | 0.002276 |
| 20 | 0.080000 | 0.003161 | 0.003161 | 0.002273 | 0.002273 |

## Device speed comparison

- alternating: CPU/GPU fit-time ratio 1.946x (above 1 favors GPU).
  Selected-model CPU/GPU RMSE difference: 1.44e-14.
- joint: CPU/GPU fit-time ratio 2.325x (above 1 favors GPU).
  Selected-model CPU/GPU RMSE difference: 3.89e-15.

## Interpretation

- alternating: test RMSE 0.62151, R2 0.53524; mean relative input-variance error 92.56%. Estimated observation std. 0.02944 versus true 0.1.
- joint: test RMSE 0.62177, R2 0.53484; mean relative input-variance error 94.63%. Estimated observation std. 0.02790 versus true 0.1.
Both methods have similar prediction accuracy on this dataset, but poor recovery of coordinate-wise input noise. This run does not establish convergence or identifiability; larger samples, more optimization and repeated datasets would be separate experiments.

## All restart outcomes

| Device | Method | Restart | Status | Fit (s) | Final loss |
|---|---|---:|---|---:|---:|
| cpu | alternating | 1 | completed | 80.820 | 118.954031 |
| cpu | alternating | 2 | completed | 83.360 | 118.982917 |
| cpu | alternating | 3 | completed | 94.093 | 118.968860 |
| cuda | alternating | 1 | completed | 40.842 | 118.954031 |
| cuda | alternating | 2 | completed | 44.112 | 118.982917 |
| cuda | alternating | 3 | completed | 47.787 | 118.968860 |
| cpu | joint | 1 | completed | 56.547 | 118.761506 |
| cpu | joint | 2 | completed | 55.062 | 118.764371 |
| cpu | joint | 3 | completed | 53.000 | 118.765058 |
| cuda | joint | 1 | completed | 23.327 | 118.761506 |
| cuda | joint | 2 | completed | 23.453 | 118.764371 |
| cuda | joint | 3 | completed | 24.008 | 118.765058 |

Raw data, histories, initializations, estimates, predictions, per-run timings, and source hashes are retained in simulation_20d_results.json. V is a variational parameter, not a direct estimate of the true input variance. With 20 input variances and only 100 observations, noise recovery may be weak even when predictions are useful. CPU/GPU comparisons apply to these workloads and this single-thread CPU configuration.
