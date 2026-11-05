# 20D: input-noise inference versus naive NNGP

## Design

100 training observations, 200 exact-input latent test targets, seed 42, float64. Nominal/test inputs, latent function, observation-noise draws, and standardized input-error draws are paired across scenarios. The latent function is unchanged from simulation_20d.py.
True marginal input variances increase linearly from 0.02 to 0.08. True observation standard deviation is 0.1.
Diagonal scenario: independent Gaussian Berkson errors. Tridiagonal scenario: adjacent-coordinate correlation 0.35 and all other off-diagonal correlations zero; covariance entries Sigma[d,d+1]=0.35 sqrt(v[d]v[d+1]). This covariance is positive definite. Both EIV fits retain the diagonal assumption.
Alternating minimizes negative approximate ELBO; joint minimizes moment Gaussian NLL. Their losses must not be compared across methods. Kernel: two-layer GenArccos, fixed equal input weights, initial biases [0.5,0.5], weights [2,1.5], observation std. 0.1.
Naive NNGP uses K(X,X)+sigma_obs^2 I and standard Gaussian NLL, optimizing only kernel parameters and observation noise. It has no input-noise parameters, KL, or Delta correction. Latent predictions exclude observation noise for all methods.
Alternating has 300 updates at lr 0.03 (15 outer iterations, 10 kernel/noise + 10 variance steps each). Joint has 25 warmup updates at lr 0.05 + 75 joint updates at lr 0.025. Naive-100 uses the same 25+75 schedule; naive-300 is a budget sensitivity check with 300 updates at lr 0.03. All settings were fixed before observing test errors.
Three EIV starts perturb S_u (and V for alternating) exactly as in the earlier study; kernel/noise initializations are identical. Naive has no input variances to perturb, so its three runs are deterministic timing repeats from the same initialization, not three distinct optimization starts. Selection uses final training loss only.
CUDA timings synchronize the GPU; CPU uses one thread. Fit totals cover three runs and include prediction-cache construction. Total measured = fit total + one selected-model prediction; transfers, construction, warmup, diagnostics and report generation are excluded. Timing repeats are not Monte Carlo datasets.
The diagonal EIV results/timings are reused from the earlier study after exact data and core-source-hash checks. Other results are newly measured. This is a single dataset in each setting; it cannot establish general superiority or statistical significance.
Tridiagonal misspecification evaluates this particular correlation pattern and response. The response has no mixed quadratic terms, limiting the conclusions about other correlation-sensitive functions.

Environment: {'python': '3.10.20', 'torch': '2.12.0+cu132', 'cpu': 'Intel64 Family 6 Model 151 Stepping 2, GenuineIntel', 'gpu': 'NVIDIA GeForce RTX 3080', 'dtype': 'float64', 'cpu_threads': 1}

## Diagonal true input covariance

| Device | Method | Runs | Fit total (s) | Predict (s) | Total measured (s) | RMSE | MAE | R2 | RMSE gain vs naive-100 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cpu | alternating | 3/3 | 258.274 | 1.395 | 259.668 | 0.621508 | 0.474868 | 0.535239 | -0.005% |
| cuda | alternating | 3/3 | 132.742 | 0.366 | 133.107 | 0.621508 | 0.474868 | 0.535239 | -0.005% |
| cpu | joint | 3/3 | 164.610 | 1.166 | 165.775 | 0.621772 | 0.475070 | 0.534843 | -0.048% |
| cuda | joint | 3/3 | 70.788 | 0.155 | 70.942 | 0.621772 | 0.475070 | 0.534843 | -0.048% |
| cpu | naive_100 | 3/3 | 0.847 | 0.004 | 0.851 | 0.621475 | 0.474874 | 0.535287 | 0.000% |
| cuda | naive_100 | 3/3 | 1.686 | 0.060 | 1.745 | 0.621475 | 0.474874 | 0.535287 | 0.000% |
| cpu | naive_300 | 3/3 | 2.215 | 0.002 | 2.216 | 0.618774 | 0.472907 | 0.539317 | 0.435% |
| cuda | naive_300 | 3/3 | 7.774 | 0.004 | 7.777 | 0.618774 | 0.472907 | 0.539317 | 0.435% |

Positive RMSE gain favors the listed method; negative values favor naive-100.

### Noise estimation and latent intervals

| Device | Method | Marginal S_u RMSE | Relative marginal error | Full covariance relative Frobenius error | Observation std. | 95% coverage | Mean interval width |
|---|---|---:|---:|---:|---:|---:|---:|
| cpu | alternating | 0.050254 | 92.56% | 94.44% | 0.029443 | 95.0% | 2.298871 |
| cuda | alternating | 0.050254 | 92.56% | 94.44% | 0.029443 | 95.0% | 2.298871 |
| cpu | joint | 0.051076 | 94.63% | 95.98% | 0.027904 | 95.0% | 2.289002 |
| cuda | joint | 0.051076 | 94.63% | 95.98% | 0.027904 | 95.0% | 2.289002 |
| cpu | naive_100 | -- | -- | 100.00% | 0.028940 | 95.0% | 2.290845 |
| cuda | naive_100 | -- | -- | 100.00% | 0.028940 | 95.0% | 2.290845 |
| cpu | naive_300 | -- | -- | 100.00% | 0.015179 | 95.0% | 2.286884 |
| cuda | naive_300 | -- | -- | 100.00% | 0.015179 | 95.0% | 2.286884 |

Naive assumes zero input covariance (100% covariance error by definition); marginal parameter recovery is not applicable. Full covariance error includes omitted off-diagonal entries for diagonal EIV fits. Coverage is descriptive for one fixed test set.

### Comparison to the benchmark

- cpu alternating: RMSE difference from naive-100 +0.000033; relative improvement -0.005%.
  Against naive-300: relative improvement -0.442%.
- cpu joint: RMSE difference from naive-100 +0.000297; relative improvement -0.048%.
  Against naive-300: relative improvement -0.484%.
- cuda alternating: RMSE difference from naive-100 +0.000033; relative improvement -0.005%.
  Against naive-300: relative improvement -0.442%.
- cuda joint: RMSE difference from naive-100 +0.000297; relative improvement -0.048%.
  Against naive-300: relative improvement -0.484%.

### All fit outcomes

| Device | Method | Run | Status | Fit seconds | Final training loss |
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
| cpu | naive_100 | 1 | completed | 0.242 | 118.735507 |
| cpu | naive_100 | 2 | completed | 0.247 | 118.735507 |
| cpu | naive_100 | 3 | completed | 0.358 | 118.735507 |
| cuda | naive_100 | 1 | completed | 0.575 | 118.735507 |
| cuda | naive_100 | 2 | completed | 0.552 | 118.735507 |
| cuda | naive_100 | 3 | completed | 0.560 | 118.735507 |
| cpu | naive_300 | 1 | completed | 0.771 | 118.477015 |
| cpu | naive_300 | 2 | completed | 0.723 | 118.477015 |
| cpu | naive_300 | 3 | completed | 0.721 | 118.477015 |
| cuda | naive_300 | 1 | completed | 2.137 | 118.477015 |
| cuda | naive_300 | 2 | completed | 2.679 | 118.477015 |
| cuda | naive_300 | 3 | completed | 2.957 | 118.477015 |

## Tridiagonal true input covariance

| Device | Method | Runs | Fit total (s) | Predict (s) | Total measured (s) | RMSE | MAE | R2 | RMSE gain vs naive-100 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cpu | alternating | 3/3 | 374.790 | 1.818 | 376.608 | 0.620203 | 0.469193 | 0.537187 | -0.000% |
| cuda | alternating | 3/3 | 256.296 | 0.324 | 256.620 | 0.620203 | 0.469193 | 0.537187 | -0.000% |
| cpu | joint | 3/3 | 284.225 | 1.920 | 286.145 | 0.620470 | 0.469436 | 0.536790 | -0.043% |
| cuda | joint | 3/3 | 168.013 | 0.351 | 168.364 | 0.620470 | 0.469436 | 0.536790 | -0.043% |
| cpu | naive_100 | 3/3 | 1.613 | 0.004 | 1.617 | 0.620203 | 0.469346 | 0.537188 | 0.000% |
| cuda | naive_100 | 3/3 | 2.660 | 0.003 | 2.663 | 0.620203 | 0.469346 | 0.537188 | 0.000% |
| cpu | naive_300 | 3/3 | 4.674 | 0.005 | 4.679 | 0.617518 | 0.467319 | 0.541187 | 0.433% |
| cuda | naive_300 | 3/3 | 7.670 | 0.003 | 7.673 | 0.617518 | 0.467319 | 0.541187 | 0.433% |

Positive RMSE gain favors the listed method; negative values favor naive-100.

### Noise estimation and latent intervals

| Device | Method | Marginal S_u RMSE | Relative marginal error | Full covariance relative Frobenius error | Observation std. | 95% coverage | Mean interval width |
|---|---|---:|---:|---:|---:|---:|---:|
| cpu | alternating | 0.050254 | 92.56% | 95.50% | 0.027791 | 95.5% | 2.298221 |
| cuda | alternating | 0.050254 | 92.56% | 95.50% | 0.027791 | 95.5% | 2.298221 |
| cpu | joint | 0.051079 | 94.64% | 96.75% | 0.026711 | 95.5% | 2.288352 |
| cuda | joint | 0.051079 | 94.64% | 96.75% | 0.026711 | 95.5% | 2.288352 |
| cpu | naive_100 | -- | -- | 100.00% | 0.027700 | 95.5% | 2.290147 |
| cuda | naive_100 | -- | -- | 100.00% | 0.027700 | 95.5% | 2.290147 |
| cpu | naive_300 | -- | -- | 100.00% | 0.014479 | 95.5% | 2.286137 |
| cuda | naive_300 | -- | -- | 100.00% | 0.014479 | 95.5% | 2.286137 |

Naive assumes zero input covariance (100% covariance error by definition); marginal parameter recovery is not applicable. Full covariance error includes omitted off-diagonal entries for diagonal EIV fits. Coverage is descriptive for one fixed test set.

### Comparison to the benchmark

- cpu alternating: RMSE difference from naive-100 +0.000000; relative improvement -0.000%.
  Against naive-300: relative improvement -0.435%.
- cpu joint: RMSE difference from naive-100 +0.000267; relative improvement -0.043%.
  Against naive-300: relative improvement -0.478%.
- cuda alternating: RMSE difference from naive-100 +0.000000; relative improvement -0.000%.
  Against naive-300: relative improvement -0.435%.
- cuda joint: RMSE difference from naive-100 +0.000267; relative improvement -0.043%.
  Against naive-300: relative improvement -0.478%.

### All fit outcomes

| Device | Method | Run | Status | Fit seconds | Final training loss |
|---|---|---:|---|---:|---:|
| cpu | alternating | 1 | completed | 141.550 | 118.917847 |
| cpu | alternating | 2 | completed | 102.144 | 118.945964 |
| cpu | alternating | 3 | completed | 131.096 | 118.929792 |
| cuda | alternating | 1 | completed | 84.612 | 118.917847 |
| cuda | alternating | 2 | completed | 85.766 | 118.945964 |
| cuda | alternating | 3 | completed | 85.919 | 118.929792 |
| cpu | joint | 1 | completed | 92.084 | 118.724976 |
| cpu | joint | 2 | completed | 96.087 | 118.727979 |
| cpu | joint | 3 | completed | 96.054 | 118.728765 |
| cuda | joint | 1 | completed | 57.214 | 118.724976 |
| cuda | joint | 2 | completed | 55.706 | 118.727979 |
| cuda | joint | 3 | completed | 55.093 | 118.728765 |
| cpu | naive_100 | 1 | completed | 0.512 | 118.697573 |
| cpu | naive_100 | 2 | completed | 0.533 | 118.697573 |
| cpu | naive_100 | 3 | completed | 0.569 | 118.697573 |
| cuda | naive_100 | 1 | completed | 0.912 | 118.697573 |
| cuda | naive_100 | 2 | completed | 0.875 | 118.697573 |
| cuda | naive_100 | 3 | completed | 0.873 | 118.697573 |
| cpu | naive_300 | 1 | completed | 1.510 | 118.441166 |
| cpu | naive_300 | 2 | completed | 1.568 | 118.441166 |
| cpu | naive_300 | 3 | completed | 1.596 | 118.441166 |
| cuda | naive_300 | 1 | completed | 2.542 | 118.441166 |
| cuda | naive_300 | 2 | completed | 2.560 | 118.441166 |
| cuda | naive_300 | 3 | completed | 2.568 | 118.441166 |

## Reproducibility

Run `python simulation_20d_naive.py` from the project root. Completed method/device groups are resumed only when the saved experiment signature matches code, configuration and source diagonal results. Raw data, true covariance matrices, initializations, histories, predictions and source hashes are saved in simulation_20d_naive_results.json. The original simulation_20d_results.json and report are preserved.

## Findings for these runs

Neither input-noise method improves latent test RMSE over naive-100 in either
scenario. Alternating essentially ties it (diagonal RMSE 0.621508 versus
0.621475; tridiagonal 0.620203283 versus 0.620203091). Joint is slightly worse.
Naive-300 has the lowest RMSE: 0.618774 under diagonal noise and 0.617518 under
tridiagonal noise. It also costs substantially less to fit.

Marginal input-variance recovery remains poor (about 92.6% relative error for
alternating and 94.6% for joint). These experiments therefore do not support
the intended superiority claim under the current data generator, sample size,
kernel, initialization and optimizer budgets. This does not establish that
EIV cannot help under other settings or after further optimization. A claim
of systematic improvement would require paired repeated-dataset experiments.

All 36 newly executed fits succeeded. The report also contains 12 previously
measured diagonal EIV fits. All 16 selected-model predictions succeeded.
Matched CPU/GPU training losses differ by at most 3.84e-13. Earlier diagonal
EIV timings and newly measured timings come from different sessions; they
should not be used to attribute timing changes solely to covariance structure.
