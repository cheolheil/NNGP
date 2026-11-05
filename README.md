# NNGP-EIV

The current PyTorch implementation consists of:

- `nngp_torch.py`: kernel and unified NNGP_EIV model, with alternating or joint inference.
- `nngp_delta.py`: shared second-order Delta approximations.

See [INFERENCE.md](INFERENCE.md) for usage and [DELTA_METHOD.md](DELTA_METHOD.md)
for the approximation details. Automatic device selection remains available
through get_environment_info() and the main toy runner.

Earlier implementations and the joint compatibility wrapper are preserved in
[legacy/](legacy/README.md) for future deprecation. Tests, benchmark scripts,
reports, environment setup, and documentation at the repository root are
supporting files, not earlier implementations.

Run checks from the repository root: `python -m unittest discover -q`.

## Dedicated 20D simulation

Run `python simulation_20d.py` for the 100-training-sample CPU/GPU comparison.
See [simulation_20d_report.md](simulation_20d_report.md) for the measured results,
protocol, noise estimates and prediction accuracy. Requires CUDA-enabled PyTorch
for GPU runs; otherwise the script records CPU results only.

## Naive benchmark and covariance misspecification

Run `python simulation_20d_naive.py` to compare alternating and joint diagonal
EIV against observation-noise-only NNGP under diagonal and tridiagonal true
input covariance. Both studies use 20 dimensions and 100 training samples.
The script verifies and reuses the original diagonal EIV results, and resumes
completed method/device groups only when its experiment signature matches.

See [simulation_20d_naive_report.md](simulation_20d_naive_report.md) for the
protocol, prediction accuracy, noise recovery, CPU/GPU timings, and both naive
optimizer budgets. Raw results are in `simulation_20d_naive_results.json`.
The baseline is study code in `simulation_20d_naive.py`; the EIV package model
continues to assume diagonal covariance.
