# NNGP-EIV Delta correction

Place all three Python modules and the notebook in the same directory. Run `python test_delta.py` to verify the implementation (PyTorch and matplotlib required).

## Changes

- Joint model: each off-diagonal entry uses half the variance-weighted Hessian trace in each independent input argument. A covariance diagonal uses one half of the Hessian trace of k(x,x), treating the repeated arguments as the same realization.
- Variational model: compute the nominal Gaussian log likelihood plus half the variance-weighted Hessian trace with respect to all training coordinates, then subtract the original Gaussian-input KL. V weights the Taylor correction; S_u enters the KL.
- Arc-cosine kernel: corresponding-row evaluation makes kernel derivative calculations practical. Exact diagonal recursion and the coincident-input second-derivative limit avoid differentiation through acos(1). Clipping occurs only within eight machine epsilons of the correlation endpoints.
- Gradients through the Hessian reach trainable parameters. Evaluation under no_grad works in multiple dimensions without freeing the derivative graph prematurely.
- Prediction: independent pairwise test/train correction, proper shared-input test diagonal. Default test inputs are exact, matching the toy latent-function targets. Pass `input_variance=<scalar or per-coordinate vector>` for noisy test inputs; this is a variance, not a standard deviation. Training moments use V in the variational model and S_u in the joint model. Observation noise remains included by default; set `include_observation_noise=False` for latent predictive variance.
- Negative predictive variances exceeding floating-point tolerance raise an error instead of being reflected by abs(). Tiny roundoff negatives are clamped to zero.
- Restart selection now evaluates the final objective at the saved, post-update parameter state.
- Notebook outputs are cleared so previous incorrect results are not presented as current.

## Interpretation and limitations

The variational expression is a second-order approximation to the ELBO, not a guaranteed lower bound after Taylor truncation. The joint expression is a Gaussian moment approximation, not the exact integrated Berkson likelihood. Prediction also uses a moment approximation rather than full integration over the variational/input posterior. The two methods now approximate different objectives; their benchmark does not isolate optimizer choice.

Second-order kernel truncation need not produce a positive semidefinite covariance at large input variance. Cholesky can therefore fail; the implementation does not silently modify the covariance to conceal this. Small input variance and exact-input, positive-bias kernel configurations are the intended local regime.

The fixed variational input means and existing S_u/V parameterization are retained. This patch does not establish identifiability or guarantee recovery of the true input variance. Exact full-covariance input errors and correlated errors between observations are outside this implementation's scope.

## Verification

Nine CPU tests passed: linear-kernel exact moments; polynomial-kernel analytic diagonal/off-diagonal corrections; recursive arc-cosine finite differences; expected-log-likelihood finite differences; variational and joint parameter gradients; evaluation under no_grad in four dimensions; zero variance; and float32 with trainable input scaling.

Short end-to-end runs also passed with the original problem sizes: N=60 in 1D and N=100 in 4D, for both models. Each run used four optimizer updates, not full convergence. All resulting predictive variances were positive. The variational runs took approximately 2.5 seconds (1D) and 20.7 seconds (4D) on this CPU, including setup and prediction. The joint runs took approximately 0.04 and 0.24 seconds. These are smoke-test timings, not comparable to the previous full benchmark. The original smoke test did not execute the full benchmark. The full three-restart comparison has since been executed via `comparison_benchmark.py`; see `comparison_report.md` and `comparison_report.html` for its results. The optimized likelihood Hessian now avoids one reverse Cholesky pass per input coordinate, while retaining the direct-autograd implementation as a reference.
