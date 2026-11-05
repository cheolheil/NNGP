# Legacy implementations

These modules are retained for reference and future deprecation. No code has
been deleted. The supported unified implementation is in ../nngp_torch.py and
../nngp_delta.py.

- nngp.py, nngp_eiv.py, kernel.py: earlier NumPy/SciPy implementations.
- doe.py, function_generator.py: associated standalone utilities.
- nngp_torch_joint.py: compatibility wrapper and joint toy example; the joint
  inference implementation now lives in the unified NNGP_EIV class.

From the repository root, use `from legacy.nngp_torch_joint import NNGP_EIV_Joint`
or run `python -m legacy.nngp_torch_joint`. Earlier top-level imports must be
updated to the legacy namespace. Prefer NNGP_EIV(..., inference_method="joint")
for new code.
