# Unified PyTorch inference

Both methods are available through `NNGP_EIV` in `nngp_torch.py`:

```python
from nngp_torch import GenArccos, NNGP_EIV, get_environment_info

device, dtype, info = get_environment_info()
kernel = GenArccos(1, 2, [0.5, 0.5], [2.0, 1.5], dtype=dtype)
model = NNGP_EIV(kernel, 1, dtype=dtype, inference_method="joint").to(device)
# X and y must use the same device and dtype as the model.
model.fit(X, y, iter_warmup=25, iter_joint=75, lr=0.05)
mean, variance = model.predict(X_test)
```

Alternatively, choose the method when fitting:

```python
model = NNGP_EIV(kernel, 1, dtype=dtype).to(device)
model.fit(X, y, inference_method="alternating",
          outer_iters=15, inner_kernel_steps=10, inner_su_steps=10,
          lr_kernel=0.03, lr_su=0.03)
# A later fit can select the other method, starting from current parameters.
model.fit(X, y, inference_method="joint", iter_warmup=25, iter_joint=75)
```

`alternating` is the default and minimizes the negative approximate ELBO,
including the input KL and variational variance V. `joint` minimizes the moment
Gaussian NLL, optimizing kernel parameters, observation noise, and S_u; V is
unused. These remain different approximation objectives, rather than two
optimizers for the same objective. Prediction uses V for alternating training
moments and S_u for joint training moments.

A fit-time selection persists for subsequent fits, forward calls, and prediction.
`fit` returns the model. Optimizer settings apply only to their respective method.
When restoring a `state_dict`, construct the model with the same inference method;
the method is a constructor option, not a tensor stored in the state dictionary.

`NNGP_EIV_Joint` and its positional constructor/fit arguments remain supported
through a compatibility wrapper in `nngp_torch_joint.py`. Existing joint tensor
state dictionaries can be loaded with `strict=False` because the unified model
also owns an unused `log_V` parameter.

Both toy scripts retain automatic CUDA / MPS / CPU selection. Direct library use
requires moving the model and data explicitly, as above. The comparison benchmark
uses the unified interface while retaining its CPU/float64 protocol.
