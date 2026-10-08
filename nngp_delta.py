"""Second-order Delta approximations for independent diagonal input errors."""
import math
import torch


def diagonal_hessian_trace(value, inputs, variances, create_graph):
    """Trace of the input Hessian, retaining parameter gradients when requested."""
    first = torch.autograd.grad(value, inputs, create_graph=True, allow_unused=True)[0]
    if first is None or not first.requires_grad:
        return value * 0.0
    terms = []
    for i in range(inputs.shape[0]):
        for d in range(inputs.shape[1]):
            component = first[i, d]
            if component.requires_grad:
                second = torch.autograd.grad(component, inputs, create_graph=create_graph,
                                             retain_graph=True, allow_unused=True)[0]
                if second is not None:
                    terms.append(second[i, d] * variances[d])
    return sum(terms, value * 0.0)


def kernel_delta(kernel, X, variance_x, Y=None, variance_y=None):
    """Expected-kernel correction. Y=None shares each diagonal input realization.

    Supplying Y explicitly means independent realizations even for equal coordinates.
    Variances are per-coordinate variances, not standard deviations.
    """
    auto = Y is None
    Y = X if auto else Y
    variance_y = variance_x if variance_y is None else variance_y
    outer_grad = torch.is_grad_enabled()
    with torch.enable_grad():
        a = X[:, None, :].expand(-1, len(Y), -1).reshape(-1, X.shape[1]).detach().clone().requires_grad_(True)
        b = Y[None, :, :].expand(len(X), -1, -1).reshape(-1, Y.shape[1]).detach().clone().requires_grad_(True)
        vals = kernel.paired(a, b)
        correction = torch.zeros_like(vals)
        for inputs, variance in ((a, variance_x), (b, variance_y)):
            first = torch.autograd.grad(vals.sum(), inputs, create_graph=True, retain_graph=True)[0]
            for d in range(inputs.shape[1]):
                if first[:, d].requires_grad:
                    second = torch.autograd.grad(first[:, d].sum(), inputs,
                        create_graph=outer_grad, retain_graph=True, allow_unused=True)[0]
                    if second is not None:
                        correction = correction + 0.5 * second[:, d] * variance[d]
        correction = correction.reshape(len(X), len(Y))
        if auto:
            z = X.detach().clone().requires_grad_(True)
            diag_values = kernel.diagonal(z)
            first = torch.autograd.grad(diag_values.sum(), z, create_graph=True)[0]
            diagonal = torch.zeros_like(diag_values)
            for d in range(z.shape[1]):
                if first[:, d].requires_grad:
                    second = torch.autograd.grad(first[:, d].sum(), z,
                        create_graph=outer_grad, retain_graph=True, allow_unused=True)[0]
                    if second is not None:
                        diagonal = diagonal + 0.5 * second[:, d] * variance_x[d]
            correction = correction - torch.diag_embed(correction.diagonal()) + torch.diag_embed(diagonal)
            correction = 0.5 * (correction + correction.T)
    return correction if outer_grad else correction.detach()


def expected_log_likelihood_delta(kernel, X, y, variance, sigma):
    """Taylor approximation to E_q[log N(y; 0, K(W)+sigma**2 I)].

    This approximates the ELBO expectation; Taylor truncation is not itself
    guaranteed to preserve the lower-bound property.
    """
    outer_grad = torch.is_grad_enabled()
    with torch.enable_grad():
        z = X.detach().clone().requires_grad_(True)
        covariance = kernel(z) + sigma.square() * torch.eye(len(X), dtype=X.dtype, device=X.device)
        L = torch.linalg.cholesky(covariance)
        log_lik = -0.5 * (y * torch.cholesky_solve(y, L)).sum() - L.diagonal().log().sum() - 0.5 * len(X) * math.log(2 * math.pi)
        result = log_lik + 0.5 * diagonal_hessian_trace(log_lik, z, variance, outer_grad)
    return result if outer_grad else result.detach()
