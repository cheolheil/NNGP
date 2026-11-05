import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import math
import platform
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from nngp_delta import kernel_delta, expected_log_likelihood_delta

from nngp_torch import GenArccos, NNGP_EIV, get_environment_info, set_seed


class NNGP_EIV_Joint(NNGP_EIV):
    """Compatibility wrapper for NNGP_EIV(inference_method="joint")."""

    def __init__(self, kernel, input_dim, S_u_init=1e-3, sigma_obs_init=0.1, dtype=torch.float64):
        super().__init__(kernel, input_dim, S_u_init=S_u_init,
                         sigma_obs_init=sigma_obs_init, dtype=dtype,
                         inference_method="joint")

    def fit(self, X, y, n_restart=1, iter_warmup=25, iter_joint=75, lr=0.05):
        return super().fit(X, y, n_restart=n_restart, inference_method="joint",
                           iter_warmup=iter_warmup, iter_joint=iter_joint, lr=lr)

# ==========================================
# 3. TOY PROBLEM EXECUTION (BERKSON ERROR - JOINT OPTIMIZATION)
# ==========================================
def run_berkson_toy_problem(input_dim=1, n_restart=3, show_plot=True, save_plot=None):
    device, dtype, env_info = get_environment_info()
    set_seed(42, device)
    
    print("=" * 50)
    print(" CROSS-PLATFORM PYTORCH DIAGNOSTICS (JOINT OPTIMIZATION)")
    print("=" * 50)
    for k, v in env_info.items():
        print(f"  {k:<20}: {v}")
    print(f"  {'Problem Dimension':<20}: {input_dim}-D")
    print(f"  {'Restarts (n_restart)':<20}: {n_restart}")
    print("=" * 50 + "\n")

    if input_dim == 1:
        N = 60
        TRUE_S_U = torch.tensor([0.05], dtype=dtype, device=device)
        X_nom = torch.linspace(-4, 4, N, dtype=dtype, device=device).unsqueeze(1)
        u_noise = torch.randn_like(X_nom) * torch.sqrt(TRUE_S_U)
        X_phys = X_nom + u_noise
        y = -0.25 * (X_phys**2) + 2.0 + (torch.randn_like(X_phys) * 0.1)

        X_test = torch.linspace(-5, 5, 100, dtype=dtype, device=device).unsqueeze(1)
        y_test_true = -0.25 * (X_test**2) + 2.0

    elif input_dim == 4:
        N = 100
        TRUE_S_U = torch.tensor([0.05, 0.03, 0.08, 0.04], dtype=dtype, device=device)
        X_nom = (torch.rand(N, 4, dtype=dtype, device=device) * 6.0 - 3.0)
        u_noise = torch.randn_like(X_nom) * torch.sqrt(TRUE_S_U)
        X_phys = X_nom + u_noise

        c_weights = torch.tensor([1.0, 0.8, 1.2, 0.5], dtype=dtype, device=device)
        y_quad = -0.25 * torch.sum((X_phys**2) * c_weights, dim=1, keepdim=True)
        y_trig = 0.5 * torch.sin(X_phys[:, 0:1]) + 0.5 * torch.cos(X_phys[:, 1:2])
        y = y_quad + y_trig + 2.0 + (torch.randn(N, 1, dtype=dtype, device=device) * 0.1)

        N_test = 200
        X_test = (torch.rand(N_test, 4, dtype=dtype, device=device) * 7.0 - 3.5)
        y_test_quad = -0.25 * torch.sum((X_test**2) * c_weights, dim=1, keepdim=True)
        y_test_trig = 0.5 * torch.sin(X_test[:, 0:1]) + 0.5 * torch.cos(X_test[:, 1:2])
        y_test_true = y_test_quad + y_test_trig + 2.0

    else:
        raise ValueError(f"Unsupported input dimension: {input_dim}. Choose 1 or 4.")

    kernel = GenArccos(input_dim=input_dim, n_layers=2, s_b=[0.5, 0.5], s_w=[2.0, 1.5], dtype=dtype).to(device)
    S_u_init_val = 0.01 if input_dim == 1 else [0.01, 0.015, 0.008, 0.012]
    model = NNGP_EIV_Joint(kernel, input_dim=input_dim, S_u_init=S_u_init_val, sigma_obs_init=0.1, dtype=dtype).to(device)

    model.fit(X_nom, y, n_restart=n_restart, iter_warmup=25, iter_joint=75, lr=0.05)

    mu, var = model.predict(X_test)
    std = torch.sqrt(var).cpu().numpy().flatten()
    
    mu_np = mu.cpu().numpy().flatten()
    y_true_np = y_test_true.cpu().numpy().flatten()

    rmse = math.sqrt(float(torch.mean((mu - y_test_true)**2).item()))
    print(f"\n>>> Prediction Evaluation (Test RMSE): {rmse:.4f}")
    print(">>> Final Estimated S_u vs Ground Truth:")
    final_Su = model.S_u.detach().cpu().numpy()
    true_Su_np = TRUE_S_U.cpu().numpy()
    for d_idx in range(input_dim):
        print(f"    Dim {d_idx+1}: Estimated S_u = {final_Su[d_idx]:.4f} | True S_u = {true_Su_np[d_idx]:.4f}")

    plt.figure(figsize=(13, 5))

    if input_dim == 1:
        X_test_np = X_test.cpu().numpy().flatten()
        plt.subplot(1, 2, 1)
        plt.scatter(X_nom.cpu(), y.cpu(), c='red', alpha=0.5, label='Noisy Obs')
        plt.plot(X_test_np, y_true_np, 'k--', label='True Latent Func')
        plt.plot(X_test_np, mu_np, 'b-', linewidth=2, label='NNGP Mean')
        plt.fill_between(X_test_np, mu_np - 1.96*std, mu_np + 1.96*std, color='blue', alpha=0.2, label='95% CI')
        plt.title("Function Fit with 1D Berkson Error (Joint)")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
    else:
        plt.subplot(1, 2, 1)
        plt.scatter(y_true_np, mu_np, c='blue', alpha=0.6, edgecolors='k', label='Test Points')
        min_val = min(y_true_np.min(), mu_np.min())
        max_val = max(y_true_np.max(), mu_np.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal 1:1 Line')
        plt.title(f"4D Function Fit (Test RMSE: {rmse:.3f})")
        plt.xlabel("True Latent Output y")
        plt.ylabel("NNGP Predicted Mean ŷ")
        plt.legend()

    plt.subplot(1, 2, 2)
    colors = ['g', 'm', 'c', 'orange']
    for d_idx in range(input_dim):
        Su_history = [s[d_idx] for s in model.history['S_u']]
        c = colors[d_idx % len(colors)]
        plt.axhline(true_Su_np[d_idx], color=c, linestyle='--', alpha=0.7,
                    label=f'True S_u[{d_idx+1}] ({true_Su_np[d_idx]:.2f})')
        plt.plot(range(len(Su_history)), Su_history, color=c, linewidth=2,
                 label=f'Est S_u[{d_idx+1}] (Final: {Su_history[-1]:.3f})')
                 
    plt.title(f"Input Noise Inference (Joint {input_dim}-D)")
    plt.xlabel("Joint Training Iteration")
    plt.ylabel("Variance (S_u)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') if input_dim > 1 else plt.legend()

    plt.tight_layout()
    if save_plot:
        os.makedirs(os.path.dirname(os.path.abspath(save_plot)), exist_ok=True)
        plt.savefig(save_plot, dpi=300, bbox_inches='tight')
        print(f">>> Plot saved to: {save_plot}")
    if show_plot:
        plt.show()
    plt.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch NNGP-EIV Joint Optimization Toy Problem")
    parser.add_argument("dim_pos", type=int, nargs="?", default=None, help="Dimension of the toy problem (1 or 4)")
    parser.add_argument("--dim", "-d", type=int, default=1, choices=[1, 4], help="Dimension of the toy problem (1 or 4). Default: 1")
    parser.add_argument("--n-restart", "-n", type=int, default=3, help="Number of multi-restart optimizations (default: 3)")
    parser.add_argument("--no-plot", action="store_true", help="Do not display plot GUI")
    parser.add_argument("--save-plot", type=str, default=None, help="Path to save generated plot image")
    
    args = parser.parse_args()
    target_dim = args.dim_pos if args.dim_pos is not None else args.dim
    if target_dim not in [1, 4]:
        parser.error("Dimension must be 1 or 4.")
        
    run_berkson_toy_problem(input_dim=target_dim, n_restart=args.n_restart, show_plot=not args.no_plot, save_plot=args.save_plot)
