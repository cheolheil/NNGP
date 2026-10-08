import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import math
import platform
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ==========================================
# 1. THE PYTORCH ARCCOSINE KERNEL
# ==========================================
class GenArccos(nn.Module):
    def __init__(self, input_dim, n_layers, s_b, s_w, D=None, dtype=torch.float64):
        super().__init__()
        self.input_dim = input_dim
        self.n_layers = n_layers
        self.dtype = dtype
        
        s_b = torch.atleast_1d(torch.tensor(s_b, dtype=dtype))
        s_w = torch.atleast_1d(torch.tensor(s_w, dtype=dtype))
        
        if D is None:
            self._D_fixed = True
            self.register_buffer('D', torch.ones(input_dim, dtype=dtype) / input_dim)
            self.log_s_b = nn.Parameter(torch.log(s_b))
            self.log_s_w = nn.Parameter(torch.log(s_w))
        else:
            D = torch.atleast_1d(torch.tensor(D, dtype=dtype))
            self._D_fixed = False
            self.log_s_b = nn.Parameter(torch.log(s_b))
            self.log_s_w = nn.Parameter(torch.log(s_w[1:])) 
            self.log_D = nn.Parameter(torch.log(D))

    @property
    def s_b(self): return torch.exp(self.log_s_b)
    @property
    def s_w(self): return torch.exp(self.log_s_w)
    @property
    def D_param(self): return self.D if self._D_fixed else torch.exp(self.log_D)

    def _init_layer(self, X, Y):
        if self._D_fixed:
            return self.s_w[0] * ((X * self.D_param) @ Y.T) + self.s_b[0]
        return ((X * self.D_param) @ Y.T) + self.s_b[0]

    def forward(self, X, Y=None):
        is_auto = Y is None
        if is_auto: Y = X
        
        K = self._init_layer(X, Y)
        
        if self.n_layers > 1:
            if is_auto:
                K_XX = K
            else:
                K_XX = self._init_layer(X, X)
                K_YY = self._init_layer(Y, Y)
                
            for l in range(1, self.n_layers):
                s_w_l = self.s_w[l] if self._D_fixed else self.s_w[l-1]
                s_b_l = self.s_b[l]

                if is_auto:
                    k_diag = torch.diag(K_XX)
                    sqrt_K_outer = torch.sqrt(torch.outer(k_diag, k_diag))
                    cos_theta = torch.clamp(K_XX / sqrt_K_outer, -1.0 + 1e-7, 1.0 - 1e-7)
                    theta_mat = torch.acos(cos_theta)
                    J1 = torch.sin(theta_mat) + (math.pi - theta_mat) * cos_theta
                    K_XX = (0.5 / math.pi) * s_w_l * sqrt_K_outer * J1 + s_b_l
                    K = K_XX 
                else:
                    kX_diag = torch.diag(K_XX)
                    kY_diag = torch.diag(K_YY)
                    sqrt_KX_outer = torch.sqrt(torch.outer(kX_diag, kX_diag))
                    sqrt_KY_outer = torch.sqrt(torch.outer(kY_diag, kY_diag))
                    sqrt_K_outer = torch.sqrt(torch.outer(kX_diag, kY_diag))
                    
                    cos_thetaX = torch.clamp(K_XX / sqrt_KX_outer, -1.0 + 1e-7, 1.0 - 1e-7)
                    cos_thetaY = torch.clamp(K_YY / sqrt_KY_outer, -1.0 + 1e-7, 1.0 - 1e-7)
                    cos_theta  = torch.clamp(K / sqrt_K_outer, -1.0 + 1e-7, 1.0 - 1e-7)
                    
                    thetaX_mat = torch.acos(cos_thetaX)
                    thetaY_mat = torch.acos(cos_thetaY)
                    theta_mat  = torch.acos(cos_theta)
                    
                    J1_X = torch.sin(thetaX_mat) + (math.pi - thetaX_mat) * cos_thetaX
                    J1_Y = torch.sin(thetaY_mat) + (math.pi - thetaY_mat) * cos_thetaY
                    J1   = torch.sin(theta_mat)  + (math.pi - theta_mat)  * cos_theta
                    
                    K_XX = (0.5 / math.pi) * s_w_l * sqrt_KX_outer * J1_X + s_b_l
                    K_YY = (0.5 / math.pi) * s_w_l * sqrt_KY_outer * J1_Y + s_b_l
                    K    = (0.5 / math.pi) * s_w_l * sqrt_K_outer  * J1   + s_b_l

        return K

# ==========================================
# 2. PYTORCH NNGP-EIV (JOINT OPTIMIZATION CLASS)
# ==========================================
class NNGP_EIV_Joint(nn.Module):
    def __init__(self, kernel, input_dim, S_u_init=1e-3, sigma_obs_init=0.1, dtype=torch.float64):
        super().__init__()
        self.kernel = kernel
        self.input_dim = input_dim
        self.dtype = dtype
        
        if isinstance(S_u_init, (int, float)):
            S_u_tensor = torch.full((input_dim,), S_u_init, dtype=dtype)
        else:
            S_u_tensor = torch.as_tensor(S_u_init, dtype=dtype)

        self.log_S_u = nn.Parameter(torch.log(S_u_tensor))
        self.log_sigma_obs = nn.Parameter(torch.tensor(math.log(sigma_obs_init), dtype=dtype))
        
        self.history = {'loss': [], 'S_u': []}

    @property
    def S_u(self): return torch.exp(self.log_S_u)
    @property
    def sigma_obs(self): return torch.exp(self.log_sigma_obs)

    def _compute_delta_correction(self, X):
        N, D = X.shape
        S_u_curr = self.S_u

        with torch.enable_grad():
            X_req = X.detach().clone().requires_grad_(True)
            K_diag = torch.diag(self.kernel(X_req))

            g1 = torch.autograd.grad(K_diag.sum(), X_req, create_graph=True)[0]

            second_diag = torch.zeros_like(X_req)
            for d in range(D):
                g2_d = torch.autograd.grad(g1[:, d].sum(), X_req, create_graph=True if self.training else False)[0][:, d]
                second_diag[:, d] = g2_d

            point_corrections = 0.5 * torch.sum(second_diag * S_u_curr, dim=1)
        return point_corrections.unsqueeze(1) + point_corrections.unsqueeze(0)

    def compute_nll(self, X, y):
        N, D = X.shape
        K_nom = self.kernel(X)
        K_corr = self._compute_delta_correction(X)
        jitter = (self.sigma_obs**2) * torch.eye(N, device=X.device, dtype=X.dtype)
        K_eff = K_nom + K_corr + jitter

        L = torch.linalg.cholesky(K_eff)
        alpha = torch.cholesky_solve(y, L)

        data_fit = 0.5 * torch.sum(y * alpha)
        complexity = torch.sum(torch.log(torch.diag(L))) + 0.5 * N * math.log(2.0 * math.pi)
        return data_fit + complexity

    def forward(self, X, y):
        return self.compute_nll(X, y)

    def fit(self, X, y, n_restart=1, iter_warmup=25, iter_joint=75, lr=0.05):
        self.X_train = X
        self.y_train = y

        base_log_S_u = self.log_S_u.data.clone()
        base_kernel_state = {k: v.clone() for k, v in self.kernel.state_dict().items()}
        base_log_sigma_obs = self.log_sigma_obs.data.clone()

        best_nll = float('inf')
        best_state = None
        best_history = None

        if n_restart > 1:
            print(f">>> Running Multi-Restart Joint NLL Optimization (n_restart = {n_restart})...")

        for restart in range(n_restart):
            if restart == 0:
                self.log_S_u.data.copy_(base_log_S_u)
            else:
                rand_perturb_S = torch.randn_like(base_log_S_u) * 0.5
                self.log_S_u.data.copy_(base_log_S_u + rand_perturb_S)

            self.kernel.load_state_dict(base_kernel_state)
            self.log_sigma_obs.data.copy_(base_log_sigma_obs)

            opt_kernel = torch.optim.Adam(list(self.kernel.parameters()) + [self.log_sigma_obs], lr=lr)
            for epoch in range(iter_warmup):
                opt_kernel.zero_grad()
                loss = self(X, y)
                loss.backward()
                opt_kernel.step()

            opt_joint = torch.optim.Adam(self.parameters(), lr=lr*0.5)
            restart_history = {'loss': [], 'S_u': []}

            for epoch in range(iter_joint):
                opt_joint.zero_grad()
                loss = self(X, y)
                loss.backward()
                opt_joint.step()

                restart_history['loss'].append(loss.item())
                restart_history['S_u'].append(self.S_u.detach().clone().cpu().numpy())

            final_nll = loss.item()
            current_Su = self.S_u.detach().clone().cpu().numpy()
            if n_restart > 1:
                print(f"  [Restart {restart+1}/{n_restart}] Final NLL: {final_nll:.4f} | Est. S_u: {current_Su}")

            if final_nll < best_nll:
                best_nll = final_nll
                best_state = {k: v.clone() for k, v in self.state_dict().items()}
                best_history = restart_history

        self.load_state_dict(best_state)
        self.history = best_history

        if n_restart > 1:
            print(f">>> Best Restart Selected | Lowest NLL: {best_nll:.4f} | Final S_u: {self.S_u.detach().cpu().numpy()}\n")

        with torch.no_grad():
            K_eff = self.kernel(X) + self._compute_delta_correction(X) + (self.sigma_obs**2) * torch.eye(X.shape[0], device=X.device, dtype=X.dtype)
            self.L = torch.linalg.cholesky(K_eff)
            self.alpha = torch.cholesky_solve(y, self.L)

    def predict(self, X_test):
        with torch.no_grad():
            corr_train = self._compute_delta_correction(self.X_train)
            corr_test = self._compute_delta_correction(X_test)
            
            delta_test = 0.5 * torch.diag(corr_test).unsqueeze(1)
            delta_train = 0.5 * torch.diag(corr_train).unsqueeze(0)
            
            K_star = self.kernel(X_test, self.X_train) + delta_test + delta_train
            K_star_star = self.kernel(X_test) + corr_test
            
            mu = K_star @ self.alpha
            v = torch.linalg.solve_triangular(self.L, K_star.T, upper=False)
            
            pred_var = torch.diag(K_star_star) - torch.sum(v**2, dim=0) + (self.sigma_obs**2)
            
            return mu, torch.abs(pred_var)

# ==========================================
# 3. TOY PROBLEM EXECUTION (BERKSON ERROR - JOINT OPTIMIZATION)
# ==========================================
def get_environment_info():
    cuda_avail = torch.cuda.is_available()
    mps_avail = getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()
    
    if cuda_avail:
        device = torch.device("cuda")
        device_name = f"CUDA GPU ({torch.cuda.get_device_name(0)})"
        default_dtype = torch.float64
    elif mps_avail:
        device = torch.device("mps")
        device_name = "Apple Silicon GPU (MPS)"
        default_dtype = torch.float32
    else:
        device = torch.device("cpu")
        device_name = "CPU"
        default_dtype = torch.float64
        
    env_info = {
        "Python Version": platform.python_version(),
        "PyTorch Version": torch.__version__,
        "OS": f"{platform.system()} {platform.release()}",
        "Device": device_name,
        "Precision (dtype)": str(default_dtype).split('.')[-1],
    }
    return device, default_dtype, env_info

def set_seed(seed=42, device=None):
    torch.manual_seed(seed)
    if device is not None and device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    elif device is not None and device.type == 'mps':
        if hasattr(torch.mps, 'manual_seed'):
            torch.mps.manual_seed(seed)

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
