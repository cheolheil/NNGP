import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ==========================================
# 1. THE PYTORCH ARCCOSINE KERNEL
# ==========================================
class GenArccos(nn.Module):
    def __init__(self, input_dim, n_layers, s_b, s_w, D=None):
        super().__init__()
        self.input_dim = input_dim
        self.n_layers = n_layers
        
        s_b = torch.atleast_1d(torch.tensor(s_b, dtype=torch.float64))
        s_w = torch.atleast_1d(torch.tensor(s_w, dtype=torch.float64))
        
        if D is None:
            self._D_fixed = True
            self.register_buffer('D', torch.ones(input_dim, dtype=torch.float64) / input_dim)
            self.log_s_b = nn.Parameter(torch.log(s_b))
            self.log_s_w = nn.Parameter(torch.log(s_w))
        else:
            D = torch.atleast_1d(torch.tensor(D, dtype=torch.float64))
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
# 2. THE PYTORCH NNGP-EIV CLASS
# ==========================================
class NNGP_EIV_PyTorch(nn.Module):
    def __init__(self, kernel, input_dim, S_u_init=1e-3, sigma_obs_init=0.1):
        super().__init__()
        self.kernel = kernel
        self.input_dim = input_dim
        
        self.log_S_u = nn.Parameter(torch.full((input_dim,), math.log(S_u_init), dtype=torch.float64))
        self.log_sigma_obs = nn.Parameter(torch.tensor(math.log(sigma_obs_init), dtype=torch.float64))
        
        self.history = {'loss': [], 'S_u': []}

    @property
    def S_u(self): return torch.exp(self.log_S_u)
    @property
    def sigma_obs(self): return torch.exp(self.log_sigma_obs)

    def _compute_delta_correction(self, X, n_probes=10):
        N, D = X.shape
        point_corrections = torch.zeros(N, device=X.device, dtype=X.dtype)
        S_u_curr = self.S_u
        
        def f(x_in):
            return self.kernel(x_in)[0, 0]

        for i in range(N):
            x_i = X[i:i+1]
            tr_est = 0.0
            
            # Hutchinson Trace Estimator via HVP
            for _ in range(n_probes):
                z = torch.randint(0, 2, (1, D), device=X.device, dtype=X.dtype) * 2.0 - 1.0
                v = torch.sqrt(S_u_curr) * z
                _, hvp = torch.autograd.functional.hvp(f, x_i, v)
                tr_est += torch.sum(v * hvp)
                
            point_corrections[i] = 0.5 * (tr_est / n_probes)
            
        return point_corrections.unsqueeze(1) + point_corrections.unsqueeze(0)

    def forward(self, X, y):
        K_nom = self.kernel(X)
        K_corr = self._compute_delta_correction(X)
        jitter = (self.sigma_obs**2) * torch.eye(X.shape[0], device=X.device, dtype=X.dtype)
        K_eff = K_nom + K_corr + jitter
        
        L = torch.linalg.cholesky(K_eff)
        alpha = torch.cholesky_solve(y, L)
        
        data_fit = 0.5 * torch.sum(y * alpha)
        complexity = torch.sum(torch.log(torch.diag(L)))
        return data_fit + complexity

    def fit(self, X, y, iter_warmup=20, iter_joint=80, lr=0.05):
        self.X_train = X
        self.y_train = y
        
        print(">>> Phase A: Warm Start (Learning Kernel Curvature)...")
        opt_kernel = torch.optim.Adam(self.kernel.parameters(), lr=lr)
        for epoch in range(iter_warmup):
            opt_kernel.zero_grad()
            loss = self(X, y)
            loss.backward()
            opt_kernel.step()
            
        print(">>> Phase B: Joint Training (Estimating S_u)...")
        # Optimize all parameters together
        opt_joint = torch.optim.Adam(self.parameters(), lr=lr*0.5) 
        
        for epoch in range(iter_joint):
            opt_joint.zero_grad()
            loss = self(X, y)
            loss.backward()
            opt_joint.step()
            
            # Track history for plotting
            self.history['loss'].append(loss.item())
            self.history['S_u'].append(self.S_u.detach().clone().cpu().numpy())
            
            if (epoch + 1) % 10 == 0:
                print(f"Iter {epoch+1:02d}/{iter_joint} | Loss: {loss.item():.4f} | Est. S_u: {self.S_u[0].item():.4f}")

        # Final Cache for Predictions
        with torch.no_grad():
            K_eff = self.kernel(X) + self._compute_delta_correction(X) + (self.sigma_obs**2) * torch.eye(X.shape[0], device=X.device, dtype=X.dtype)
            self.L = torch.linalg.cholesky(K_eff)
            self.alpha = torch.cholesky_solve(y, self.L)

    def predict(self, X_test):
        with torch.no_grad():
            K_star = self.kernel(X_test, self.X_train)
            K_star_star = self.kernel(X_test)
            
            mu = K_star @ self.alpha
            v = torch.linalg.solve_triangular(self.L, K_star.T, upper=False)
            
            # We add the test-point correction to the predictive variance
            corr_test = self._compute_delta_correction(X_test)
            pred_var = torch.diag(K_star_star) + torch.diag(corr_test) - torch.sum(v**2, dim=0)
            
            return mu, torch.abs(pred_var)

# ==========================================
# 3. TOY PROBLEM EXECUTION (BERKSON ERROR)
# ==========================================
def run_berkson_toy_problem():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on: {device}\n")

    # --- A. Generate Berkson Data ---
    torch.manual_seed(42)
    N = 60
    d = 1
    TRUE_S_U = 0.05 # The Ground Truth we want to recover

    # 1. Nominal Inputs (What the model sees)
    X_nom = torch.linspace(-4, 4, N, dtype=torch.float64, device=device).unsqueeze(1)

    # 2. Hidden Input Noise (What physically happens)
    u_noise = torch.randn_like(X_nom) * math.sqrt(TRUE_S_U)
    X_phys = X_nom + u_noise

    # 3. The Physical Process (Parabola)
    y = -0.25 * (X_phys**2) + 2.0 + (torch.randn_like(X_phys) * 0.1)

    # --- B. Train the Model ---
    # We deliberately initialize the kernel flexibly to allow curvature
    kernel = GenArccos(input_dim=d, n_layers=2, s_b=[0.5, 0.5], s_w=[2.0, 1.5]).to(device)
    
    # We deliberately start S_u with a terrible guess (0.01)
    model = NNGP_EIV_PyTorch(kernel, input_dim=d, S_u_init=0.01, sigma_obs_init=0.1).to(device)

    model.fit(X_nom, y, iter_warmup=25, iter_joint=75, lr=0.05)

    # --- C. Predict & Evaluate ---
    X_test = torch.linspace(-5, 5, 100, dtype=torch.float64, device=device).unsqueeze(1)
    mu, var = model.predict(X_test)
    std = torch.sqrt(var).cpu().numpy().flatten()
    
    X_test_np = X_test.cpu().numpy().flatten()
    mu_np = mu.cpu().numpy().flatten()
    y_true_np = -0.25 * (X_test_np**2) + 2.0

    # --- D. Plotting ---
    plt.figure(figsize=(12, 5))

    # Subplot 1: Function Fit
    plt.subplot(1, 2, 1)
    plt.scatter(X_nom.cpu(), y.cpu(), c='red', alpha=0.5, label='Noisy Obs')
    plt.plot(X_test_np, y_true_np, 'k--', label='True Latent Func')
    plt.plot(X_test_np, mu_np, 'b-', linewidth=2, label='NNGP Mean')
    plt.fill_between(X_test_np, mu_np - 1.96*std, mu_np + 1.96*std, color='blue', alpha=0.2, label='95% CI')
    plt.title("Function Fit with Berkson Error")
    plt.legend()

    # Subplot 2: Variance Convergence
    plt.subplot(1, 2, 2)
    Su_history = [s[0] for s in model.history['S_u']] # Extract dimension 0
    plt.axhline(TRUE_S_U, color='k', linestyle='--', label=f'True S_u ({TRUE_S_U})')
    plt.plot(range(len(Su_history)), Su_history, 'g-', linewidth=2, label='Estimated S_u')
    plt.title(f"Input Noise Inference\nFinal Estimate: {Su_history[-1]:.3f}")
    plt.xlabel("Joint Training Iteration")
    plt.ylabel("Variance")
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_berkson_toy_problem()