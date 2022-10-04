import numpy as np

from scipy.linalg import solve_triangular
from scipy.optimize import minimize


def ArccosKernel(X, Y=None, theta=np.ones(2), n_layers=3):
    if len(theta) != 2 and len(theta) != 2 * n_layers:
        raise Exception("Param Error: Synchronize len(theta) with n_layers")

    # len(tehta)==2 case means that the same (theta_w, theta_b) will be used for all layers.
    if len(theta) == 2:
        theta_w = theta[:1] * np.ones(n_layers)
        theta_b = theta[1:] * np.ones(n_layers)
    # If len(theta) is specified other than 2 (and n_layer != 1), parameters will be differ by layers.
    elif len(theta) == 2 * n_layers:
        theta_w = theta[:n_layers]
        theta_b = theta[n_layers:]

    if Y is None:
        K = np.inner(X, X * theta_w[0] / X.shape[1]) + theta_b[0]

        if n_layers > 1:
            for l in range(1, n_layers):
                k_diag = np.diag(K)
                sqrt_K_outer = np.sqrt(np.outer(k_diag, k_diag))
                theta_mat = np.arccos(np.clip(K / sqrt_K_outer, -1, 1))
                K = 0.5 / np.pi * theta_w[l] * sqrt_K_outer * (np.sin(theta_mat) + (np.pi - theta_mat)
                                                               * np.cos(theta_mat)) + theta_b[l]
    elif Y is not None:
        if X.shape[1] != Y.shape[1]:
            raise Exception("InputError: X.shape[1] and Y.shape[1] are not compatible")

        K = np.inner(X, Y * theta_w[0] / X.shape[1]) + theta_b[0]

        if n_layers > 1:
            K_XX = np.inner(X, X * theta_w[0] / X.shape[1]) + theta_b[0]
            K_YY = np.inner(Y, Y * theta_w[0] / X.shape[1]) + theta_b[0]
            for l in range(1, n_layers):
                kX_diag = np.diag(K_XX)
                kY_diag = np.diag(K_YY)
                sqrt_KX_outer = np.sqrt(np.outer(kX_diag, kX_diag))
                sqrt_KY_outer = np.sqrt(np.outer(kY_diag, kY_diag))
                sqrt_K_outer = np.sqrt(np.outer(kX_diag, kY_diag))
                thetaX_mat = np.arccos(np.clip(K_XX / sqrt_KX_outer, -1, 1))
                thetaY_mat = np.arccos(np.clip(K_YY / sqrt_KY_outer, -1, 1))
                theta_mat = np.arccos(np.clip(K / sqrt_K_outer, -1, 1))
                K_XX = 0.5 / np.pi * theta_w[l] * sqrt_KX_outer * (np.sin(thetaX_mat) + (np.pi - thetaX_mat)
                                                                   * np.cos(thetaX_mat)) + theta_b[l]
                K_YY = 0.5 / np.pi * theta_w[l] * sqrt_KY_outer * (np.sin(thetaY_mat) + (np.pi - thetaY_mat)
                                                                   * np.cos(thetaY_mat)) + theta_b[l]
                K = 0.5 / np.pi * theta_w[l] * sqrt_K_outer * (np.sin(theta_mat) + (np.pi - theta_mat)
                                                               * np.cos(theta_mat)) + theta_b[l]
    return K


class NNGPR:
    def __init__(self, theta=np.ones(2), alpha=1e-8, n_layers=3, theta_bounds=None, n_restart=0):
        self.theta = theta
        self.alpha = alpha
        self.n_layers = n_layers
        self.n_restart = n_restart
        if theta_bounds == None:
            self.theta_bounds = [(1e-4, 1e4) for _ in range(len(self.theta))]
        else:
            assert len(theta_bounds) == len(theta)
            self.theta_bounds = theta_bounds

    def neg_log_likelihood(self, theta):
        K = ArccosKernel(self.X, theta=theta, n_layers=self.n_layers) + np.eye(self.n) * self.alpha
        try:
            self.L = np.linalg.cholesky(K)
            self.a = solve_triangular(self.L.T, solve_triangular(self.L, self.y, lower=True))
        except:
            print(theta)
        return 0.5 * np.dot(self.y, self.a) + np.log(np.diag(self.L)).sum() + 0.5 * self.n * np.log(2 * np.pi)

    def fit(self, X, y):
        if len(X) != len(y):
            raise Exception("InconsistentDataError: len(X) != len(y)")

        if np.ndim(X) != 2:
            raise Exception("InputShapeError: ndim(X) != 2")

        self.X = X.copy()
        self.y = y.copy()
        self.n = len(self.X)

        res = minimize(self.neg_log_likelihood, self.theta, method='L-BFGS-B', bounds=self.theta_bounds)
        self.nlm = res.fun
        self.theta = res.x

        if self.n_restart > 0:
            nlm_cand = [self.nlm]
            theta_cand = [self.theta]
            L_cand = [self.L]
            a_cand = [self.a]

            while self.n_restart > 0:
                theta0 = np.random.uniform([self.theta_bounds[i][0] for i in range(len(self.theta))],
                                           [self.theta_bounds[i][1] for i in range(len(self.theta))])
                res = minimize(self.neg_log_likelihood, theta0, method='L-BFGS-B', bounds=self.theta_bounds)

                nlm_cand.append(res.fun)
                theta_cand.append(res.x)
                L_cand.append(self.L)
                a_cand.append(self.a)
                self.n_restart -= 1

            min_index = np.argmin(nlm_cand)
            self.nlm = nlm_cand[min_index]
            self.theta = theta_cand[min_index]
            self.L = L_cand[min_index]
            self.a = a_cand[min_index]

        self.K = ArccosKernel(X, theta=self.theta, n_layers=self.n_layers) + np.eye(self.n) * np.finfo(np.float32).eps

    def predict(self, X_new, return_std=False):
        k_new = ArccosKernel(self.X, X_new, theta=self.theta, n_layers=self.n_layers)
        y_new = k_new.T @ self.a
        if return_std:
            v = solve_triangular(self.L, k_new, lower=True)
            std_new = np.diag(ArccosKernel(X_new, theta=self.theta, n_layers=self.n_layers)) \
                      + self.alpha - np.diag(v.T @ v)

            # for i in range(len(X_new)):
            #     std_new[i] = ArccosKernel(X_new[i][np.newaxis, :], theta=self.theta, n_layers=self.n_layers) \
            #                  + np.finfo(np.float32).eps - np.dot(v[:, i], v[:, i])
            return y_new, std_new
        else:
            return y_new


def psd_convert(K):
    if not np.array_equal(K, K.T):
        raise Exception("The matrix is not Hermitian (symmetric).")
    l, v = np.linalg.eigh(K)
    if np.all(l >= 0):
        # print("The matrix is already positive semidefinite.")
        return K
    else:
        l_new = l
        for i in range(len(l)):
            l_new[i] = max(l[i], 1e-8 * np.random.rand())
        K_new = v @ np.diag(l_new) @ v.T
        print("The matrix is successfully converted.")
        return K_new
