import numpy as np
import scipy.stats as stats

from scipy.linalg import solve_triangular
from scipy.optimize import minimize

from kernel import GenArccos


class NNGPIU:
    """
    Neural Network Gaussian Process Input Uncertainty with Generalized Arccosine Kernel
    input_dim : int
    n_layers : int
    kernel : callable object
    alpha : float / the nugget effect
    gamma : float / penalty weight associated with S (i.e., variable selection)
    s_w : float or 1-D array
    S = 1-D array
    """
    def __init__(self, input_dim, n_layers, kernel=None, alpha=1e-8, gamma=1e-3, theta_bounds=None, n_restart=0, input_noise_cov=None, n_noise_samples=500):
        self.input_dim = input_dim
        self.n_layers = n_layers
        if kernel is None:
            self.kernel = GenArccos(self.input_dim, self.n_layers, np.ones(self.n_layers), np.ones(self.n_layers))
        else:
            self.kernel = kernel
        self.alpha = alpha
        self.gamma = gamma
        self.n_restart = n_restart

        
        if theta_bounds == None:
            # theta_bounds should be a list
            self.theta_bounds = [(1e-4, 1e4) for _ in range(len(self.kernel.theta))]
        else:
            assert len(theta_bounds) == len(self.kernel.theta)
            self.theta_bounds = theta_bounds
        if input_noise_cov == None:
            self._input_noise_exists = False
        else:
            self._input_noise_exists = True
            self.input_noise_cov = input_noise_cov
            self.n_noise_samples = n_noise_samples

    def neg_log_likelihood(self, theta, return_La=False):
        """
        It only gives the likelihood value, and should never touch the original kernel feature. So, it needs to be copied with given theta.
        Calculates the expected value of the likelihood value given the input noise.
        """
        theta_old = self.kernel.theta
        self.kernel.theta = theta

        if self._input_noise_exists:
            K = np.zeros((len(self.u), len(self.X), len(self.X)))
            for i in range(len(self.u)):
                K[i] = self.kernel(self.Xu[i])
            K = K.sum(axis=0)
            K /= len(self.u)
        else:
            K = self.kernel(self.X)

        K += np.eye(len(self.X)) * self.alpha
        
        try:
            L = np.linalg.cholesky(K)
            a = solve_triangular(L.T, solve_triangular(L, self.y, lower=True))
        except:
            raise Exception("Cholesky decomp on K is failed; K may not be PSD.")
        
        # recover the original theta value
        
        nlm = 0.5 * np.dot(self.y, a) + np.log(np.diag(L)).sum() + 0.5 * len(self.X) * np.log(2 * np.pi) + self.gamma * self.kernel.S.sum()

        self.kernel.theta = theta_old

        if return_La:
            return nlm, L, a
        else:
            return nlm

    def fit(self, X, y):
        if len(X) != len(y):
            raise Exception("InconsistentDataError: len(X) != len(y)")

        if np.ndim(X) != 2:
            raise Exception("InputShapeError: ndim(X) != 2")

        self.X = X.copy()
        self.y = y.copy()
        if self._input_noise_exists:
            self.u = np.atleast_3d(stats.multivariate_normal(np.zeros(self.input_dim), np.diag(self.input_noise_cov)).rvs(size=(self.n_noise_samples, len(self.X))))
            self.Xu = self.X + self.u
        else:
            pass

        # e-step
        res = minimize(self.neg_log_likelihood, self.kernel.theta, method='L-BFGS-B', bounds=self.theta_bounds)
        self.set_theta(res.x)

        if self.n_restart > 0:
            n_restart = self.n_restart
            nlm_cand = [self.nlm]
            theta_cand = [self.kernel.theta]
            # L_cand = [self.L]
            # a_cand = [self.a]

            while n_restart > 0:
                theta0 = np.random.uniform([self.theta_bounds[i][0] for i in range(len(self.kernel.theta))],
                                           [self.theta_bounds[i][1] for i in range(len(self.kernel.theta))])
                res = minimize(self.neg_log_likelihood, theta0, method='L-BFGS-B', bounds=self.theta_bounds)

                nlm_cand.append(res.fun)
                theta_cand.append(res.x)
                n_restart -= 1
                # print(n_restart)

            # print(nlm_cand)
            min_index = np.argmin(nlm_cand)
            # self.nlm = nlm_cand[min_index]
            self.set_theta(theta_cand[min_index])
            # self.kernel.theta = theta_cand[min_index]
            # self.L = L_cand[min_index]
            # self.a = a_cand[min_index]

        

    def predict(self, X_new, return_std=False):
        k_new = self.kernel(self.X, X_new)
        y_new = k_new.T @ self.a
        if return_std:
            v = solve_triangular(self.L, k_new, lower=True)
            std_new = np.diag(self.kernel(X_new, theta=self.theta, n_layers=self.n_layers)) \
                      + self.alpha - np.diag(v.T @ v)

            # for i in range(len(X_new)):
            #     std_new[i] = ArccosKernel(X_new[i][np.newaxis, :], theta=self.theta, n_layers=self.n_layers) \
            #                  + np.finfo(np.float32).eps - np.dot(v[:, i], v[:, i])
            return y_new, std_new
        else:
            return y_new
    
    def set_theta(self, theta):
        self.kernel.theta = theta
        # self.K = self.kernel(self.X) + np.eye(self.n) * self.alpha
        self.nlm, self.L, self.a = self.neg_log_likelihood(theta, return_La=True)

    @property
    def theta(self):
        if self._input_noise_exists:
            return np.hstack((self.kernel.theta, self.input_noise_cov))
        if not self._input_noise_exists:
            return self.kernel.theta
        