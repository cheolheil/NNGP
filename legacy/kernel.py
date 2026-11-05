import numpy as np
import scipy.stats as stats

from scipy.linalg import solve_triangular
from scipy.optimize import minimize


class GenArccos:
    """
    Generalized Arccosine Kernel:
    It generalizes existing arccosine kernel in Lee et al. (2017) by parametering the Sigma matrix in the input layer.
    input_dim : int
    n_layers : int
    s_b : float or 1-D array
    s_w : float or 1-D array
    S = 1-D array

    features
    _S_fixed: checks S is parameterized (if so, theta includes S along with s_w and s_b)
    Also, it determines refer to s_w[0] in the initial layer.
    """
    def __init__(self, input_dim, n_layers, s_b, s_w, S=None):
        # if S == None, it is regarded as the usual Arccos Kernel in Lee et al. (2017)
        # add Exception : len(s_b) != n_layers, len(s_w) != n_layers, len(S) != input_dim
        # if S is not given, then we fix S with its default value as in Lee et al. (2017)
        self.input_dim = input_dim
        self.n_layers = n_layers
        
        if S == None:
            # in this case, we don't parametrize S
            self.s_b = np.atleast_1d(s_b)
            self.s_w = np.atleast_1d(s_w)
            self._S_fixed = True
            self.S = np.ones(input_dim) / input_dim
        else:
            # in this case, we omit the first s_w, since it conflicts with S in the initial layer
            assert input_dim == len(S)
            self.s_b = np.atleast_1d(s_b)
            self.s_w = np.atleast_1d(s_w)[1:]
            self._S_fixed = False
            self.S = S

    def _init_layer(self, X, Y):
        # it distinguishes the tunability of S to conisder the first value of s_w or not
        if self._S_fixed:
            K = self.s_w[0] * X @ np.diag(self.S) @ Y.T + self.s_b[0]
        if not self._S_fixed:
            K = X @ np.diag(self.S) @ Y.T + self.s_b[0]
        return K

    def __call__(self, X, Y=None):
        if Y is None:
            K = self._init_layer(X, X)

            if self.n_layers > 1:
                for l in range(1, self.n_layers):
                    k_diag = np.diag(K)
                    sqrt_K_outer = np.sqrt(np.outer(k_diag, k_diag))
                    theta_mat = np.arccos(np.clip(K / sqrt_K_outer, -1, 1))
                    K = 0.5 / np.pi * self.s_w[l] * sqrt_K_outer * (np.sin(theta_mat) + (np.pi - theta_mat)
                                                                    * np.cos(theta_mat)) + self.s_b[l]
        else:
            if X.shape[1] != Y.shape[1]:
                raise Exception("InputError: X.shape[1] and Y.shape[1] are not compatible")

            K = self._init_layer(X, Y)

            if self.n_layers > 1:
                K_XX = self._init_layer(X, X)
                K_YY = self._init_layer(Y, Y)
                for l in range(1, self.n_layers):
                    kX_diag = np.diag(K_XX)
                    kY_diag = np.diag(K_YY)
                    sqrt_KX_outer = np.sqrt(np.outer(kX_diag, kX_diag))
                    sqrt_KY_outer = np.sqrt(np.outer(kY_diag, kY_diag))
                    sqrt_K_outer = np.sqrt(np.outer(kX_diag, kY_diag))
                    thetaX_mat = np.arccos(np.clip(K_XX / sqrt_KX_outer, -1, 1))
                    thetaY_mat = np.arccos(np.clip(K_YY / sqrt_KY_outer, -1, 1))
                    theta_mat = np.arccos(np.clip(K / sqrt_K_outer, -1, 1))
                    K_XX = 0.5 / np.pi * self.s_w[l] * sqrt_KX_outer * (np.sin(thetaX_mat) + (np.pi - thetaX_mat)
                                                                    * np.cos(thetaX_mat)) + self.s_b[l]
                    K_YY = 0.5 / np.pi * self.s_w[l] * sqrt_KY_outer * (np.sin(thetaY_mat) + (np.pi - thetaY_mat)
                                                                    * np.cos(thetaY_mat)) + self.s_b[l]
                    K = 0.5 / np.pi * self.s_w[l] * sqrt_K_outer * (np.sin(theta_mat) + (np.pi - theta_mat)
                                                                * np.cos(theta_mat)) + self.s_b[l]
        return K
    
    @property
    def theta(self):
        if self._S_fixed:
            return np.hstack([self.s_b, self.s_w])
        else:
            return np.hstack([self.s_b, self.s_w, self.S])
    
    @theta.setter
    def theta(self, theta):
        self.s_b = theta[:len(self.s_b)]
        self.s_w = theta[len(self.s_b):len(self.s_b) + len(self.s_w)]
        if self._S_fixed:
            pass
        else:
            self.S = theta[len(self.s_b) + len(self.s_w):]