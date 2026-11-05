import scipy.stats as stats
import numpy as np

class simfun:
    def __init__(self, pF, pT, mu_a=0, sigma_a=1, mu_b=1, sigma_b=1, sigma_e=1e-3, Sigma_uT=None, Sigma_uF=None):
        self.pF = pF
        self.pT = pT
        self.mu_a = mu_a
        self.sigma_a = sigma_a
        self.mu_b = mu_b
        self.sigma_b = sigma_b
        self.sigma_e = sigma_e
        self.Sigma_uT = Sigma_uT
        self.Sigma_uF = Sigma_uF
    
    def gen_params(self, save=False):
        self.a = stats.norm(loc=self.mu_a, scale=self.sigma_a).rvs(self.pT)
        self.b = stats.norm(loc=self.mu_b, scale=self.sigma_b).rvs(self.pT)
        if save:
            np.save('param_a.npy', self.a)
            np.save('param_b.npy', self.b)
        else:
            pass

    def set_params(self, a, b):
        self.a = a
        self.b = b

    def observe(self, X, return_W=False):
        X -= 0.5
        X *= 2
        assert X.shape[1] == self.p

        U = stats.multivariate_normal(mean=np.zeros(self.p), cov=self.Sigma_u, allow_singular=True).rvs(X.shape[0])
        W = X + U
        E = stats.norm(loc=0, scale=self.sigma_e).rvs(X.shape[0])

        if return_W:
            return np.sum(self.a * np.sin(np.pi * W[:, :self.pT] / self.b), axis=1) + E, W
        else:
            return np.sum(self.a * np.sin(np.pi * W[:, :self.pT] / self.b), axis=1) + E
    
    def get_test(self, X):
        X -= 0.5
        X *= 2
        assert X.shape[1] == self.p
        
        return np.sum(self.a * np.sin(np.pi * X[:, :self.pT] / self.b), axis=1)
    
    @property
    def p(self):
        return self.pT + self.pF
    
    @property
    def Sigma_u(self):
        return np.block([[self.Sigma_uT, np.zeros((self.pT, self.pF))], [np.zeros((self.pF, self.pT)), self.Sigma_uF]])