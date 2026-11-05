import unittest
import torch
from nngp_torch import GenArccos
from simulation_20d import generate_data
from simulation_20d_naive import NaiveNNGP, covariance, tridiagonal_data

class NaiveStudyTests(unittest.TestCase):
    def test_observation_only_gaussian_reference(self):
        x=torch.tensor([[-0.7],[0.2],[0.8]],dtype=torch.float64)
        y=x.square()
        model=NaiveNNGP(GenArccos(1,2,[0.5,0.5],[2.,1.5]))
        cov=model.kernel(x)+model.sigma_obs.square()*torch.eye(3,dtype=x.dtype)
        expected=-torch.distributions.MultivariateNormal(torch.zeros(3,dtype=x.dtype),covariance_matrix=cov).log_prob(y[:,0])
        torch.testing.assert_close(model(x,y),expected)
        self.assertEqual(set(dict(model.named_parameters())),{'kernel.log_s_b','kernel.log_s_w','log_sigma_obs'})
        model.fit(x,y)
        mean,var=model.predict(x)
        fitted=model.training_covariance(x)
        cross=model.kernel(x)
        torch.testing.assert_close(mean,cross@torch.linalg.solve(fitted,y))
        torch.testing.assert_close(var,(cross-cross@torch.linalg.solve(fitted,cross)).diagonal())

    def test_tridiagonal_covariance_and_paired_data(self):
        original=generate_data(); correlated=tridiagonal_data()
        for index in (0,2,3,4):
            torch.testing.assert_close(original[index],correlated[index],rtol=0,atol=0)
        self.assertFalse(torch.equal(original[1],correlated[1]))
        v=original[4];sigma=covariance(v)
        torch.testing.assert_close(sigma.diagonal(),v)
        torch.testing.assert_close(sigma.diagonal(1),0.35*(v[:-1]*v[1:]).sqrt())
        i=torch.arange(20);self.assertEqual(torch.count_nonzero(sigma[(i[:,None]-i).abs()>1]),0)
        self.assertGreater(torch.linalg.eigvalsh(sigma).min().item(),0)
        torch.testing.assert_close(covariance(v,0),v.diag())

if __name__=='__main__': unittest.main()
