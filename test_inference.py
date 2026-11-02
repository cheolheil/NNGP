import copy
import unittest
import torch
from nngp_torch import GenArccos, NNGP_EIV
from nngp_torch_joint import NNGP_EIV_Joint

torch.set_num_threads(1)

class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.x = torch.tensor([[-0.6], [0.2], [0.9]], dtype=torch.float64)
        self.y = self.x.square()
        self.kernel = GenArccos(1, 2, [0.7, 0.5], [1.4, 1.2])

    def test_joint_objective_and_legacy_wrapper(self):
        old = NNGP_EIV_Joint(copy.deepcopy(self.kernel), 1)
        new = NNGP_EIV(copy.deepcopy(self.kernel), 1, inference_method='joint')
        covariance = new.kernel(self.x) + new._compute_delta_correction(self.x)
        covariance = covariance + new.sigma_obs.square() * torch.eye(3, dtype=self.x.dtype)
        expected = -torch.distributions.MultivariateNormal(
            torch.zeros(3, dtype=self.x.dtype), covariance_matrix=covariance).log_prob(self.y[:, 0])
        torch.testing.assert_close(expected, new(self.x, self.y))
        torch.testing.assert_close(old(self.x, self.y), new(self.x, self.y))
        old.fit(self.x, self.y, iter_warmup=1, iter_joint=2)
        new.fit(self.x, self.y, iter_warmup=1, iter_joint=2)
        for name, parameter in old.named_parameters():
            torch.testing.assert_close(parameter, dict(new.named_parameters())[name])
        for a, b in zip(old.predict(self.x), new.predict(self.x)):
            torch.testing.assert_close(a, b)
        self.assertIsNone(new.log_V.grad)

    def test_fit_override_and_switch_back(self):
        model = NNGP_EIV(self.kernel, 1)
        self.assertIs(model.fit(self.x, self.y, inference_method='joint', iter_warmup=0, iter_joint=1), model)
        torch.testing.assert_close(model(self.x, self.y), model.compute_nll(self.x, self.y))
        model.fit(self.x, self.y, inference_method='alternating', outer_iters=1, inner_kernel_steps=1, inner_su_steps=1)
        torch.testing.assert_close(model(self.x, self.y), -model.compute_elbo(self.x, self.y))
        self.assertEqual(len(model.history['V']), 1)
        model.predict(self.x)

    def test_invalid_method_and_restart(self):
        with self.assertRaises(ValueError):
            NNGP_EIV(self.kernel, 1, inference_method='unknown')
        model = NNGP_EIV(self.kernel, 1)
        with self.assertRaises(ValueError):
            model.fit(self.x, self.y, inference_method='unknown')
        with self.assertRaises(ValueError):
            model.fit(self.x, self.y, n_restart=0)
        self.assertEqual(model.inference_method, 'alternating')

    def test_legacy_joint_positional_fit(self):
        model = NNGP_EIV_Joint(self.kernel, 1)
        model.fit(self.x, self.y, 1, 0, 1, 0.01)
        self.assertEqual(model.inference_method, 'joint')

if __name__ == '__main__':
    unittest.main()
