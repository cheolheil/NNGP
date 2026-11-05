import math
import unittest
import torch
from nngp_delta import kernel_delta, expected_log_likelihood_delta, expected_log_likelihood_delta_reference
from nngp_torch import GenArccos, NNGP_EIV
from legacy.nngp_torch_joint import NNGP_EIV_Joint

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

class PolynomialKernel(torch.nn.Module):
    def paired(self, x, y):
        return (1 + (x*y).sum(-1)).square()
    def diagonal(self, x):
        return (1 + x.square().sum(-1)).square()
    def forward(self, x, y=None):
        y = x if y is None else y
        return self.paired(x[:,None,:], y[None,:,:])

class DeltaTests(unittest.TestCase):
    def test_linear_exact_diagonal_and_cross(self):
        k = GenArccos(2, 1, [0.7], [1.4])
        x = torch.tensor([[0.2, -0.4], [0.8, 1.1]])
        variance = torch.tensor([0.02, 0.03])
        expected = torch.eye(2) * (variance*k.D_param*k.s_w[0]).sum()
        torch.testing.assert_close(kernel_delta(k, x, variance), expected)
        torch.testing.assert_close(kernel_delta(k, x, variance, x), torch.zeros(2,2))

    def test_polynomial_off_diagonal_and_shared_diagonal(self):
        k = PolynomialKernel()
        x = torch.tensor([[-0.6], [0.9]])
        variance = torch.tensor([0.02])
        off = variance[0]*(x.square() + x.square().T)
        diag = variance[0]*(2 + 6*x[:,0].square())
        expected = off - off.diagonal().diag() + diag.diag()
        torch.testing.assert_close(kernel_delta(k, x, variance), expected)

    def test_arccos_kernel_delta_finite_differences(self):
        k = GenArccos(2, 3, [0.7,0.6,0.5], [1.4,1.3,1.2])
        x = torch.tensor([[0.2,-0.3],[0.8,0.6]])
        variance = torch.tensor([0.02,0.03]); h=1e-4
        base = k(x).detach(); expected=torch.zeros_like(base)
        for i in range(len(x)):
            for d in range(2):
                plus=x.clone(); minus=x.clone(); plus[i,d]+=h; minus[i,d]-=h
                expected += 0.5*variance[d]*(k(plus).detach()-2*base+k(minus).detach())/h**2
        torch.testing.assert_close(kernel_delta(k,x,variance), expected, atol=2e-8, rtol=2e-5)

    def test_expected_log_likelihood_hessian(self):
        k = GenArccos(2, 2, [0.7,0.5], [1.4,1.2])
        x=torch.tensor([[0.2,-0.3],[0.8,0.6],[-0.4,0.7]])
        y=torch.tensor([[0.3],[-0.5],[0.1]]); variance=torch.tensor([0.001,0.002]); sigma=torch.tensor(0.3)
        def loglik(z):
            c=k(z)+sigma.square()*torch.eye(len(z))
            return torch.distributions.MultivariateNormal(torch.zeros(len(z)), covariance_matrix=c).log_prob(y[:,0])
        base=loglik(x).detach(); expected=base.clone(); h=1e-4
        for i in range(len(x)):
            for d in range(2):
                plus=x.clone(); minus=x.clone(); plus[i,d]+=h; minus[i,d]-=h
                expected+=0.5*variance[d]*(loglik(plus).detach()-2*base+loglik(minus).detach())/h**2
        torch.testing.assert_close(expected_log_likelihood_delta(k,x,y,variance,sigma),expected,atol=1e-7,rtol=1e-7)

    def test_trainable_hessian_gradient(self):
        k=GenArccos(2,2,[0.7,0.5],[1.4,1.2])
        model=NNGP_EIV(k,2,V_init=0.001)
        x=torch.tensor([[0.2,-0.3],[0.8,0.6],[-0.4,0.7]])
        y=torch.tensor([[0.3],[-0.5],[0.1]])
        loss=model(x,y); loss.backward()
        analytic=k.log_s_w.grad[0].item(); h=1e-5
        with torch.no_grad(): k.log_s_w[0]+=h
        plus=model(x,y).item()
        with torch.no_grad(): k.log_s_w[0]-=2*h
        minus=model(x,y).item()
        with torch.no_grad(): k.log_s_w[0]+=h
        self.assertAlmostEqual(analytic,(plus-minus)/(2*h),places=5)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_eval_no_grad_four_dimensions_and_prediction(self):
        x=torch.tensor([[0.2,-0.3,0.4,0.1],[0.8,0.6,-0.4,0.3],[-0.4,0.7,0.2,-0.5]])
        y=torch.tensor([[0.3],[-0.5],[0.1]])
        for cls in [NNGP_EIV,NNGP_EIV_Joint]:
            model=cls(GenArccos(4,2,[0.7,0.5],[1.4,1.2]),4)
            if cls is NNGP_EIV:
                model.fit(x,y,outer_iters=1,inner_kernel_steps=1,inner_su_steps=1)
            else:
                model.fit(x,y,iter_warmup=1,iter_joint=1)
            model.eval()
            with torch.no_grad():
                self.assertTrue(torch.isfinite(model(x,y)))
                correction=model._compute_delta_correction(x)
                self.assertFalse(correction.requires_grad)
                for variance in [None,0.001]:
                    mu,var=model.predict(x+0.1,input_variance=variance)
                    self.assertTrue(torch.isfinite(mu).all() and (var>=0).all())

    def test_joint_parameter_gradient(self):
        model=NNGP_EIV_Joint(GenArccos(2,2,[0.7,0.5],[1.4,1.2]),2)
        x=torch.tensor([[0.2,-0.3],[0.8,0.6],[-0.4,0.7]])
        y=torch.tensor([[0.3],[-0.5],[0.1]])
        loss=model(x,y);loss.backward()
        analytic=model.log_S_u.grad[0].item();h=1e-5
        with torch.no_grad():model.log_S_u[0]+=h
        plus=model(x,y).item()
        with torch.no_grad():model.log_S_u[0]-=2*h
        minus=model(x,y).item()
        self.assertAlmostEqual(analytic,(plus-minus)/(2*h),places=5)

    def test_float32_and_trainable_input_scale(self):
        k=GenArccos(2,3,[0.7,0.6,0.5],[1.4,1.3,1.2],D=[0.4,0.6],dtype=torch.float32)
        x=torch.tensor([[0.2,-0.3],[0.8,0.6]],dtype=torch.float32)
        variance=torch.tensor([0.001,0.002],dtype=torch.float32)
        correction=kernel_delta(k,x,variance)
        correction.sum().backward()
        self.assertTrue(torch.isfinite(k.log_D.grad).all())
        with torch.no_grad():
            self.assertTrue(torch.isfinite(kernel_delta(k,x,variance)).all())

    def test_sparse_likelihood_matches_reference_value_and_gradients(self):
        torch.manual_seed(123)
        for dimensions in (1, 4):
            for layers in (1, 3):
                k=GenArccos(dimensions,layers,[0.7]*layers,[1.4]*layers)
                x=torch.randn(5,dimensions); y=torch.randn(5,1)
                variance=torch.full((dimensions,),0.002,requires_grad=True)
                sigma=torch.tensor(0.3,requires_grad=True)
                parameters=[*k.parameters(),variance,sigma]
                fast=expected_log_likelihood_delta(k,x,y,variance,sigma)
                reference=expected_log_likelihood_delta_reference(k,x,y,variance,sigma)
                torch.testing.assert_close(fast,reference,rtol=1e-9,atol=1e-9)
                fast_grad=torch.autograd.grad(fast,parameters)
                ref_grad=torch.autograd.grad(reference,parameters)
                for actual,expected in zip(fast_grad,ref_grad):
                    torch.testing.assert_close(actual,expected,rtol=1e-8,atol=1e-8)

    def test_zero_variance(self):
        k=GenArccos(2,2,[0.7,0.5],[1.4,1.2])
        x=torch.tensor([[0.2,-0.3],[0.8,0.6]])
        torch.testing.assert_close(kernel_delta(k,x,torch.zeros(2)),torch.zeros(2,2))

if __name__=='__main__': unittest.main(verbosity=2)
