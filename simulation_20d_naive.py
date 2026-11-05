"""20D EIV versus naive NNGP under diagonal and tridiagonal true noise.

python simulation_20d_naive.py
Reuses verified diagonal-study EIV results; computes all missing results.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import copy
import hashlib
import json
import math
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
import torch
from nngp_torch import GenArccos, NNGP_EIV
from simulation_20d import generate_data, ALT, JOINT, sync, vector

ROOT=Path(__file__).resolve().parent
OUTPUT=ROOT/'simulation_20d_naive_results.json'
METHODS=('alternating','joint','naive_100','naive_300')

def covariance(variances, rho=0.35):
    off=rho*(variances[:-1]*variances[1:]).sqrt()
    return variances.diag()+off.diag(1)+off.diag(-1)

def tridiagonal_data():
    x,_,test,target,truth=generate_data()
    g=torch.Generator().manual_seed(42)
    torch.rand(100,20,generator=g,dtype=x.dtype)  # align draws with original study
    z=torch.randn(100,20,generator=g,dtype=x.dtype)
    physical=x+z@torch.linalg.cholesky(covariance(truth)).T
    weights=torch.linspace(0.5,1.5,20,dtype=x.dtype)
    latent=(2-0.25*(physical.square()*weights).sum(-1,keepdim=True)/math.sqrt(20)
            +0.5*physical[:,:1].sin()+0.5*physical[:,1:2].cos())
    y=latent+0.1*torch.randn(100,1,generator=g,dtype=x.dtype)
    return x,y,test,target,truth

class NaiveNNGP(torch.nn.Module):
    """Exact Gaussian NLL on nominal inputs, with observation noise only."""
    def __init__(self,kernel):
        super().__init__()
        self.kernel=kernel
        self.log_sigma_obs=torch.nn.Parameter(torch.tensor(math.log(0.1),dtype=torch.float64))
        self.history={'loss':[]}
    @property
    def sigma_obs(self): return self.log_sigma_obs.exp()
    def training_covariance(self,x):
        return self.kernel(x)+self.sigma_obs.square()*torch.eye(len(x),device=x.device,dtype=x.dtype)
    def forward(self,x,y):
        L=torch.linalg.cholesky(self.training_covariance(x))
        return 0.5*(y*torch.cholesky_solve(y,L)).sum()+L.diagonal().log().sum()+0.5*len(x)*math.log(2*math.pi)
    def fit(self,x,y,steps=100):
        self.X_train=x
        self.history={'loss':[]}
        # 100-update schedule matches joint; 300-update schedule matches alternating's total updates/rate.
        phases=((25,0.05),(75,0.025)) if steps==100 else ((300,0.03),)
        for count,lr in phases:
            opt=torch.optim.Adam(self.parameters(),lr=lr)
            for _ in range(count):
                opt.zero_grad();loss=self(x,y);loss.backward();opt.step()
                self.history['loss'].append(loss.item())
        with torch.no_grad():
            self.L=torch.linalg.cholesky(self.training_covariance(x))
            self.alpha=torch.cholesky_solve(y,self.L)
        return self
    def predict(self,x,include_observation_noise=False):
        with torch.no_grad():
            cross=self.kernel(x,self.X_train)
            mean=cross@self.alpha
            v=torch.linalg.solve_triangular(self.L,cross.T,upper=False)
            var=self.kernel.diagonal(x)-v.square().sum(0)
            if include_observation_noise: var=var+self.sigma_obs.square()
            tolerance=100*torch.finfo(var.dtype).eps*self.kernel.diagonal(x).abs().max().clamp_min(1)
            if (var < -tolerance).any(): raise RuntimeError('Negative naive prediction variance')
            return mean,var.clamp_min(0)

def starts(method):
    g=torch.Generator().manual_seed(42);out=[]
    for restart in range(3):
        su=torch.full((20,),0.01,dtype=torch.float64);v=torch.full((20,),0.001,dtype=torch.float64)
        if restart:
            su*=torch.exp(torch.randn(20,generator=g,dtype=torch.float64)*0.5)
            if method=='alternating':v*=torch.exp(torch.randn(20,generator=g,dtype=torch.float64)*0.5)
        out.append((vector(su),vector(v)))
    return out

def write_report(result):
    lines=['# 20D: input-noise inference versus naive NNGP','',
        '## Design','',
        '100 training observations, 200 exact-input latent test targets, seed 42, float64. Nominal/test inputs, latent function, observation-noise draws, and standardized input-error draws are paired across scenarios. The latent function is unchanged from simulation_20d.py.',
        'True marginal input variances increase linearly from 0.02 to 0.08. True observation standard deviation is 0.1.',
        'Diagonal scenario: independent Gaussian Berkson errors. Tridiagonal scenario: adjacent-coordinate correlation 0.35 and all other off-diagonal correlations zero; covariance entries Sigma[d,d+1]=0.35 sqrt(v[d]v[d+1]). This covariance is positive definite. Both EIV fits retain the diagonal assumption.',
        'Alternating minimizes negative approximate ELBO; joint minimizes moment Gaussian NLL. Their losses must not be compared across methods. Kernel: two-layer GenArccos, fixed equal input weights, initial biases [0.5,0.5], weights [2,1.5], observation std. 0.1.',
        'Naive NNGP uses K(X,X)+sigma_obs^2 I and standard Gaussian NLL, optimizing only kernel parameters and observation noise. It has no input-noise parameters, KL, or Delta correction. Latent predictions exclude observation noise for all methods.',
        'Alternating has 300 updates at lr 0.03 (15 outer iterations, 10 kernel/noise + 10 variance steps each). Joint has 25 warmup updates at lr 0.05 + 75 joint updates at lr 0.025. Naive-100 uses the same 25+75 schedule; naive-300 is a budget sensitivity check with 300 updates at lr 0.03. All settings were fixed before observing test errors.',
        'Three EIV starts perturb S_u (and V for alternating) exactly as in the earlier study; kernel/noise initializations are identical. Naive has no input variances to perturb, so its three runs are deterministic timing repeats from the same initialization, not three distinct optimization starts. Selection uses final training loss only.',
        'CUDA timings synchronize the GPU; CPU uses one thread. Fit totals cover three runs and include prediction-cache construction. Total measured = fit total + one selected-model prediction; transfers, construction, warmup, diagnostics and report generation are excluded. Timing repeats are not Monte Carlo datasets.',
        'The diagonal EIV results/timings are reused from the earlier study after exact data and core-source-hash checks. Other results are newly measured. This is a single dataset in each setting; it cannot establish general superiority or statistical significance.',
        'Tridiagonal misspecification evaluates this particular correlation pattern and response. The response has no mixed quadratic terms, limiting the conclusions about other correlation-sensitive functions.',
        '', 'Environment: '+str(result['environment'])]
    for scenario,study in result['studies'].items():
        lines+=['',f'## {scenario.capitalize()} true input covariance','',
            '| Device | Method | Runs | Fit total (s) | Predict (s) | Total measured (s) | RMSE | MAE | R2 | RMSE gain vs naive-100 |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        rows=study['summaries']
        for r in rows:
            base=next((b for b in rows if b['device']==r['device'] and b['method']=='naive_100' and b.get('prediction_status')=='completed'),None)
            if r.get('prediction_status')=='completed':
                gain=f"{100*(1-r['rmse']/base['rmse']):.3f}%" if base else '--'
                lines.append(f"| {r['device']} | {r['method']} | {r['successful']}/3 | {r['total_fit_seconds']:.3f} | {r['prediction_seconds']:.3f} | {r['total_measured_seconds']:.3f} | {r['rmse']:.6f} | {r['mae']:.6f} | {r['r2']:.6f} | {gain} |")
        lines+=['','Positive RMSE gain favors the listed method; negative values favor naive-100.','','### Noise estimation and latent intervals','',
            '| Device | Method | Marginal S_u RMSE | Relative marginal error | Full covariance relative Frobenius error | Observation std. | 95% coverage | Mean interval width |',
            '|---|---|---:|---:|---:|---:|---:|---:|']
        for r in rows:
            if r.get('prediction_status')!='completed':continue
            su=f"{r['su_rmse']:.6f}" if not r['method'].startswith('naive') else '--'
            rel=f"{100*r['su_relative_error']:.2f}%" if not r['method'].startswith('naive') else '--'
            lines.append(f"| {r['device']} | {r['method']} | {su} | {rel} | {100*r['covariance_relative_error']:.2f}% | {r['sigma_obs']:.6f} | {100*r['coverage']:.1f}% | {r['interval_width']:.6f} |")
        lines+=['','Naive assumes zero input covariance (100% covariance error by definition); marginal parameter recovery is not applicable. Full covariance error includes omitted off-diagonal entries for diagonal EIV fits. Coverage is descriptive for one fixed test set.','','### Comparison to the benchmark','']
        for device in result['devices']:
            candidates=[r for r in rows if r['device']==device and r.get('prediction_status')=='completed']
            base=next((r for r in candidates if r['method']=='naive_100'),None)
            sensitivity=next((r for r in candidates if r['method']=='naive_300'),None)
            if base:
                for r in candidates:
                    if r['method'] in ('alternating','joint'):
                        lines.append(f"- {device} {r['method']}: RMSE difference from naive-100 {r['rmse']-base['rmse']:+.6f}; relative improvement {100*(1-r['rmse']/base['rmse']):+.3f}%.")
                        if sensitivity:lines.append(f"  Against naive-300: relative improvement {100*(1-r['rmse']/sensitivity['rmse']):+.3f}%.")
        lines+=['','### All fit outcomes','','| Device | Method | Run | Status | Fit seconds | Final training loss |','|---|---|---:|---|---:|---:|']
        for r in study['runs']:
            loss=f"{r['loss']:.6f}" if 'loss' in r else '--'
            lines.append(f"| {r['device']} | {r['method']} | {r['restart']} | {r['status']} | {r['fit_seconds']:.3f} | {loss} |")
            if 'error' in r:lines.append('\nFailure: '+r['error'])
        for r in rows:
            if 'prediction_error' in r:lines.append('\nPrediction failure: '+r['prediction_error'])
    lines+=['','## Reproducibility','',
        'Run `python simulation_20d_naive.py` from the project root. Completed method/device groups are resumed only when the saved experiment signature matches code, configuration and source diagonal results. Raw data, true covariance matrices, initializations, histories, predictions and source hashes are saved in simulation_20d_naive_results.json. The original simulation_20d_results.json and report are preserved.']
    (ROOT/'simulation_20d_naive_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    old=json.loads((ROOT/'simulation_20d_results.json').read_text())
    for name in ('nngp_torch.py','nngp_delta.py'):
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=old['source_hashes'][name]:
            raise RuntimeError('Core source changed; cannot reuse diagonal EIV results')
    diagonal=generate_data()
    for key,t in zip(('X','y','X_test','target'),diagonal[:4]):
        torch.testing.assert_close(t,torch.tensor(old['data'][key],dtype=t.dtype),rtol=0,atol=0)
    devices=['cpu']+(['cuda'] if torch.cuda.is_available() else [])
    hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ('simulation_20d_naive.py','simulation_20d.py','nngp_torch.py','nngp_delta.py','simulation_20d_results.json')}
    signature=hashlib.sha256(json.dumps(dict(hashes=hashes,devices=devices,ALT=ALT,JOINT=JOINT),sort_keys=True).encode()).hexdigest()
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),signature=signature,source_hashes=hashes,devices=devices,environment=dict(python=platform.python_version(),torch=torch.__version__,cpu=platform.processor(),gpu=torch.cuda.get_device_name(0) if 'cuda' in devices else None,dtype='float64',cpu_threads=1),studies={})
    if OUTPUT.exists():
        saved=json.loads(OUTPUT.read_text())
        if saved['signature']!=signature:raise RuntimeError('Existing results have different signature; archive them before rerunning')
        result=saved
    def save():
        OUTPUT.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8');write_report(result)
    for scenario,data in (('diagonal',diagonal),('tridiagonal',tridiagonal_data())):
        x,y,test,target,truth=data
        sigma=truth.diag() if scenario=='diagonal' else covariance(truth)
        if scenario not in result['studies']:
            study=dict(data={k:vector(v) for k,v in zip(('X','y','X_test','target'),data[:4])},true_covariance=vector(sigma),runs=[],summaries=[])
            if scenario=='diagonal':
                study['runs']=copy.deepcopy(old['runs']);study['summaries']=copy.deepcopy(old['summaries'])
                for r in study['summaries']:r['provenance']='reused simulation_20d_results.json'
            result['studies'][scenario]=study
        study=result['studies'][scenario]
        for r in study['summaries']:
            estimate=torch.tensor(r['S_u'],dtype=truth.dtype).diag()
            r['covariance_relative_error']=((estimate-sigma).norm()/sigma.norm()).item()
        for method in METHODS:
            for device_name in devices:
                if any(r['method']==method and r['device']==device_name for r in study['summaries']):continue
                device=torch.device(device_name);xd,yd,td=x.to(device),y.to(device),test.to(device)
                def make(start):
                    k=GenArccos(20,2,[0.5,0.5],[2.,1.5])
                    return (NaiveNNGP(k) if method.startswith('naive') else NNGP_EIV(k,20,S_u_init=start[0],V_init=start[1],inference_method=method)).to(device)
                initial=starts(method)
                warm=make(initial[0]);warm.fit(xd,yd,**(dict(steps=100) if method.startswith('naive') else dict(outer_iters=1,inner_kernel_steps=1,inner_su_steps=1) if method=='alternating' else dict(iter_warmup=1,iter_joint=1)))
                sync(device);del warm
                candidates=[];group=[]
                # Remove incomplete group's prior records before resuming it.
                study['runs']=[r for r in study['runs'] if (r['method'],r['device'])!=(method,device_name)]
                for restart,start in enumerate(initial,1):
                    model=make(start);record=dict(method=method,device=device_name,restart=restart,initial_S_u=None if method.startswith('naive') else start[0],initial_V=start[1] if method=='alternating' else None)
                    print(f'{scenario} {method} {device_name} run {restart}: fitting',flush=True)
                    sync(device);began=time.perf_counter()
                    try:
                        model.fit(xd,yd,**(dict(steps=int(method.split('_')[1])) if method.startswith('naive') else ALT if method=='alternating' else JOINT))
                        sync(device);record['fit_seconds']=time.perf_counter()-began
                        loss=model(xd,yd).item()
                        if not math.isfinite(loss):raise RuntimeError('Nonfinite final loss')
                        record.update(status='completed',loss=loss,sigma_obs=model.sigma_obs.item(),history={k:[a.tolist() if hasattr(a,'tolist') else a for a in values] for k,values in model.history.items()})
                        if not method.startswith('naive'):record['S_u']=vector(model.S_u)
                        candidates.append((loss,restart,model))
                    except (RuntimeError,ValueError) as error:
                        sync(device);record.setdefault('fit_seconds',time.perf_counter()-began);record.update(status='failed',error=str(error))
                    group.append(record);study['runs'].append(record);save()
                    print(f"  {record['status']} {record['fit_seconds']:.3f}s",flush=True)
                summary=dict(method=method,device=device_name,successful=len(candidates),total_fit_seconds=sum(r['fit_seconds'] for r in group),provenance='new run')
                if candidates:
                    loss,restart,model=min(candidates,key=lambda t:t[0])
                    estimate=torch.zeros_like(truth) if method.startswith('naive') else model.S_u.detach().cpu()
                    summary.update(selected_restart=restart,loss=loss,S_u=vector(estimate),sigma_obs=model.sigma_obs.item(),su_rmse=(estimate-truth).square().mean().sqrt().item(),su_relative_error=((estimate-truth).abs()/truth).mean().item(),covariance_relative_error=((estimate.diag()-sigma).norm()/sigma.norm()).item())
                    try:
                        sync(device);began=time.perf_counter();mean,var=model.predict(td,include_observation_noise=False);sync(device);elapsed=time.perf_counter()-began
                        mean,var=mean.cpu(),var.cpu();error=mean-target
                        summary.update(prediction_status='completed',prediction_seconds=elapsed,total_measured_seconds=summary['total_fit_seconds']+elapsed,rmse=error.square().mean().sqrt().item(),mae=error.abs().mean().item(),r2=1-error.square().sum().item()/(target-target.mean()).square().sum().item(),coverage=(error[:,0].abs()<=1.95996398454*var.sqrt()).double().mean().item(),interval_width=(2*1.95996398454*var.sqrt()).mean().item(),mean=vector(mean),latent_variance=vector(var))
                    except RuntimeError as error:summary.update(prediction_status='failed',prediction_error=str(error))
                study['summaries'].append(summary);save()
                print('SUMMARY '+json.dumps({k:v for k,v in summary.items() if k not in ('mean','latent_variance','S_u')}),flush=True)
    print('Both scenarios complete.',flush=True)

if __name__=='__main__': main()
