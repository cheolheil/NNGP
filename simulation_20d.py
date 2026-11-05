"""Dedicated 20D Berkson-error study (100 training samples).

Run: python simulation_20d.py
Outputs simulation_20d_results.json and simulation_20d_report.md.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import hashlib
import json
import math
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
import torch
from nngp_torch import GenArccos, NNGP_EIV

ROOT = Path(__file__).resolve().parent
ALT = dict(outer_iters=15, inner_kernel_steps=10, inner_su_steps=10, lr_kernel=0.03, lr_su=0.03)
JOINT = dict(iter_warmup=25, iter_joint=75, lr=0.05)

def generate_data():
    g = torch.Generator().manual_seed(42)
    dtype = torch.float64
    variance = torch.linspace(0.02, 0.08, 20, dtype=dtype)
    weights = torch.linspace(0.5, 1.5, 20, dtype=dtype)
    def latent(z):
        return (2 - 0.25 * (z.square() * weights).sum(-1, keepdim=True) / math.sqrt(20)
                + 0.5 * z[:, :1].sin() + 0.5 * z[:, 1:2].cos())
    x = torch.rand(100, 20, generator=g, dtype=dtype) * 6 - 3
    physical = x + torch.randn(100, 20, generator=g, dtype=dtype) * variance.sqrt()
    y = latent(physical) + 0.1 * torch.randn(100, 1, generator=g, dtype=dtype)
    test = torch.rand(200, 20, generator=g, dtype=dtype) * 6 - 3
    target = latent(test)
    return x, y, test, target, variance

def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)

def vector(t):
    return t.detach().cpu().tolist()

def report(result):
    rows = result['summaries']
    lines = ['# 20D NNGP-EIV: CPU versus GPU', '', '## Problem and protocol', '',
        '100 training observations; 200 independent exact-input test points; 20 dimensions; seed 42.',
        'Nominal inputs are uniform on [-3,3]^20. Physical inputs W=X+u, with independent Gaussian Berkson errors.',
        'True coordinate variances increase linearly from 0.02 to 0.08. Observation standard deviation is 0.1.',
        'The latent response is f(z)=2-0.25 sum_d(a_d z_d^2)/sqrt(20)+0.5 sin(z_1)+0.5 cos(z_2), with a_d increasing linearly from 0.5 to 1.5.', '',
        '- Alternating minimizes negative approximate ELBO: second-order expected log likelihood plus input KL. Taylor truncation does not guarantee a bound.',
        '- Joint minimizes moment Gaussian NLL based on the second-order expected kernel.',
        '- Three matched initializations per method/device. Initial S_u=0.01, V=0.001; later starts use matched CPU-generated lognormal perturbations. Restart selection uses final training loss only.',
        '- Alternating: 15 outer iterations, 10 kernel/noise and 10 S_u/V steps (300 updates), learning rates 0.03. Joint: 25 warmup and 75 joint steps (100 updates), learning rates 0.05 and 0.025. Objective values are not comparable between methods.',
        '- Float64 on both devices; one CPU thread. Warmup excluded. CUDA synchronized around timings. Fit includes history and prediction-cache construction. Model construction and data transfers excluded.',
        '- Prediction timing is one call on the selected restart, including latent covariance/interval computation. Total measured time is all three fits plus selected-model prediction; diagnostics and report generation are excluded.',
        '- This is one synthetic dataset with optimization restarts, not a repeated-dataset Monte Carlo study. Test intervals exclude observation and test-input noise.', '',
        'Environment: '+str(result['environment']), '', '## Selected-restart results', '',
        '| Device | Method | Successful fits | Selected restart | Total fit (s) | Predict (s) | Total measured (s) | RMSE | MAE | R2 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        if r.get('prediction_status') == 'completed':
            lines.append(f"| {r['device']} | {r['method']} | {r['successful']}/3 | {r['selected_restart']} | {r['total_fit_seconds']:.3f} | {r['prediction_seconds']:.3f} | {r['total_measured_seconds']:.3f} | {r['rmse']:.5f} | {r['mae']:.5f} | {r['r2']:.5f} |")
        else:
            lines.append(f"| {r['device']} | {r['method']} | {r['successful']}/3 | {r.get('selected_restart','--')} | {r['total_fit_seconds']:.3f} | -- | -- | -- | -- | -- |")
    lines += ['', '## Input-noise inference and intervals', '', '| Device | Method | S_u RMSE | Mean relative S_u error | Observation std. | 95% latent coverage | Mean width | Min training eigenvalue |', '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        if r.get('prediction_status') == 'completed':
            lines.append(f"| {r['device']} | {r['method']} | {r['su_rmse']:.5f} | {100*r['su_relative_error']:.2f}% | {r['sigma_obs']:.5f} | {100*r['coverage']:.1f}% | {r['interval_width']:.5f} | {r['min_eigenvalue']:.6g} |")
    lines += ['', 'Coverage is descriptive for this fixed test set, not evidence of repeated-sample calibration.', '', '## Coordinate-wise input variance estimates', '', '| Coordinate | True S_u | CPU alternating | GPU alternating | CPU joint | GPU joint |', '|---|---:|---:|---:|---:|---:|']
    for d,truth in enumerate(result['true_S_u']):
        values=[]
        for method in ('alternating','joint'):
            for device in ('cpu','cuda'):
                r=next((r for r in rows if r['method']==method and r['device']==device),{})
                values.append(f"{r['S_u'][d]:.6f}" if 'S_u' in r else '--')
        lines.append(f"| {d+1} | {truth:.6f} | "+' | '.join(values)+' |')
    lines += ['', '## Device speed comparison', '']
    for method in ('alternating','joint'):
        c=next((r for r in rows if r['device']=='cpu' and r['method']==method),None)
        g=next((r for r in rows if r['device']=='cuda' and r['method']==method),None)
        if c and g and c['successful']==g['successful']==3:
            lines.append(f"- {method}: CPU/GPU fit-time ratio {c['total_fit_seconds']/g['total_fit_seconds']:.3f}x (above 1 favors GPU).")
            if c.get('prediction_status')==g.get('prediction_status')=='completed':
                lines.append(f"  Selected-model CPU/GPU RMSE difference: {abs(c['rmse']-g['rmse']):.3g}.")
    lines += ['', '## Interpretation', '']
    cpu_rows = [r for r in rows if r['device'] == 'cpu' and r.get('prediction_status') == 'completed']
    for r in cpu_rows:
        lines.append(f"- {r['method']}: test RMSE {r['rmse']:.5f}, R2 {r['r2']:.5f}; mean relative input-variance error {100*r['su_relative_error']:.2f}%. Estimated observation std. {r['sigma_obs']:.5f} versus true 0.1.")
    if len(cpu_rows) == 2:
        lines.append('Both methods have similar prediction accuracy on this dataset, but poor recovery of coordinate-wise input noise. This run does not establish convergence or identifiability; larger samples, more optimization and repeated datasets would be separate experiments.')
    lines += ['', '## All restart outcomes', '', '| Device | Method | Restart | Status | Fit (s) | Final loss |', '|---|---|---:|---|---:|---:|']
    for r in result['runs']:
        lines.append(f"| {r['device']} | {r['method']} | {r['restart']} | {r['status']} | {r['fit_seconds']:.3f} | "+(f"{r['loss']:.6f}" if 'loss' in r else '--')+' |')
        if 'error' in r: lines.append('\nFailure: '+r['error']+'\n')
    for r in rows:
        if 'prediction_error' in r: lines.append('\nPrediction failure: '+r['prediction_error'])
    lines += ['', 'Raw data, histories, initializations, estimates, predictions, per-run timings, and source hashes are retained in simulation_20d_results.json. V is a variational parameter, not a direct estimate of the true input variance. With 20 input variances and only 100 observations, noise recovery may be weak even when predictions are useful. CPU/GPU comparisons apply to these workloads and this single-thread CPU configuration.']
    (ROOT/'simulation_20d_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    x,y,test,target,truth=generate_data()
    devices=['cpu']+(['cuda'] if torch.cuda.is_available() else [])
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),environment=dict(python=platform.python_version(),torch=torch.__version__,cpu=platform.processor(),gpu=torch.cuda.get_device_name(0) if 'cuda' in devices else None,dtype='float64',cpu_threads=1),
        settings=dict(alternating=ALT,joint=JOINT),true_S_u=vector(truth),data=dict(X=vector(x),y=vector(y),X_test=vector(test),target=vector(target)),
        source_hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ('simulation_20d.py','nngp_torch.py','nngp_delta.py')},runs=[],summaries=[])
    def save():
        (ROOT/'simulation_20d_results.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
        report(result)
    for method in ('alternating','joint'):
        generator=torch.Generator().manual_seed(42)
        starts=[]
        for r in range(3):
            su=torch.full((20,),0.01,dtype=torch.float64)
            v=torch.full((20,),0.001,dtype=torch.float64)
            if r:
                su*=torch.exp(torch.randn(20,generator=generator,dtype=torch.float64)*0.5)
                if method=='alternating': v*=torch.exp(torch.randn(20,generator=generator,dtype=torch.float64)*0.5)
            starts.append((su.tolist(),v.tolist()))
        for device_name in devices:
            device=torch.device(device_name)
            xd,yd,td=x.to(device),y.to(device),test.to(device)
            def make(start):
                return NNGP_EIV(GenArccos(20,2,[0.5,0.5],[2.,1.5]),20,S_u_init=start[0],V_init=start[1],inference_method=method).to(device)
            warm=make(starts[0])
            warm.fit(xd,yd,**(dict(outer_iters=1,inner_kernel_steps=1,inner_su_steps=1) if method=='alternating' else dict(iter_warmup=1,iter_joint=1)))
            sync(device);del warm
            candidates=[]
            group=[]
            for restart,start in enumerate(starts,1):
                model=make(start)
                record=dict(method=method,device=device_name,restart=restart,initial_S_u=start[0],initial_V=start[1])
                print(f'{method} {device_name} restart {restart}: fitting',flush=True)
                sync(device);began=time.perf_counter()
                try:
                    model.fit(xd,yd,**(ALT if method=='alternating' else JOINT))
                    sync(device);record['fit_seconds']=time.perf_counter()-began
                    loss=model(xd,yd).item()
                    if not math.isfinite(loss): raise RuntimeError('Nonfinite final loss')
                    record.update(status='completed',loss=loss,S_u=vector(model.S_u),sigma_obs=model.sigma_obs.item(),V=vector(model.V) if method=='alternating' else None,
                        history={k:[a.tolist() if hasattr(a,'tolist') else a for a in values] for k,values in model.history.items()})
                    candidates.append((loss,restart,model))
                except (RuntimeError,ValueError) as error:
                    sync(device);record.setdefault('fit_seconds',time.perf_counter()-began)
                    record.update(status='failed',error=str(error))
                group.append(record);result['runs'].append(record);save()
                print(f"  {record['status']}: {record['fit_seconds']:.3f}s",flush=True)
            summary=dict(method=method,device=device_name,successful=len(candidates),total_fit_seconds=sum(r['fit_seconds'] for r in group))
            if candidates:
                loss,restart,model=min(candidates,key=lambda t:t[0])
                summary.update(selected_restart=restart,loss=loss,S_u=vector(model.S_u),V=vector(model.V) if method=='alternating' else None,sigma_obs=model.sigma_obs.item(),su_rmse=(model.S_u.cpu()-truth).square().mean().sqrt().item(),su_relative_error=((model.S_u.cpu()-truth).abs()/truth).mean().item())
                try:
                    sync(device);began=time.perf_counter()
                    mean,var=model.predict(td,include_observation_noise=False)
                    sync(device);elapsed=time.perf_counter()-began
                    mean,var=mean.cpu(),var.cpu();error=mean-target
                    covariance=model.kernel(xd)+model._compute_delta_correction(xd)+model.sigma_obs.square()*torch.eye(100,device=device,dtype=xd.dtype)
                    summary.update(prediction_status='completed',prediction_seconds=elapsed,total_measured_seconds=summary['total_fit_seconds']+elapsed,
                        rmse=error.square().mean().sqrt().item(),mae=error.abs().mean().item(),r2=1-error.square().sum().item()/(target-target.mean()).square().sum().item(),coverage=(error[:,0].abs()<=1.95996398454*var.sqrt()).double().mean().item(),interval_width=(2*1.95996398454*var.sqrt()).mean().item(),min_eigenvalue=torch.linalg.eigvalsh(covariance).min().item(),mean=vector(mean),latent_variance=vector(var))
                except RuntimeError as error:
                    summary.update(prediction_status='failed',prediction_error=str(error))
            result['summaries'].append(summary);save()
            print('SUMMARY '+json.dumps({k:v for k,v in summary.items() if k not in ('mean','latent_variance','S_u','V')}),flush=True)
    print('Study complete.',flush=True)

if __name__=='__main__':
    main()
