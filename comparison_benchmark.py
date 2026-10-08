"""Reproduce the corrected NNGP-EIV benchmark and generate both reports.

Run: python comparison_benchmark.py
Uses the original seed, data sizes, three restarts, and optimizer budgets.
"""
import argparse
import hashlib
import html
import json
import math
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from nngp_torch import GenArccos, NNGP_EIV, set_seed
from nngp_torch_joint import NNGP_EIV_Joint

ROOT = Path(__file__).resolve().parent
ALT_SETTINGS = dict(outer_iters=15, inner_kernel_steps=10, inner_su_steps=10, lr_kernel=0.03, lr_su=0.03)
JOINT_SETTINGS = dict(iter_warmup=25, iter_joint=75, lr=0.05)


def generate_data(dim, seed=42):
    set_seed(seed)
    dtype=torch.float64
    if dim==1:
        n=60; true_variance=torch.tensor([0.05],dtype=dtype)
        x=torch.linspace(-4,4,n,dtype=dtype)[:,None]
    else:
        n=100; true_variance=torch.tensor([0.05,0.03,0.08,0.04],dtype=dtype)
        x=torch.rand(n,dim,dtype=dtype)*6-3
    w=x+torch.randn_like(x)*true_variance.sqrt()
    def latent(z):
        if dim==1:return -0.25*z.square()+2
        return (-0.25*(z.square()*torch.tensor([1.,0.8,1.2,0.5],dtype=dtype)).sum(-1,keepdim=True)
                +0.5*z[:,0:1].sin()+0.5*z[:,1:2].cos()+2)
    y=latent(w)+torch.randn(n,1,dtype=dtype)*0.1
    test=torch.linspace(-5,5,100,dtype=dtype)[:,None] if dim==1 else torch.rand(200,dim,dtype=dtype)*7-3.5
    target=latent(test)
    fingerprint=hashlib.sha256(b''.join(t.numpy().tobytes() for t in (x,y,test,target))).hexdigest()
    return x,y,test,target,true_variance,fingerprint


def vector(t):return t.detach().cpu().tolist()


def run_method(dim, method, data, restarts):
    x,y,test,target,true_variance,_=data
    initial_su=torch.tensor([0.01] if dim==1 else [0.01,0.015,0.008,0.012],dtype=x.dtype)
    initial_v=torch.full((dim,),0.001,dtype=x.dtype)
    set_seed(42)
    results=[]; candidates=[]
    for restart in range(restarts):
        su=initial_su if restart==0 else initial_su*torch.exp(torch.randn(dim,dtype=x.dtype)*0.5)
        v=initial_v if restart==0 else (initial_v*torch.exp(torch.randn(dim,dtype=x.dtype)*0.5) if method=='alternating' else initial_v)
        kernel=GenArccos(dim,2,[0.5,0.5],[2.,1.5],dtype=x.dtype)
        model=(NNGP_EIV(kernel,dim,S_u_init=su.tolist(),V_init=v.tolist(),sigma_obs_init=0.1)
               if method=='alternating' else NNGP_EIV_Joint(kernel,dim,S_u_init=su.tolist(),sigma_obs_init=0.1))
        record=dict(restart=restart+1,initial_S_u=vector(su),initial_V=vector(v) if method=='alternating' else None)
        started=time.perf_counter()
        print(f'{dim}D {method} restart {restart+1}/{restarts}: training',flush=True)
        try:
            model.fit(x,y,n_restart=1,**(ALT_SETTINGS if method=='alternating' else JOINT_SETTINGS))
            record['fit_seconds']=time.perf_counter()-started
            with torch.no_grad():
                objective=model(x,y).item()
            if not math.isfinite(objective): raise RuntimeError('Non-finite final training objective')
            record.update(status='completed',selection_loss=objective,S_u=vector(model.S_u),
                          V=vector(model.V) if method=='alternating' else None,
                          sigma_obs=model.sigma_obs.item(),history=model.history)
            # Training objective alone selects the restart; test data never participates.
            candidates.append((objective,model,len(results)))
            print(f'  completed in {record["fit_seconds"]:.2f}s; selection loss {objective:.6f}',flush=True)
        except (RuntimeError,ValueError) as error:
            record.update(status='failed',fit_seconds=time.perf_counter()-started,error=str(error))
            print(f'  failed: {error}',flush=True)
        results.append(record)
    output=dict(method=method,restarts=results,total_fit_seconds=sum(r['fit_seconds'] for r in results))
    if not candidates:
        output.update(status='failed',error='No completed restart');return output
    _,model,index=min(candidates,key=lambda item:item[0])
    selected=results[index]
    output.update(status='completed',selected_restart=selected['restart'],selection_loss=selected['selection_loss'],
                  S_u=selected['S_u'],V=selected['V'],sigma_obs=selected['sigma_obs'])
    started=time.perf_counter()
    try:
        mean,variance=model.predict(test,include_observation_noise=False)
        errors=mean-target
        z=1.959963984540054
        output.update(rmse=errors.square().mean().sqrt().item(),mae=errors.abs().mean().item(),
                      latent_interval_coverage=((errors[:,0].abs()<=z*variance.sqrt()).double().mean().item()),
                      mean_latent_interval_width=(2*z*variance.sqrt()).mean().item(),
                      relative_S_u_error=(torch.abs(model.S_u-true_variance)/true_variance).mean().item(),
                      mean=vector(mean[:,0]),latent_variance=vector(variance),prediction_status='completed')
        with torch.no_grad():
            covariance=model.kernel(x)+model._compute_delta_correction(x)+model.sigma_obs.square()*torch.eye(len(x),dtype=x.dtype)
            output['min_training_covariance_eigenvalue']=torch.linalg.eigvalsh(covariance).min().item()
    except RuntimeError as error:
        output.update(prediction_status='failed',prediction_error=str(error))
    output['prediction_seconds']=time.perf_counter()-started
    return output


def fmt(value, digits=4):return '—' if value is None else f'{value:.{digits}f}'
def vec(values):return '—' if values is None else '['+', '.join(f'{v:.6g}' for v in values)+']'


def create_plot(problem):
    dim=problem['dimension']; data=problem['data']; x=np.array(data['X_test']); target=np.array(data['target'])
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for method,color,label in [('alternating','#2563eb','Alternating'),('joint','#dc2626','Joint')]:
        result=problem[method]
        if result.get('prediction_status')=='completed':
            mean=np.array(result['mean']);std=np.sqrt(result['latent_variance'])
            if dim==1:
                axes[0,0].plot(x[:,0],mean,color=color,label=f'{label}: RMSE {result["rmse"]:.3f}')
                axes[0,0].fill_between(x[:,0],mean-1.96*std,mean+1.96*std,color=color,alpha=0.13)
            else:axes[0,0].scatter(target,mean,s=13,color=color,alpha=0.5,label=f'{label}: RMSE {result["rmse"]:.3f}')
        if result.get('status')!='completed': continue
        history=result['restarts'][result['selected_restart']-1]['history']
        updates=(np.arange(len(history['S_u']))+1)*(20 if method=='alternating' else 1)+(0 if method=='alternating' else 25)
        ratios=np.array(history['S_u'])/np.array(problem['true_S_u'])
        for d in range(dim):
            axes[0,1].plot(updates,ratios[:,d],color=color,linestyle=['-','--',':','-.'][d],label=f'{label} d{d+1}')
        ax=axes[1,0] if method=='alternating' else axes[1,1]
        ax.plot(updates-1,history['loss'],color=color)
        ax.scatter([300 if method=='alternating' else 100],[result['selection_loss']],color=color,s=30,label='Final reevaluated loss')
        ax.set_title('Approximate negative ELBO' if method=='alternating' else 'Moment Gaussian NLL')
        ax.set_xlabel('Optimizer updates');ax.set_ylabel('Training loss');ax.legend(fontsize=8)
    if dim==1:
        axes[0,0].plot(x[:,0],target,'k--',label='Latent target')
        axes[0,0].scatter(np.array(data['X_train'])[:,0],data['y'],s=9,c='#555555',alpha=0.4,label='Training observations')
        axes[0,0].set_xlabel('Exact test coordinate');axes[0,0].set_ylabel('Output')
    else:
        lo,hi=float(target.min()),float(target.max());axes[0,0].plot([lo,hi],[lo,hi],'k--')
        axes[0,0].set_xlabel('Latent target');axes[0,0].set_ylabel('Predicted mean')
    axes[0,0].set_title(f'{dim}D: exact-input latent prediction');axes[0,0].legend(fontsize=8)
    axes[0,1].axhline(1,c='black',linestyle=':',label='True variance')
    axes[0,1].set_title('Selected-restart input variance trajectories')
    axes[0,1].set_xlabel('Optimizer updates');axes[0,1].set_ylabel('Estimated S_u / true S_u');axes[0,1].legend(fontsize=7,ncol=2)
    for ax in axes.flat:ax.grid(alpha=0.15)
    name=f'comparison_{dim}d.svg';fig.savefig(ROOT/name);plt.close(fig)
    path=ROOT/name;path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    return name


def write_reports(result):
    lines=['# NNGP-EIV comparison after the Delta-method correction','',
           f'Run: {result["created_utc"]}. Branch: `fix/nngpeiv-delta-method`.','',
           '## Methods and protocol','',
           '- **Alternating:** second-order approximation of the expected Gaussian log likelihood, minus the Gaussian input KL. Its training objective is an approximate ELBO; Taylor truncation does not guarantee a lower bound.',
           '- **Joint:** Gaussian negative log likelihood using the second-order expected-kernel covariance. This is a moment approximation, not the exact integrated Berkson likelihood.',
           '- Both methods use identical data for each problem, seed 42, three independent restarts, the same initial kernel and observation noise, and the original optimizer settings. Restart selection minimizes the final training loss; test metrics do not select models.',
           '- Alternating uses 15 outer iterations, 10 kernel/noise and 10 S_u/V steps each (300 updates per restart), with learning rates 0.03. Joint uses 25 kernel/noise warmup steps and 75 joint steps (100 updates), with learning rates 0.05 and 0.025.',
           '- Predictions target the latent deterministic function at exact test coordinates; intervals exclude observation noise and test input noise. Prediction covariance uses V for alternating training moments and S_u for joint training moments.',
           '- This compares two complete approximation methods with their original, unequal update budgets. It does not isolate optimizer choice, and objective values must not be ranked across methods.',
           '',f'Environment: Python {result["environment"]["python"]}; PyTorch {result["environment"]["torch"]}; CPU, float64, one thread.',
           '', '## Selected-restart results','',
           '| Problem | Method | Successful restarts | Selected restart | Total fit time (s) | Prediction time (s) | Test RMSE | Test MAE | Final selection loss |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for p in result['problems']:
        for key,label in [('alternating','Alternating / negative approx. ELBO'),('joint','Joint / moment NLL')]:
            r=p[key];completed=sum(x['status']=='completed' for x in r['restarts'])
            lines.append(f'| {p["dimension"]}D (N={p["n_train"]}) | {label} | {completed}/3 | {r.get("selected_restart","—")} | {fmt(r["total_fit_seconds"],2)} | {fmt(r.get("prediction_seconds"),2)} | {fmt(r.get("rmse"))} | {fmt(r.get("mae"))} | {fmt(r.get("selection_loss"))} |')
    lines+=['','## Input-noise inference','',
            '| Problem | Method | True S_u | Estimated S_u | V | Mean relative S_u error | Observation std. |','|---|---|---|---|---|---:|---:|']
    for p in result['problems']:
        for key in ('alternating','joint'):
            r=p[key];relative=r.get('relative_S_u_error')
            lines.append(f'| {p["dimension"]}D | {key.title()} | {vec(p["true_S_u"])} | {vec(r.get("S_u"))} | {vec(r.get("V"))} | {fmt(None if relative is None else 100*relative,1)}% | {fmt(r.get("sigma_obs"))} |')
    lines+=['','## Latent intervals and covariance diagnostics','',
            '| Problem | Method | Target inside 95% latent interval | Mean interval width | Minimum training covariance eigenvalue |','|---|---|---:|---:|---:|']
    for p in result['problems']:
        for key in ('alternating','joint'):
            r=p[key];coverage=r.get('latent_interval_coverage')
            lines.append(f'| {p["dimension"]}D | {key.title()} | {fmt(None if coverage is None else 100*coverage,1)}% | {fmt(r.get("mean_latent_interval_width"))} | {fmt(r.get("min_training_covariance_eigenvalue"),6)} |')
    lines+=['','The coverage column is descriptive coverage of a fixed test grid/sample from one synthetic dataset. It is not evidence of repeated-sample calibration. The true observation standard deviation is 0.1.','', '## All restart outcomes','',
            '| Problem | Method | Restart | Status | Fit time (s) | Final selection loss | Final S_u |','|---|---|---:|---|---:|---:|---|']
    for p in result['problems']:
        for key in ('alternating','joint'):
            for r in p[key]['restarts']:
                lines.append(f'| {p["dimension"]}D | {key.title()} | {r["restart"]} | {r["status"]} | {fmt(r["fit_seconds"],2)} | {fmt(r.get("selection_loss"))} | {vec(r.get("S_u"))} |')
                if r.get('error'):lines+=['',f'Failure: {r["error"]}','']
    lines+=['','## Interpretation','']
    for p in result['problems']:
        a,j=p['alternating'],p['joint']
        if a.get('rmse') is not None and j.get('rmse') is not None:
            winner='Alternating' if a['rmse']<j['rmse'] else 'Joint'
            lines.append(f'- **{p["dimension"]}D:** {winner} has the lower test RMSE on this dataset ({a["rmse"]:.4f} alternating versus {j["rmse"]:.4f} joint). Total fit time is {a["total_fit_seconds"]:.2f}s versus {j["total_fit_seconds"]:.2f}s.')
        for key in ('alternating','joint'):
            r=p[key]
            if r.get('prediction_status')=='failed':lines.append(f'- **{p["dimension"]}D {key}:** prediction failed: {r["prediction_error"]}')
    lines += ['- Objective histories record the loss before the final update of each logged block; the separate final point is reevaluated after all updates. Input-variance trajectories use the post-update values.',
              '- Low function-prediction error does not establish accurate input-noise recovery. The tables above report both separately.',
              '- Three restarts on a single data seed describe sensitivity to initialization, not uncertainty across datasets. No global-optimum or convergence guarantee is made.',
              '- The fixed-mean variational family remains restrictive. S_u can approach V through the KL term, and the Taylor objective need not behave well outside the small-variance regime. Second-order kernel truncation can lose positive semidefiniteness.',
              '- Timings are end-to-end fit timings on this CPU, including final objective evaluation and prediction-cache construction. Prediction timing includes metrics and covariance diagnostics. They should not be compared directly to the previous hardware-dependent report.',
              '- The optimized variational computation uses the same matrix-chain-rule Hessian as the direct-autograd reference; value and parameter-gradient equivalence are tested.',
              '', '## Plots','']
    for p in result['problems']:
        name=create_plot(p);lines += [f'### {p["dimension"]}D', '', f'![{p["dimension"]}D fits, input-variance trajectories, and objective histories]({name})','']
    lines += ['## Reproduction','', '`python comparison_benchmark.py` runs all training and writes both reports, the SVG plots, and `comparison_results.json`. `python comparison_benchmark.py --render-only` regenerates the reports from saved results without retraining. Run `python test_delta.py` for derivative and gradient checks.','',
              '### Data and source provenance','',
              f'Code baseline before this comparison update: `{result["source_commit"]}`. SHA-256 fingerprints cover the training inputs/outputs and test inputs/targets in float64 byte order.','']
    for p in result['problems']:lines.append(f'- {p["dimension"]}D: `{p["data_sha256"]}`; train N={p["n_train"]}, test N={p["n_test"]}, true S_u={vec(p["true_S_u"])}.')
    for name,digest in result['source_sha256'].items():lines.append(f'- `{name}`: `{digest}`.')
    md='\n'.join(lines)+'\n';(ROOT/'comparison_report.md').write_text(md)
    # Render Markdown without external JS, stylesheets, or image requests.
    import mistune
    rendered=mistune.create_markdown(plugins=['table'])(md)
    for p in result['problems']:
        name=f'comparison_{p["dimension"]}d.svg'
        import re
        svg=(ROOT/name).read_text();svg=svg[svg.index('<svg'):]
        rendered=re.sub(r'<img src="'+re.escape(name)+r'"[^>]*>',svg,rendered)
    document='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>NNGP-EIV corrected Delta comparison</title><style>
    body{max-width:1180px;margin:40px auto;padding:0 24px;font:16px/1.6 system-ui,sans-serif;color:#192334;background:#fff}h1{font-size:30px;line-height:1.2}h2{margin-top:40px}table{border-collapse:collapse;display:block;overflow-x:auto;width:100%;font-size:13px;margin:20px 0}th,td{border:1px solid #d6dce5;padding:9px 12px;white-space:nowrap}th{background:#eef2f8;text-align:left}tr:nth-child(even){background:#fafbfd}code{font-size:12px;overflow-wrap:anywhere}svg{width:100%;height:auto}a{color:#2456b8}li{margin:8px 0}@media print{body{margin:0;max-width:none}h2,h3{break-after:avoid}svg{break-inside:avoid}}
    </style></head><body>'''+rendered+'</body></html>'
    (ROOT/'comparison_report.html').write_text('\n'.join(line.rstrip() for line in document.splitlines())+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--render-only',action='store_true');args=parser.parse_args()
    path=ROOT/'comparison_results.json'
    if args.render_only:
        result=json.loads(path.read_text());write_reports(result);return
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),environment=dict(python=platform.python_version(),torch=torch.__version__,os=platform.platform(),dtype='float64',device='cpu',threads=1),
                source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['nngp_torch.py','nngp_torch_joint.py','nngp_delta.py','comparison_benchmark.py']},
                seed=42,n_restarts=3,alternating_settings=ALT_SETTINGS,joint_settings=JOINT_SETTINGS,problems=[])
    for dim in (1,4):
        data=generate_data(dim);x,y,test,target,true_variance,fingerprint=data
        problem=dict(dimension=dim,n_train=len(x),n_test=len(test),true_S_u=vector(true_variance),data_sha256=fingerprint,
                     data=dict(X_train=vector(x),y=vector(y[:,0]),X_test=vector(test),target=vector(target[:,0])))
        for method in ('alternating','joint'):
            problem[method]=run_method(dim,method,data,3)
        result['problems'].append(problem)
        path.write_text(json.dumps(result,indent=2,default=lambda value:value.tolist(),allow_nan=False))
    write_reports(result)
    print('Saved comparison_results.json, both reports, and plots.',flush=True)

if __name__=='__main__':main()
