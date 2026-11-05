"""Matched CPU/CUDA timing: python gpu_cpu_benchmark.py."""
import json
import platform
import statistics
import time
from pathlib import Path
import torch
from comparison_benchmark import generate_data, ALT_SETTINGS, JOINT_SETTINGS
from nngp_torch import GenArccos, NNGP_EIV

ROOT = Path(__file__).resolve().parent

def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)

def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
    result = dict(environment=dict(python=platform.python_version(), torch=torch.__version__,
                  cpu=platform.processor(), gpu=torch.cuda.get_device_name(0) if 'cuda' in devices else None,
                  cpu_threads=1, dtype='float64'), settings=dict(alternating=ALT_SETTINGS, joint=JOINT_SETTINGS),
                  protocol='Seed 42; identical CPU-generated data and initial parameters; 3 paired restarts; full fit includes prediction cache and history; transfers, model construction and warmup excluded; prediction median of 5 calls per restart; CUDA synchronized.', runs=[])
    def save():
        (ROOT/'gpu_cpu_results.json').write_text(json.dumps(result, indent=2))
    for dim in (1,4):
        data=generate_data(dim)
        x,y,test,target,_,fingerprint=data
        for method in ('alternating','joint'):
            generator=torch.Generator().manual_seed(42)
            initial_su=torch.tensor([0.01] if dim==1 else [0.01,0.015,0.008,0.012],dtype=torch.float64)
            initial_v=torch.full((dim,),0.001,dtype=torch.float64)
            starts=[]
            for restart in range(3):
                su=initial_su if restart==0 else initial_su*torch.exp(torch.randn(dim,generator=generator,dtype=torch.float64)*0.5)
                v=initial_v if restart==0 or method=='joint' else initial_v*torch.exp(torch.randn(dim,generator=generator,dtype=torch.float64)*0.5)
                starts.append((su.tolist(),v.tolist()))
            for device_name in devices:
                device=torch.device(device_name)
                xd,yd,td=x.to(device),y.to(device),test.to(device)
                def make(su,v):
                    return NNGP_EIV(GenArccos(dim,2,[0.5,0.5],[2.,1.5]),dim,
                        S_u_init=su,V_init=v,sigma_obs_init=0.1,inference_method=method).to(device)
                warm=make(*starts[0])
                warm.fit(xd,yd,**(dict(outer_iters=1,inner_kernel_steps=1,inner_su_steps=1) if method=='alternating' else dict(iter_warmup=1,iter_joint=1)))
                warm.predict(td,include_observation_noise=False)
                sync(device)
                del warm
                for restart,(su,v) in enumerate(starts,1):
                    record=dict(dimension=dim,n_train=len(x),n_test=len(test),method=method,device=device_name,restart=restart,data_sha256=fingerprint)
                    model=make(su,v)
                    sync(device); started=time.perf_counter()
                    try:
                        model.fit(xd,yd,**(ALT_SETTINGS if method=='alternating' else JOINT_SETTINGS))
                        sync(device);record['fit_seconds']=time.perf_counter()-started
                        times=[]
                        for _ in range(5):
                            sync(device);started=time.perf_counter()
                            mean,var=model.predict(td,include_observation_noise=False)
                            sync(device);times.append(time.perf_counter()-started)
                        record.update(prediction_seconds=statistics.median(times),prediction_samples=times,
                            loss=model(xd,yd).item(),rmse=(mean.cpu()-target).square().mean().sqrt().item(),
                            S_u=model.S_u.detach().cpu().tolist(),status='completed')
                    except Exception as error:
                        sync(device)
                        record.update(status='failed',error=str(error))
                    result['runs'].append(record);save()
                    print(json.dumps(record),flush=True)
    lines=['# CPU versus GPU simulation timings','',str(result['environment']),'',result['protocol'],'',
           '| Dimension | Method | CPU fit total (s) | GPU fit total (s) | CPU/GPU speedup | CPU predict (s) | GPU predict (s) |',
           '|---|---|---:|---:|---:|---:|---:|']
    for dim in (1,4):
        for method in ('alternating','joint'):
            grouped={d:[r for r in result['runs'] if r['dimension']==dim and r['method']==method and r['device']==d and r['status']=='completed'] for d in devices}
            if all(len(grouped.get(d,[]))==3 for d in ('cpu','cuda')):
                c,g=grouped['cpu'],grouped['cuda'];ct=sum(r['fit_seconds'] for r in c);gt=sum(r['fit_seconds'] for r in g)
                lines.append(f"| {dim}D | {method} | {ct:.3f} | {gt:.3f} | {ct/gt:.3f}x | {statistics.median(r['prediction_seconds'] for r in c):.6f} | {statistics.median(r['prediction_seconds'] for r in g):.6f} |")
    lines+=['','Fit totals cover three restarts. Speedup below 1 means GPU is slower. CPU uses one thread to match the existing simulation protocol. These are small-data workloads; results do not establish scaling at larger N. See JSON for individual times, losses, RMSE, failures and settings.']
    (ROOT/'gpu_cpu_report.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    main()
