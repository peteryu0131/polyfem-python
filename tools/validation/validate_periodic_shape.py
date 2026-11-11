"""Project gradient checks for reduced periodic parameters on the original model."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import torch
import polyfem.diff as diff
from tools.validation._boundary_config import ROOT, configuration


def validate(output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    report={'case':'homogenize-stress-periodic','passed':False,'checks':[], 'rtol':2.3e-7,'atol':0.,
            'scope':'Project periodic-parameter checks on original model; upstream Windows-excluded C++ test not run',
            'adaptations':['absolute paths','one thread','original SPQR retained','separate output',
                           'PyTorch RNG seed 42','three reduced-interior directions and one affine direction',
                           'second FD step; not identical to original full-vertex-index direction construction'],
            'commits':{n:subprocess.check_output(['git','-C',str(ROOT/n),'rev-parse','HEAD'],text=True).strip()
                       for n in ('polyfem','differentiability-data')}}
    try:
        spec=json.loads((ROOT/'differentiability-data/input/homogenize-stress-periodic-opt.json').read_text())
        config=configuration('homogenize-stress-periodic',output/'forward')
        original=json.loads((ROOT/'differentiability-data/input/homogenize-stress-periodic.json').read_text())
        config['solver']['linear']=original['solver']['linear']
        f=spec['functionals'][0]
        objective=diff.objectives.create(f['type'],**{k:v for k,v in f.items() if k!='type'})
        obj=diff.periodic_shape(config,objective=objective)
        x=obj.initial_parameters.requires_grad_()
        loss=lambda p:obj(parameters=p)
        value=loss(x);gradient,=torch.autograd.grad(value,x)
        report.update(parameter_count=x.numel(),initial=x.tolist(),value=value.item(),gradient=gradient.tolist(),config=config)
        coords=x.detach()[:-4].reshape(-1,2)
        boundary=((coords<coords.amin(0)+1e-5)|(coords>coords.amax(0)-1e-5)).any(1)
        generator=torch.Generator().manual_seed(42)
        directions=[]
        for i in range(3):
            direction=torch.zeros_like(x)
            direction[:-4]=torch.rand(coords.shape,dtype=x.dtype,generator=generator).masked_fill(boundary[:,None],0).flatten()
            directions.append((f'interior_{i}',direction))
        affine=torch.zeros_like(x);affine[-4]=.1;affine[-1]=.2
        directions.append(('affine_diagonal',affine))
        for name,direction in directions:
            actual=(gradient*direction).sum().item()
            for h in (1e-7,3e-7):
                with torch.no_grad():fd=((loss(x+h*direction)-loss(x-h*direction))/(2*h)).item()
                error=abs(actual-fd)/max(abs(actual),abs(fd),1e-30)
                check={'direction':name,'h':h,'autograd':actual,'finite_difference':fd,'relative_error':error,'passed':error<=report['rtol']}
                report['checks'].append(check);print(json.dumps(check),flush=True)
                (output/'progress.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
        scale=torch.zeros((),dtype=x.dtype,requires_grad=True)
        outer,=torch.autograd.grad(loss(x.detach()+scale*affine).square(),scale)
        expected=2*value.item()*(gradient*affine).sum().item()
        report['mapping']={'autograd':outer.item(),'expected':expected,'passed':abs(outer.item()-expected)<=1e-10+1e-9*abs(expected)}
        report['passed']=all(c['passed'] for c in report['checks']) and report['mapping']['passed']
    except Exception as e:
        report['error']=f'{type(e).__name__}: {e}';print(report['error'],flush=True)
    finally:
        report['elapsed_seconds']=time.perf_counter()-started
        (output/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    raise SystemExit(0 if validate(p.parse_args().output)['passed'] else 1)
