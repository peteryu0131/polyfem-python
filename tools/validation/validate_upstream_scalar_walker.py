"""Validate original scalar Laplacian and walker shape gradients."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import torch
from tools.validation._boundary_config import ROOT
from tools.validation._scalar_walker_config import load_case


CASES = {'laplacian': (1, 8.3e-7, 0., 1.), 'walker-shape': (3, 2.2e-4, 0., 1.)}


def validate(case, output, steps=(1e-7, 3e-7)):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    repeat, tolerance, low, high = CASES[case]
    report = {'case':case, 'passed':False, 'rtol':tolerance, 'atol':0., 'checks':[], 'fd_steps':list(steps),
              'scope':'Original model adapted to Python, not original C++ executable or optimizer convergence',
              'adaptations':['absolute resource paths', 'one thread', 'SimplicialLDLT', 'separate output',
                             'PyTorch RNG seed 42 with original direction count and range',
                             'Laplacian uses exact original V*cos(V) direction, no random direction',
                             'additional FD step', 'walker boundary edge centers reconstructed from original OBJ',
                             'separate public objective calls summed in Python; component checks added'],
              'commits':{n:subprocess.check_output(['git','-C',str(ROOT/n),'rev-parse','HEAD'],text=True).strip()
                         for n in ('polyfem','differentiability-data')}}
    try:
        objects, vertices, ids, config, objectives, groups, metadata = load_case(case, output/'forward')
        x = vertices[ids].clone().requires_grad_()

        def components(x):
            V = vertices.index_copy(0, ids, x)
            return torch.stack([obj(V=V) for obj in objects])

        report.update(vertex_count=len(vertices), selected_vertex_ids=ids.tolist(),
                      parameter_shape=list(x.shape), objectives=objectives, boundary_groups=groups, model_metadata=metadata,
                      boundary_conditions=config['boundary_conditions'], forward_solver=config['solver'])
        # Differentiate each native solve once; retained native graphs are unsupported.
        initial_values, initial_gradients = [], []
        for obj in objects:
            value = obj(V=vertices.index_copy(0, ids, x))
            initial_gradients.append(torch.autograd.grad(value, x)[0])
            initial_values.append(value.detach())
        values = torch.stack(initial_values)
        gradients = torch.stack(initial_gradients)
        report.update(values=values.detach().tolist(), total_value=values.sum().item(), gradients=gradients.tolist())
        generator = torch.Generator().manual_seed(42)
        for trial in range(repeat):
            direction = x.detach()*torch.cos(x.detach()) if case == 'laplacian' else low+(high-low)*torch.rand(x.shape,dtype=x.dtype,generator=generator)
            actual = (gradients*direction).sum(dim=(1,2))
            for h in steps:
                with torch.no_grad():
                    fd = (components(x+h*direction)-components(x-h*direction))/(2*h)
                for label, a, b in [(str(i), actual[i].item(), fd[i].item()) for i in range(len(objects))]+[
                    ('sum',actual.sum().item(),fd.sum().item())]:
                    error=abs(a-b)/max(abs(a),abs(b),1e-30)
                    check={'trial':trial,'h':h,'component':label,'autograd':a,'finite_difference':b,
                           'relative_error':error,'passed':error<=tolerance}
                    report['checks'].append(check)
                    print(json.dumps(check),flush=True)
                report.setdefault('directions',{})[str(trial)]=direction.tolist()
                (output/'progress.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
        scalar=torch.zeros((),dtype=x.dtype,requires_grad=True)
        outer,=torch.autograd.grad(components(x.detach()+scalar*direction).sum().square(),scalar)
        expected=(2*values.detach().sum()*(gradients*direction).sum()).item()
        report['mapping']={'autograd':outer.item(),'expected':expected,
                           'passed':abs(outer.item()-expected)<=1e-12+1e-9*abs(expected)}
        report['passed']=all(c['passed'] for c in report['checks']) and report['mapping']['passed']
    except Exception as exc:
        report['error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['elapsed_seconds']=time.perf_counter()-start
        (output/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    if not report['passed']:
        raise AssertionError('Scalar/walker shape checks failed; see result.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=CASES,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=float,nargs='+',default=[1e-7,3e-7],help='Alternative steps for a separate diagnostic run')
    args=parser.parse_args()
    validate(args.case,args.output,args.steps)
