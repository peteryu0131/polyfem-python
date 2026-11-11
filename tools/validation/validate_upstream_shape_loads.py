"""Check boundary-only shape gradients in two original transient load cases."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import torch
from tools.validation._boundary_config import ROOT
from tools.validation._shape_load_config import load_case


CASES = {'shape-neumann-nodes': 2e-5, 'shape-pressure-nodes-2d': 5.5e-3,
         'shape-pressure-neumann-nodes': 1e-3}


def validate(case, output, variant='historical'):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    report = {'case': case, 'passed': False, 'rtol': CASES[case], 'atol': 0., 'checks': [],
              'scope': 'Original model adapted to Python; not original C++ test or optimizer convergence',
              'adaptations': ['absolute paths', 'one thread', 'SimplicialLDLT', 'separate output',
                              'case-specific geometric boundary vertex selection',
                              'PyTorch RNG seed 42, same [0,1] range and three directions as upstream',
                              'additional FD step and scalar mapping/squared loss check'],
              'commits': {n:subprocess.check_output(['git','-C',str(ROOT/n),'rev-parse','HEAD'],text=True).strip()
                          for n in ('polyfem','differentiability-data')}}
    try:
        if variant != 'historical' and case != 'shape-pressure-neumann-nodes':
            raise ValueError('Diagnostic variants apply only to shape-pressure-neumann-nodes')
        report['variant'] = variant
        obj, vertices, ids, config, objective = load_case(case, output/'forward', variant)
        x = vertices[ids].clone().requires_grad_()

        def loss(x):
            return obj(V=vertices.index_copy(0, ids, x))

        report.update(vertex_count=len(vertices), selected_vertex_ids=ids.tolist(),
                      selected_coordinates=x.tolist(), objective=objective.as_dict(),
                      boundary_conditions=config['boundary_conditions'], time=config['time'],
                      forward_solver=config['solver'])
        value = loss(x)
        gradient, = torch.autograd.grad(value, x)
        report.update(value=value.item(), gradient=gradient.tolist())
        generator = torch.Generator().manual_seed(42)
        legacy = case == 'shape-pressure-neumann-nodes'
        if legacy:
            report['adaptations'][5] = 'one seeded direction in [0,1] as in historical test; RNG differs'
            report['adaptations'] += ['runtime boundary 2 and namesake state from historical test 9e4861d542',
                                      'rest_pressure_boundary renamed upstream in 628e816a6e']
            if variant == 'opt-state':
                report['adaptations'][-2:] = ['runtime boundary 2; current opt JSON state and its original Neumann loads']
            if variant == 'zero-load':
                report['adaptations'].append('diagnostic only: set the normal-aligned load to zero')
        for trial in range(1 if legacy else 3):
            direction = torch.rand(x.shape, dtype=x.dtype, generator=generator)
            actual = (gradient*direction).sum().item()
            for h in ((1e-8, 3e-8) if legacy else (1e-7, 3e-7)):
                with torch.no_grad():
                    fd = ((loss(x+h*direction)-loss(x-h*direction))/(2*h)).item()
                error = abs(actual-fd)/max(abs(actual),abs(fd),1e-30)
                check = {'trial':trial, 'h':h, 'direction':direction.tolist(), 'autograd':actual,
                         'finite_difference':fd, 'relative_error':error, 'passed':error <= report['rtol']}
                report['checks'].append(check)
                print(json.dumps({k:v for k,v in check.items() if k != 'direction'}),flush=True)
                (output/'progress.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
        scalar = torch.zeros((), dtype=x.dtype, requires_grad=True)
        outer, = torch.autograd.grad(loss(x.detach()+scalar*direction).square(), scalar)
        expected = (2*value.detach()*(gradient*direction).sum()).item()
        report['mapping'] = {'autograd':outer.item(), 'expected':expected,
                             'passed':abs(outer.item()-expected) <= 1e-12+1e-9*abs(expected)}
        report['passed'] = all(c['passed'] for c in report['checks']) and report['mapping']['passed']
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['elapsed_seconds'] = time.perf_counter()-start
        (output/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    if not report['passed']:
        raise AssertionError('Shape load validation failed; see result.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=CASES, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variant', choices=['historical','opt-state','zero-load'], default='historical')
    args = parser.parse_args()
    validate(args.case, args.output, args.variant)
