"""Check original nodal-displacement and pressure cases through PyTorch."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import torch
import polyfem.diff as diff
from tools.validation._boundary_config import ROOT, configuration

CASES = {
    'dirichlet-nodes-3d': (3, (1e-7, 3e-7), 1.8e-7),
    'static-control-pressure-nodes-3d': (2, (1e-3, 3e-3), 2.7e-7),
    'control-pressure-nodes-3d': (1, (1e-8, 3e-8), 1.8e-6),
    'walker': (3, (1e-4, 3e-4), 1.8e-5),
}


def validate(case, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    repeats, steps, tolerance = CASES[case]
    source = ROOT / 'differentiability-data/input' / f'{case}-opt.json'
    spec = json.loads(source.read_text())
    config = configuration(case, output/'state')
    functional = spec['functionals'][0]
    objective = diff.objectives.create(functional['type'], **{k: v for k, v in functional.items() if k != 'type'})
    nodal = case == 'dirichlet-nodes-3d'
    if nodal:
        nodes = spec['variable_to_simulation'][0]['active_geometry_nodes']
        obj = diff.dirichlet(config, node_ids=nodes, objective=objective)
        # Upstream inverse_eval returns zero, not the constraint file's values.
        x = torch.zeros(obj.input_shape, dtype=torch.float64, requires_grad=True)
    else:
        boundaries = config['boundary_conditions']['pressure_boundary']
        obj = diff.pressure(config, boundary_ids=[b['id'] for b in boundaries], objective=objective)
        if config.get('time') is not None:
            # Original optimization groups parameters by boundary. Keep that order
            # in x; the public API takes (time, boundary). Step 0 stays fixed.
            x = torch.tensor([b['value'][1:] for b in boundaries], dtype=torch.float64).flatten().requires_grad_()
        else:
            x = torch.tensor([b['value'] for b in boundaries], dtype=torch.float64, requires_grad=True)

    def loss(x):
        values = x.reshape(len(boundaries), -1).T if not nodal and config.get('time') is not None else x
        return obj(values=values)

    report = {
        'source': str(source.relative_to(ROOT)), 'case': case,
        'scope': 'Python adaptation of upstream test; original C++ executable not run',
        'adaptations': ['absolute mesh paths and resource root', 'one thread', 'SimplicialLDLT',
                        'separate output', 'PyTorch seed 42 directions instead of C++ RNG', 'second FD step'],
        'commits': {name: subprocess.check_output(['git', '-C', str(ROOT/name), 'rev-parse', 'HEAD'], text=True).strip()
                    for name in ('polyfem', 'differentiability-data')},
        'objective': objective.as_dict(), 'input_shape': list(x.shape), 'api_input_shape': list(obj.input_shape), 'initial': x.tolist(),
        'layout': 'boundary-major vector mapped to (time, boundary)' if not nodal and config.get('time') is not None else 'direct',
        'rtol': tolerance, 'atol': 0., 'checks': [], 'passed': False,
    }
    try:
        value = loss(x)
        gradient, = torch.autograd.grad(value, x)
        report.update(value=value.item(), gradient=gradient.tolist())
        print(json.dumps({'case': case, 'value': value.item(), 'gradient_norm': gradient.norm().item()}), flush=True)
        generator = torch.Generator().manual_seed(42)
        for trial in range(repeats):
            direction = torch.rand(x.shape, generator=generator, dtype=x.dtype)
            actual = (gradient*direction).sum().item()
            for h in steps:
                with torch.no_grad():
                    fd = ((loss(x+h*direction)-loss(x-h*direction))/(2*h)).item()
                check = {'trial': trial, 'direction': direction.tolist(), 'h': h,
                         'autograd': actual, 'finite_difference': fd,
                         'relative_error': abs(actual-fd)/max(abs(fd), 1e-30),
                         'passed': abs(actual-fd) <= report['rtol']*max(abs(actual), abs(fd))}
                report['checks'].append(check)
                print(json.dumps(check), flush=True)
                (output/'progress.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        parameter = torch.zeros((), dtype=x.dtype, requires_grad=True)
        outer, = torch.autograd.grad(loss(x.detach()+parameter*direction).square(), parameter)
        expected = (2*value.detach()*(gradient*direction).sum()).item()
        report['mapping'] = {'autograd': outer.item(), 'expected': expected,
                             'passed': abs(outer.item()-expected) <= 1e-10+1e-10*abs(expected)}
        report['passed'] = all(c['passed'] for c in report['checks']) and report['mapping']['passed']
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['elapsed_seconds'] = time.perf_counter()-start
        (output/'result.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    if not report['passed']:
        raise AssertionError(f'{case} checks failed; see result.json')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=list(CASES), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    validate(args.case, args.output)
