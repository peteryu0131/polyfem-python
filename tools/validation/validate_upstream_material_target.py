"""Validate per-element E against the original transient material target case."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import torch
import polyfem.diff as diff
from tools.validation._boundary_config import ROOT, configuration


def validate(output, coordinate_steps=None):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    spec = json.loads((ROOT/'differentiability-data/input/material-transient-opt.json').read_text())
    config = configuration('material-transient', output/'primary')
    target = configuration('material-transient-target', output/'target')
    functional = spec['functionals'][0]
    assert functional['static_objective'].pop('matching') == 'exact'
    assert functional['static_objective'].pop('volume_selection') == [1]
    objective = diff.objectives.create(functional['type'], **{k: v for k, v in functional.items() if k != 'type'})
    obj = diff.material(config, target=target, objective=objective)
    assert obj.n_elements == spec['parameters'][0]['number'] == 115
    x = torch.full((obj.n_elements,), config['materials']['E'], dtype=torch.float64, requires_grad=True)

    def loss(x):
        return obj(materials={body: {'E': x[list(indices)], 'nu': .3}
                              for body, indices in obj.element_indices_by_body.items()})

    report = {
        'case': 'material-transient', 'passed': False,
        'scope': 'Python adaptation of original model, not original C++ executable',
        'commits': {n: subprocess.check_output(['git', '-C', str(ROOT/n), 'rev-parse', 'HEAD'], text=True).strip()
                    for n in ('polyfem', 'differentiability-data')},
        'adaptations': ['absolute mesh/resource paths', 'one thread', 'SimplicialLDLT', 'separate output',
                        'omit unused matching and volume_selection from TargetForm',
                        'upstream constant-1000 direction once instead of three identical repeats',
                        'additional random and coordinate directions, second FD step'],
        'parameter': 'E in internal units, nu=.3 fixed', 'initial': x.tolist(),
        'element_indices_by_body': obj.element_indices_by_body,
        'objective': objective.as_dict(), 'rtol': 5.3e-6, 'atol': 0., 'checks': [],
        'diagnostic_coordinate_steps': coordinate_steps,
    }
    try:
        value = loss(x)
        gradient, = torch.autograd.grad(value, x)
        report.update(value=value.item(), gradient=gradient.tolist())
        coordinate = torch.zeros_like(x)
        coordinate[gradient.abs().argmax()] = 1000.
        random = 1000.*torch.rand(x.shape, dtype=x.dtype, generator=torch.Generator().manual_seed(42))
        directions = [('upstream_constant', torch.full_like(x, 1000.)), ('random', random), ('coordinate', coordinate)]
        if coordinate_steps is not None:
            directions = [('coordinate', coordinate)]
        for name, direction in directions:
            actual = (gradient*direction).sum().item()
            for h in coordinate_steps if coordinate_steps is not None else (1e-5, 3e-5):
                with torch.no_grad():
                    fd = ((loss(x+h*direction)-loss(x-h*direction))/(2*h)).item()
                check = {'direction_name': name, 'direction': direction.tolist(), 'h': h,
                         'autograd': actual, 'finite_difference': fd,
                         'relative_error': abs(actual-fd)/max(abs(fd), 1e-30),
                         'passed': abs(actual-fd) <= report['rtol']*max(abs(actual), abs(fd))}
                report['checks'].append(check)
                print(json.dumps({k:v for k,v in check.items() if k != 'direction'}), flush=True)
                (output/'progress.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        parameter = torch.zeros((), dtype=x.dtype, requires_grad=True)
        outer, = torch.autograd.grad(loss(x.detach()+parameter*random).square(), parameter)
        expected = (2*value.detach()*(gradient*random).sum()).item()
        report['mapping'] = {'autograd': outer.item(), 'expected': expected,
                             'passed': abs(outer.item()-expected) <= 1e-12+1e-9*abs(expected)}
        report['passed'] = all(c['passed'] for c in report['checks']) and report['mapping']['passed']
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['elapsed_seconds'] = time.perf_counter()-start
        (output/'result.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    if not report['passed']:
        raise AssertionError('Material target checks failed; see result.json')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--coordinate-steps', type=float, nargs='+', help='Diagnostic sweep only; does not replace the full validation')
    args = parser.parse_args()
    validate(args.output, args.coordinate_steps)
