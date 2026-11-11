"""Check the original 2D/3D elastic material smoothing cases through diff.material."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import torch
import polyfem.diff as diff
from tools.validation._boundary_config import ROOT, configuration


def validate(dimension, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    case = f'elastic-material-smoothing-{dimension}d'
    spec = json.loads((ROOT/'differentiability-data/input'/f'{case}-opt.json').read_text())
    report = {'case': case, 'passed': False, 'rtol': 2e-7, 'atol': 0., 'checks': [],
              'scope': 'Original model adapted to public Python API; not the original C++ test executable',
              'adaptations': ['absolute paths', 'one thread', 'SimplicialLDLT', 'separate output',
                              'PyTorch random direction in upstream range, not identical C++ RNG',
                              'additional coordinate direction and second finite-difference step'],
              'commits': {n: subprocess.check_output(['git', '-C', str(ROOT/n), 'rev-parse', 'HEAD'], text=True).strip()
                          for n in ('polyfem', 'differentiability-data')}}
    try:
        obj = diff.material(configuration(case, output/'forward'), body=1,
                            objective=diff.objectives.create('elastic_material_smoothing'))
        x = torch.tensor(spec['parameters'][0]['initial'], dtype=torch.float64, requires_grad=True)

        def loss(E):
            return obj(E=E, nu=.3)

        value = loss(x)
        gradient, = torch.autograd.grad(value, x)
        report.update(initial=x.tolist(), value=value.item(), gradient=gradient.tolist())
        low, high = (-.4, .6) if dimension == 2 else (-.3, .2)
        random = low+(high-low)*torch.rand(x.shape, dtype=x.dtype, generator=torch.Generator().manual_seed(42))
        coordinate = torch.zeros_like(x)
        coordinate[gradient.abs().argmax()] = 1.
        for name, direction in [('random', random), ('coordinate', coordinate)]:
            actual = (gradient*direction).sum().item()
            for h in (1e-6, 3e-6):
                with torch.no_grad():
                    fd = ((loss(x+h*direction)-loss(x-h*direction))/(2*h)).item()
                error = abs(actual-fd)/max(abs(actual), abs(fd), 1e-30)
                report['checks'].append({'direction': direction.tolist(), 'name': name, 'h': h,
                                         'autograd': actual, 'finite_difference': fd,
                                         'relative_error': error, 'passed': error <= report['rtol']})
        scalar = torch.zeros((), dtype=x.dtype, requires_grad=True)
        outer, = torch.autograd.grad(loss(x.detach()+scalar*random).square(), scalar)
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
        print(json.dumps(report, indent=2), flush=True)
    if not report['passed']:
        raise AssertionError('Material smoothing validation failed; see result.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dimension', type=int, choices=(2, 3), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    validate(args.dimension, args.output)
