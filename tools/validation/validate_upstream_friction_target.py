"""Friction target gradient checks using the upstream P2 contact scene."""
import argparse
import json
from pathlib import Path
import struct
import subprocess
import time

import torch
import polyfem.diff as diff

ROOT = Path(__file__).resolve().parents[2]


def validate(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    source = ROOT / 'differentiability-data/input/friction-contact-opt.json'
    spec = json.loads(source.read_text())
    configs = []
    for index, entry in enumerate(spec['states']):
        path = (source.parent / entry['path']).resolve()
        config = json.loads(path.read_text())
        for geometry in config['geometry']:
            geometry['mesh'] = str((path.parent / geometry['mesh']).resolve())
        config['solver']['max_threads'] = 1
        config['solver']['linear']['solver'] = 'Eigen::SimplicialLDLT'
        config['output'] = {
            'directory': str(output / f'state{index}'), 'json': '',
            'log': {'level': 'warning'}, 'paraview': {'file_name': ''},
            'advanced': {'save_time_sequence': False},
        }
        configs.append(config)
    functional = spec['functionals'][0]
    assert functional['static_objective'].pop('matching') == 'exact'
    objective = diff.objectives.create(functional['type'], **{k: v for k, v in functional.items() if k != 'type'})
    obj = diff.friction(configs[0], target=configs[1], objective=objective)
    # Upstream uses the C++ float literal 0.2f, promoted to double.
    initial = struct.unpack('f', struct.pack('f', .2))[0]
    x = torch.tensor(initial, dtype=torch.float64, requires_grad=True)

    def loss(x):
        return obj(coefficient=x)

    report = {
        'source': str(source.relative_to(ROOT)), 'upstream_test': 'friction-contact',
        'scope': 'Python adaptation; original C++ test executable not run',
        'adaptations': ['absolute mesh paths', 'one thread', 'SimplicialLDLT', 'separate output',
                        'omit obsolete matching=exact (not read by TargetForm)', 'additional FD step 3e-5'],
        'commits': {name: subprocess.check_output(['git', '-C', str(ROOT/name), 'rev-parse', 'HEAD'], text=True).strip()
                    for name in ('polyfem', 'differentiability-data')},
        'objective': objective.as_dict(), 'x': initial, 'rtol': 3.9e-7, 'atol': 0.,
        'checks': [], 'passed': False,
        'reference_note': 'Reference initial velocity differs; matching its friction does not imply zero loss.',
    }
    try:
        value = loss(x)
        gradient, = torch.autograd.grad(value, x)
        report.update(value=value.item(), gradient=gradient.item())
        print(json.dumps({'value': value.item(), 'gradient': gradient.item()}), flush=True)
        for h in (1e-4, 3e-5):
            with torch.no_grad():
                fd = ((loss(x+h)-loss(x-h))/(2*h)).item()
            actual = gradient.item()
            error = abs(actual-fd)
            check = {'h': h, 'autograd': actual, 'finite_difference': fd,
                     'relative_error': error/max(abs(fd), 1e-30),
                     'passed': error <= report['rtol']*max(abs(actual), abs(fd))}
            report['checks'].append(check)
            print(json.dumps(check), flush=True)
        scale = torch.ones((), dtype=x.dtype, requires_grad=True)
        outer, = torch.autograd.grad(loss(scale*x.detach()).square(), scale)
        expected = (2*value.detach()*gradient*x.detach()).item()
        report['mapping'] = {'autograd': outer.item(), 'expected': expected,
                             'passed': abs(outer.item()-expected) <= 1e-12+1e-8*abs(expected)}
        report['passed'] = all(c['passed'] for c in report['checks']) and report['mapping']['passed']
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['elapsed_seconds'] = time.perf_counter()-start
        (output/'result.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    if not report['passed']:
        raise AssertionError('Friction target checks failed; see result.json')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    validate(parser.parse_args().output)
