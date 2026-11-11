"""Check damping target gradients using the upstream 40-step contact scene."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import torch
import polyfem.diff as diff

ROOT = Path(__file__).resolve().parents[2]


def validate(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    source = ROOT / 'differentiability-data/input/damping-transient-opt.json'
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
    # TargetForm always integrates over surface_selection, not volume_selection.
    assert functional['static_objective'].pop('volume_selection') == [1]
    objective = diff.objectives.create(functional['type'], **{k: v for k, v in functional.items() if k != 'type'})
    obj = diff.damping(configs[0], target=configs[1], objective=objective)
    x = torch.tensor(spec['parameters'][0]['initial'], dtype=torch.float64, requires_grad=True)

    def loss(x):
        return obj(psi=x[0], phi=x[1])

    report = {
        'source': str(source.relative_to(ROOT)), 'upstream_test': 'damping-transient',
        'scope': 'Python adaptation of upstream test; original C++ test executable not run',
        'adaptations': ['absolute mesh paths', 'one thread', 'SimplicialLDLT', 'separate output',
                        'omit obsolete matching=exact (not read by TargetForm)',
                        'omit unused volume_selection; TargetForm uses surface_selection',
                        'test upstream [1,1] direction once, plus both coordinate directions, at two FD steps'],
        'commits': {name: subprocess.check_output(['git', '-C', str(ROOT/name), 'rev-parse', 'HEAD'], text=True).strip()
                    for name in ('polyfem', 'differentiability-data')},
        'objective': objective.as_dict(), 'rtol': 2.5e-6, 'atol': 0.,
        'checks': [], 'passed': False,
    }
    try:
        value = loss(x)
        gradient, = torch.autograd.grad(value, x)
        report.update(value=value.item(), gradient=gradient.tolist())
        print(json.dumps({'value': value.item(), 'gradient': gradient.tolist()}), flush=True)
        for direction in ([1., 1.], [1., 0.], [0., 1.]):
            d = x.new_tensor(direction)
            actual = (gradient*d).sum().item()
            for h in (1e-3, 3e-3):
                with torch.no_grad():
                    fd = ((loss(x+h*d)-loss(x-h*d))/(2*h)).item()
                error = abs(actual-fd)
                check = {'direction': direction, 'h': h, 'autograd': actual, 'finite_difference': fd,
                         'relative_error': error/max(abs(fd), 1e-30),
                         'passed': error <= report['rtol']*max(abs(actual), abs(fd))}
                report['checks'].append(check)
                print(json.dumps(check), flush=True)
        with torch.no_grad():
            reference = configs[1]['materials']
            report['reference_value'] = loss(x.new_tensor([reference['psi'], reference['phi']])).item()
        scale = torch.ones((), dtype=x.dtype, requires_grad=True)
        outer, = torch.autograd.grad(loss(scale*x.detach()).square(), scale)
        expected = (2*value.detach()*(gradient*x.detach()).sum()).item()
        report['mapping'] = {'autograd': outer.item(), 'expected': expected,
                             'passed': abs(outer.item()-expected) <= 1e-10+1e-8*abs(expected)}
        report['passed'] = (all(c['passed'] for c in report['checks'])
                            and report['reference_value'] < 1e-12 and report['mapping']['passed'])
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['elapsed_seconds'] = time.perf_counter()-start
        (output/'result.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    if not report['passed']:
        raise AssertionError('Damping target checks failed; see result.json')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    validate(parser.parse_args().output)
