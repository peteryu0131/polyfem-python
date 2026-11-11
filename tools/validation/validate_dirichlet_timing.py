"""Diagnose the original shifted Dirichlet time grid; never bypass it in public API."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import polyfem.diff as diff
from polyfempy import polyfempy as backend
from tools.validation._boundary_config import ROOT, configuration


def validate(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    config = configuration('dirichlet-sdf', output/'state')
    original = json.loads((ROOT/'differentiability-data/input/dirichlet-sdf-opt.json').read_text())
    functional = original['functionals'][0]
    objective = diff.objectives.create(functional['type'], **{k: v for k, v in functional.items() if k != 'type'})
    boundaries = config['boundary_conditions']['dirichlet_boundary']
    x = np.stack([np.asarray(b['value'], dtype=float)[:, 1:].T for b in boundaries], axis=1)
    report = {
        'case': 'dirichlet-sdf', 'status': 'diagnosing', 'passed': False,
        'scope': 'Original time grid, diagnostic direct bridge calls only; not a supported public API path',
        'has_test_diff_reference': 'dirichlet-sdf-opt.json' in (ROOT/'polyfem/tests/test_diff.cpp').read_text(),
        'commits': {n: subprocess.check_output(['git', '-C', str(ROOT/n), 'rev-parse', 'HEAD'], text=True).strip()
                    for n in ('polyfem', 'differentiability-data')},
        'time': config['time'], 'boundary_time_reference': {b['id']: b['time_reference'] for b in boundaries},
        'api_shape': list(x.shape), 'objective': objective.as_dict(), 'checks': [],
        'adaptations': ['absolute mesh/resource paths', 'one thread', 'SimplicialLDLT', 'separate output'],
    }

    def session(values):
        native = backend.DifferentiableSession()
        native.set_settings(config)
        count = native.prepare_control({'type': 'dirichlet-boundary', 'active_boundary_ids': [1, 2, 3], 'active_time_slices': []})
        assert count == values.size
        native.set_control_parameters(values.reshape(-1).copy())
        native.set_objective(objective.as_dict())
        return native

    try:
        try:
            diff.dirichlet(config, boundary_ids=[1, 2, 3], objective=objective)
        except ValueError as exc:
            report['public_api_error'] = str(exc)
        else:
            raise AssertionError('Expected shifted time grid to be rejected')
        native = session(x)
        value = native.solve_control_objective()
        gradient = np.asarray(native.backward_control(1.)).reshape(x.shape)
        del native
        report.update(value=value, gradient=gradient.tolist())
        # Last array slot is timestamp .22, beyond the simulation end .20.
        index = np.unravel_index(np.argmax(np.abs(gradient[-1])), gradient[-1].shape)
        index = (x.shape[0]-1, *index)
        direction = np.zeros_like(x)
        direction[index] = 1.
        actual = float(gradient[index])
        print(json.dumps({'index': index, 'gradient': actual}, default=int), flush=True)
        for h in (1e-4, 3e-5):
            plus = session(x+h*direction).solve_control_objective()
            minus = session(x-h*direction).solve_control_objective()
            fd = (plus-minus)/(2*h)
            check = {'index': [int(i) for i in index], 'h': h, 'autograd': actual,
                     'finite_difference': fd, 'absolute_error': abs(actual-fd)}
            report['checks'].append(check)
            print(json.dumps(check), flush=True)
        report['status'] = 'blocked_by_shifted_time_reference'
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['elapsed_seconds'] = time.perf_counter()-start
        (output/'result.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    validate(parser.parse_args().output)
