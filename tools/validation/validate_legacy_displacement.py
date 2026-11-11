"""Check the SDF control on its original timestamps; audit the missing reference assets."""
import argparse
import copy
import json
from pathlib import Path

import torch
import polyfem.diff as diff
from polyfempy import polyfempy as native

from tools.validation._boundary_config import ROOT, configuration


def validate(case, output):
    output.mkdir(parents=True, exist_ok=False)
    config = configuration(case, output / 'primary')
    spec = json.loads((ROOT / 'differentiability-data/input' / f'{case}-opt.json').read_text())
    if case == 'dirichlet-ref':
        missing = [Path(g['mesh']).name for g in config['geometry'] if not Path(g['mesh']).is_file()]
        return {'case':case,'passed':False,'missing_meshes':missing,
                'status':'missing original mesh' if missing else 'original mesh available; validate reference derivative before exposing API',
                'historical_test':'20f3d86012^:tests/test_diff.cpp dirichlet-ref-new',
                'scope':'Reference state constructed at runtime, not a standalone optimizer JSON'}
    raw = copy.deepcopy(config)
    boundaries = config['boundary_conditions']['dirichlet_boundary']
    ids = [b['id'] for b in boundaries]
    dt = config['time']['dt']
    n = round(config['time']['tend'] / dt)
    grid = [round(k * dt, 10) for k in range(n + 1)]
    series = torch.tensor([b['value'] for b in boundaries], dtype=torch.float64).permute(2, 0, 1)
    # These files start at dt. At t=0 the upstream lookup falls back to zero.
    # Preserve every original table entry and also supply the zero initial displacement.
    for b in boundaries:
        assert all(abs(t - (k+1)*dt) < 1e-12 for k, t in enumerate(b['time_reference']))
        assert len(b['time_reference']) == n+1
        b['value'] = [[0., *row] for row in b['value']]
        b['time_reference'] = [0., *b['time_reference']]
    f = spec['functionals'][0]
    objective = diff.objectives.create(f['type'], **{k:v for k,v in f.items() if k != 'type'})
    x = series[1:].clone().requires_grad_()
    obj = diff.dirichlet(config, boundary_ids=ids, objective=objective)
    def loss(p):
        return obj(values=torch.cat((series[:1], p[:-1])))
    value = loss(x)
    grad, = torch.autograd.grad(value, x)
    report = {'case':case, 'value':value.item(), 'checks':[], 'time_grid':grid, 'parameter_shape':list(x.shape)}
    random = 2*torch.rand(x.shape, dtype=x.dtype, generator=torch.Generator().manual_seed(42))-1
    directions = [('random', random)]
    tail = torch.zeros_like(x)
    tail[-1,1,1] = 1.
    directions.append(('unused_last_slot',tail))
    for name, d in directions:
        derivative = (grad*d).sum().item()
        for h in (1e-4, 3e-5):
            with torch.no_grad():
                plus = loss(x+h*d).item()
                minus = loss(x-h*d).item()
            fd = (plus-minus)/(2*h)
            check = {'direction':name,'h':h,'autograd':derivative,'fd':fd,
                     'passed':abs(derivative-fd)<=1e-5*max(abs(derivative),abs(fd))}
            if name in ('random', 'unused_last_slot'):
                # Independent forward with original timestamps and original parameter slots.
                session = native.DifferentiableSession()
                session.set_settings(raw)
                session.prepare_control({'type':'dirichlet-boundary','active_boundary_ids':ids,'active_time_slices':[]})
                session.set_control_parameters((x+h*d).detach().flatten().numpy().copy())
                session.set_objective(objective.as_dict())
                legacy = session.solve_control_objective()
                check.update(raw_forward=legacy,aligned_forward=plus)
                check['passed'] &= abs(legacy-plus) <= 1e-10*max(1.,abs(legacy))
            report['checks'].append(check)
            print(check, flush=True)
    u = torch.zeros((),dtype=x.dtype,requires_grad=True)
    outer, = torch.autograd.grad(loss(x.detach()+u*random).square(), u)
    expected = (2*value.detach()*(grad*random).sum()).item()
    report['chain_rule'] = {'actual':outer.item(),'expected':expected}
    report['passed'] = all(c['passed'] for c in report['checks']) and abs(outer.item()-expected)<=1e-12+1e-9*abs(expected)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', choices=['dirichlet-ref','dirichlet-sdf'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output directory')
    try:
        result = validate(args.case,args.output)
    except Exception as exc:
        result = {'case':args.case,'passed':False,'error':repr(exc)}
    print(result,flush=True)
    (args.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    raise SystemExit(0 if result['passed'] else 1)
