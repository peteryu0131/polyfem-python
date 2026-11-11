"""Independent checks of the two small-step finite difference discrepancies."""
import argparse
from collections import Counter
import json
from pathlib import Path

import torch
import polyfem.diff as diff

from tools.validation._boundary_config import ROOT, configuration
from tools.validation._scalar_walker_config import load_case
from tools.validation.compare_original_json import run_native, write_json


def material(output):
    spec = json.loads((ROOT/'differentiability-data/input/material-transient-opt.json').read_text())
    configs = []
    for i, state in enumerate(spec['states']):
        cfg = configuration(Path(state['path']).stem, output/f'state{i}')
        configs.append(cfg)
        state_path = output / f'state{i}.json'
        write_json(state_path, cfg)
        state['path'] = str(state_path)
    spec['solver']['max_threads'] = 1
    spec['output'] = {'directory':str(output)}
    write_json(output/'opt.json', spec)
    run_native('polyfem_reference_probe.exe', [output/'opt.json',output/'reference.json','--no-strict-validation'], output/'probe.log')
    ref = json.loads((output/'reference.json').read_text())
    f = spec['functionals'][0]
    for key in ('matching','volume_selection'):
        f['static_objective'].pop(key)
    objective = diff.objectives.create(f['type'], **{k:v for k,v in f.items() if k != 'type'})
    obj = diff.material(configs[0],target=configs[1],objective=objective)
    x = torch.tensor(ref['x'],dtype=torch.float64,requires_grad=True)
    assert x.shape == (115,) and torch.allclose(x,torch.full_like(x,configs[0]['materials']['E']))
    value = obj(materials={body:{'E':x[list(ids)],'nu':.3} for body,ids in obj.element_indices_by_body.items()})
    grad, = torch.autograd.grad(value,x)
    return compare(value,grad,ref['value'],torch.tensor(ref['gradient'],dtype=x.dtype))


def laplacian(output):
    objects, V, _, config, _, _, _ = load_case('laplacian',output/'state')
    # Empty surface_selection: the native form iterates n_boundary_elements(),
    # which is n_edges() in 2D, without an exterior-edge filter.
    lines = Path(config['geometry'][0]['mesh']).read_text().splitlines()
    rows = lines[lines.index('$Nodes')+2:lines.index('$EndNodes')]
    ids = {}
    for row in rows:
        node, px, py, _ = row.split()
        coord = torch.tensor([float(px),float(py)],dtype=V.dtype)
        matches = ((V-coord).abs().amax(1)<1e-12).nonzero().flatten()
        assert len(matches)==1
        ids[int(node)] = matches.item()
    counts = Counter()
    for row in lines[lines.index('$Elements')+2:lines.index('$EndElements')]:
        fields = list(map(int,row.split()))
        if fields[1] == 2:
            tri = [ids[node] for node in fields[3+fields[2]:]]
            counts.update(tuple(sorted((tri[k],tri[(k+1)%3]))) for k in range(3))
    edges = torch.tensor(list(counts))
    start,end = torch.cat((edges,edges.flip(1))).T
    x = V.clone().requires_grad_()
    offsets = x[start]-x[end]
    sums = torch.zeros_like(x).index_add(0,start,offsets)
    lengths = x.new_zeros(len(x)).index_add(0,start,offsets.norm(dim=1))
    active = start.unique()
    value_ref = ((sums[active]/lengths[active,None]).norm(dim=1)**3).sum()
    grad_ref, = torch.autograd.grad(value_ref,x)
    value = objects[1](V=x)
    grad, = torch.autograd.grad(value,x)
    return compare(value,grad,value_ref.item(),grad_ref)


def pressure_shape(output, api_report):
    """Compare the disputed boundary direction with upstream OptState on the same state."""
    evidence = json.loads(api_report.read_text())
    assert evidence['case'] == 'shape-pressure-neumann-nodes'
    assert evidence['boundary_conditions']['normal_aligned_neumann_boundary'][0]['value'] == '-200.0 * t'
    config = configuration(evidence['case'], output/'state')
    bc = config['boundary_conditions']
    bc['normal_aligned_neumann_boundary'] = bc.pop('rest_pressure_boundary')
    spec = json.loads((ROOT/'differentiability-data/input/shape-pressure-neumann-nodes-opt.json').read_text())
    # Get the full gradient; project onto the historical boundary direction afterwards.
    spec['parameters'] = 'auto'
    write_json(output/'state.json',config)
    spec['states'][0]['path'] = str(output/'state.json')
    spec['solver']['max_threads'] = 1
    spec['output'] = {'directory':str(output)}
    write_json(output/'opt.json',spec)
    run_native('polyfem_reference_probe.exe',[output/'opt.json',output/'reference.json','--no-strict-validation'],output/'probe.log')
    ref = json.loads((output/'reference.json').read_text())
    check = evidence['checks'][0]
    ids = evidence['selected_vertex_ids']
    v = torch.tensor(ref['x'],dtype=torch.float64).reshape(-1,2)
    torch.testing.assert_close(v[ids],torch.tensor(evidence['selected_coordinates'],dtype=v.dtype))
    g = v.new_tensor(ref['gradient']).reshape_as(v)
    directional = (g[ids]*v.new_tensor(check['direction'])).sum().item()
    return {'passed':abs(directional-check['autograd'])<=1e-8*abs(directional),
            'scope':'agreement with OptState only; original finite-difference discrepancy remains',
            'reference_directional':directional,'api_directional':check['autograd'],'fd':check['finite_difference']}


def compare(value,grad,ref_value,ref_grad):
    torch.testing.assert_close(value.detach(),value.new_tensor(ref_value),rtol=1e-9,atol=1e-14)
    rel = ((grad-ref_grad).norm()/ref_grad.norm()).item()
    assert rel < 1e-8, rel
    return {'passed':True,'value':value.item(),'reference_value':ref_value,'gradient_entries':grad.numel(),
            'gradient_relative_l2_error':rel,'gradient_max_abs_error':(grad-ref_grad).abs().max().item()}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('case',choices=['material-transient','laplacian','shape-pressure-neumann-nodes'])
    p.add_argument('--api-report',type=Path,help='Historical shape run result for OptState comparison')
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    try:
        if args.case=='shape-pressure-neumann-nodes':
            result = pressure_shape(output,args.api_report)
        else:
            result = (material if args.case=='material-transient' else laplacian)(output)
    except Exception as exc:
        result = {'passed':False,'error':repr(exc)}
    result['case'] = args.case
    write_json(output/'result.json',result)
    print(result,flush=True)
    raise SystemExit(0 if result['passed'] else 1)
