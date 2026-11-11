"""Prepare two original shape-load examples; selectors are specific to these meshes."""
import json

import torch
import polyfem.diff as diff
from polyfempy import polyfempy as backend
from tools.validation._boundary_config import ROOT, configuration


def load_case(case, output=None, variant='historical'):
    config = configuration(case, output)
    spec = json.loads((ROOT/'differentiability-data/input'/f'{case}-opt.json').read_text())
    legacy = case == 'shape-pressure-neumann-nodes'
    if legacy:
        # Historical C++ test 9e4861d542 chose boundary 2 and loaded the namesake state.
        if variant == 'opt-state':
            config = configuration(spec['states'][0]['path'].removesuffix('.json'), output)
        else:
            bc = config['boundary_conditions']
            bc['normal_aligned_neumann_boundary'] = bc.pop('rest_pressure_boundary')
            if variant == 'zero-load':
                bc['normal_aligned_neumann_boundary'][0]['value'] = 0.
    else:
        assert spec['variable_to_simulation'][0]['active_geometry_nodes'] == {'type': 'boundary', 'selection': [2]}
    session = backend.DifferentiableSession()
    session.set_settings(config)
    vertices = torch.tensor(session.mesh_vertices(), dtype=torch.float64)
    del session
    assert vertices.shape[1] == 2
    if legacy:
        ids = torch.tensor(diff.model(config).boundary_vertex_ids([2]), dtype=torch.long)
        assert ids.numel() > 0
    elif case == 'shape-neumann-nodes':
        # Original square boundary: top plus the uppermost edge on each side.
        top = torch.isclose(vertices[:, 1], torch.tensor(1.06, dtype=vertices.dtype))
        sides = torch.isclose(vertices[:, 0].abs(), torch.tensor(.5, dtype=vertices.dtype))
        selected = top | (sides & (vertices[:, 1] >= .81-1e-10))
        assert int(selected.sum()) == 7
    else:
        # The original relative y=0.99 selection contains the refined top edge.
        selected = torch.isclose(vertices[:, 1], vertices[:, 1].max())
        assert int(selected.sum()) == 3
    if not legacy:
        ids = selected.nonzero().flatten()
    functional = spec['functionals'][0]
    objective = diff.objectives.create(functional['type'], **{k:v for k,v in functional.items() if k != 'type'})
    obj = diff.shape(config, objective=objective)
    return obj, vertices, ids, config, objective

