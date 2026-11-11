"""Prepare the original 3D position and plate-target validation cases."""
import json

import torch
import polyfem.diff as diff
from polyfempy import polyfempy as backend
from tools.validation._boundary_config import ROOT, configuration


def load_case(case, output=None):
    spec_path = ROOT/'differentiability-data/input'/f'{case}-opt.json'
    spec = json.loads(spec_path.read_text())
    config = configuration(case, output)
    session = backend.DifferentiableSession()
    session.set_settings(config)
    vertices = torch.tensor(session.mesh_vertices(), dtype=torch.float64)
    del session
    assert vertices.shape[1] == 3
    if case == 'linear_elasticity-surface-3d':
        ids = torch.arange(len(vertices))
    elif case == '3d-shape-mesh-target':
        assert spec['variable_to_simulation'][0]['active_geometry_nodes'] == {
            'type': 'boundary_excluding_surface', 'selection': [1, 2, 3, 4]}
        lower, upper = vertices.amin(0), vertices.amax(0)
        on_plane = torch.isclose(vertices, lower) | torch.isclose(vertices, upper)
        # This original plate has two vertex layers, both on the boundary.
        assert bool(on_plane[:, 1].all())
        ids = (on_plane[:, 1] & ~on_plane[:, 0] & ~on_plane[:, 2]).nonzero().flatten()
        assert ids.numel()*3 == spec['parameters'][0]['number'] == 1944
    else:
        raise ValueError(f'Unknown 3D case: {case}')
    objects, objectives = [], []
    for functional in spec['functionals']:
        if 'mesh_path' in functional:
            functional['mesh_path'] = str((spec_path.parent/functional['mesh_path']).resolve())
        objective = diff.objectives.create(functional['type'], **{k:v for k,v in functional.items() if k != 'type'})
        objectives.append(objective.as_dict())
        objects.append(diff.shape(config, objective=objective))
    return objects, vertices, ids, config, objectives
