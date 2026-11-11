"""Prepare original Laplacian and walker shape cases without modifying upstream files."""
from collections import Counter
import json

import torch
import polyfem.diff as diff
from polyfempy import polyfempy as backend
from tools.validation._boundary_config import ROOT, configuration


def load_case(case, output=None):
    spec = json.loads((ROOT/'differentiability-data/input'/f'{case}-opt.json').read_text())
    state_name = spec['states'][0]['path'].removesuffix('.json')
    config = configuration(state_name, output)
    session = backend.DifferentiableSession()
    session.set_settings(config)
    vertices = torch.tensor(session.mesh_vertices(), dtype=torch.float64)
    metadata = session.model_metadata()
    del session
    groups = []
    if case == 'laplacian':
        ids = torch.arange(len(vertices))
    else:
        geometry = config['geometry'][0]
        assert geometry['transformation'] == {'scale': .1} and geometry['n_refs'] == 0
        points, edges = [], Counter()
        for line in open(geometry['mesh'], encoding='utf-8'):
            parts = line.split()
            if parts and parts[0] == 'v':
                points.append([float(v)*.1 for v in parts[1:3]])
            elif parts and parts[0] == 'f':
                face = [int(v.split('/')[0])-1 for v in parts[1:]]
                assert len(face) == 3 and min(face) >= 0
                edges.update(tuple(sorted((face[i],face[(i+1)%3]))) for i in range(3))
        points = torch.tensor(points, dtype=vertices.dtype)
        boundary = [edge for edge,count in edges.items() if count == 1]
        for boundary_id in (2,3):
            selection = next(s for s in geometry['surface_selection'] if s['id']==boundary_id)
            lower,upper = torch.tensor(selection['box'],dtype=vertices.dtype)
            nodes=set()
            for edge in boundary:
                center=points[list(edge)].mean(0)
                if bool(((center>=lower)&(center<=upper)).all()):
                    nodes.update(edge)
            mapped=[]
            for node in sorted(nodes):
                matches=torch.isclose(vertices,points[node],rtol=0,atol=1e-10).all(1).nonzero().flatten()
                assert len(matches)==1
                mapped.append(matches.item())
            assert mapped
            groups.append(sorted(mapped))
        assert not set(groups[0]) & set(groups[1])
        ids=torch.tensor(groups[0]+groups[1])
    objects,objectives=[],[]
    for f in spec['functionals']:
        objective=diff.objectives.create(f['type'],**{k:v for k,v in f.items() if k!='type'})
        objects.append(diff.shape(config,objective=objective))
        objectives.append(objective.as_dict())
    return objects,vertices,ids,config,objectives,groups,metadata
