from types import SimpleNamespace

import pytest
import torch

from polyfempy import differentiable_api as diff
from tools.validation._boundary_config import configuration


def test_boundary_query_validates_inputs_before_loading_backend():
    model = diff.model({})
    for ids in ([True], [1.2], ['2']):
        with pytest.raises(TypeError, match='integers'):
            model.boundary_vertex_ids(ids)
    with pytest.raises(TypeError, match='bool'):
        model.boundary_vertex_ids([1], exclude=1)
    with pytest.raises(RuntimeError, match='rebuilding'):
        model.boundary_vertex_ids([1], backend=SimpleNamespace(DifferentiableSession=lambda: object()))


def test_boundary_query_matches_verified_neumann_selector():
    from polyfempy import polyfempy as native
    config = configuration('shape-neumann-nodes')
    session = native.DifferentiableSession()
    session.set_settings(config)
    vertices = torch.tensor(session.mesh_vertices(), dtype=torch.float64)
    expected = (torch.isclose(vertices[:, 1], vertices.new_tensor(1.06)) |
                (torch.isclose(vertices[:, 0].abs(), vertices.new_tensor(.5)) &
                 (vertices[:, 1] >= .81-1e-10))).nonzero().flatten().tolist()
    model = diff.model(config)
    actual = model.boundary_vertex_ids([2])
    assert list(actual) == expected and len(actual) == 7
    all_boundary = set(session.boundary_vertex_ids([], False).tolist())
    remaining = set(session.boundary_vertex_ids([2], True).tolist())
    assert remaining == all_boundary-set(actual)
    assert session.boundary_vertex_ids([987654], False).size == 0


def test_center_target_requires_fixed_reference():
    from test_differentiable_api_initial_object import _InitialSession
    objective = diff.objectives.create('center-target', target_state=1).final()
    with pytest.raises(ValueError, match='requires target'):
        diff.initial({}, objective=objective)
    obj = diff.initial({}, target={}, objective=objective,
                       backend=SimpleNamespace(DifferentiableSession=_InitialSession))
    assert obj.objective['static_objective']['type'] == 'center-target'
