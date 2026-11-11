from copy import deepcopy
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest
import torch
from polyfempy import differentiable_api as diff
from test_differentiable_api_controls import _Session


class Session(_Session):
    def prepare_control(self, spec):
        self.spec = spec
        return len(spec['active_geometry_nodes'])*2


def test_node_shapes_mapping_and_validation():
    backend = SimpleNamespace(DifferentiableSession=Session)
    obj = diff.dirichlet({}, node_ids=[3, 1], objective=diff.Objective.COMPLIANCE, backend=backend)
    assert obj.input_shape == (2, 2)
    assert obj.spec['active_geometry_nodes'] == [3, 1]
    x = torch.ones(2, 2, dtype=torch.float64, requires_grad=True)
    a, b = obj(values=x), obj(values=x*2)
    (a.square()+b).backward()
    torch.testing.assert_close(x.grad, 24*x.detach())
    for invalid in ({}, {'node_ids': [], 'boundary_ids': []}, {'node_ids': [1, 1]},
                    {'node_ids': [-1]}, {'node_ids': [True]}, {'node_ids': [1], 'time_slices': []}):
        with pytest.raises(ValueError):
            diff.dirichlet({}, objective=diff.Objective.COMPLIANCE, backend=backend, **invalid)
    with pytest.raises(ValueError, match='shape'):
        obj(values=torch.ones(4))


def test_real_node_sample_order_and_rejections(tmp_path, monkeypatch):
    samples = Path(__file__).resolve().parents[1] / 'differentiable_example/samples'
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(samples))
    sample = runpy.run_path(str(samples/'boundary_controls'/'dirichlet_nodes.py'))
    with torch.no_grad():
        assert sample['loss'](sample['x']) < sample['value'].detach()
    x = torch.arange(12, dtype=torch.float64).reshape(4, 3).mul(.0001).requires_grad_()
    a = sample['obj'](values=x)
    ga, = torch.autograd.grad(a, x)
    obj = diff.dirichlet(sample['config'], node_ids=[6, 4, 1, 0],
                         objective=diff.objectives.stress_norm(selection=1))
    b = obj(values=x.flip(0))
    gb, = torch.autograd.grad(b, x)
    torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(ga, gb, rtol=1e-9, atol=1e-9)
    for nodes, message in (([1000], 'range'), ([3], 'not a nodal')):
        with pytest.raises(RuntimeError, match=message):
            diff.dirichlet(sample['config'], node_ids=nodes, objective=diff.Objective.COMPLIANCE)
    transient = deepcopy(sample['config'])
    transient['time'] = {'dt': .01, 'tend': .02}
    with pytest.raises(RuntimeError, match='quasistatic'):
        diff.dirichlet(transient, node_ids=[0], objective=diff.Objective.COMPLIANCE)
