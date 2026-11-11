from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest
import torch
from polyfempy import differentiable_api as diff
from test_differentiable_api_controls import _Session


def config():
    return {'contact': {'enabled': True}, 'solver': {'contact': {'barrier_stiffness': 100.}}}


def test_friction_target_validation_copy_and_old_backend():
    objective = diff.objectives.target(target_state=1).final()
    target = {'reference': [.5]}
    backend = SimpleNamespace(DifferentiableSession=_Session)
    obj = diff.friction(config(), target=target, objective=objective, backend=backend)
    target['reference'][0] = 3.
    assert obj._target['reference'][0] == .5
    with pytest.raises(RuntimeError, match='Rebuild.*set_target_settings'):
        obj(coefficient=torch.tensor(.3))
    with pytest.raises(ValueError, match='requires target'):
        diff.friction(config(), objective=objective)
    with pytest.raises(ValueError, match='target requires'):
        diff.friction(config(), target={}, objective=diff.Objective.COMPLIANCE)
    invalid = config()
    invalid['solver']['contact']['barrier_stiffness'] = 'adaptive'
    with pytest.raises(ValueError, match='fixed barrier_stiffness'):
        diff.friction(invalid, target={}, objective=objective, backend=backend)
    for value in (-.1, float('nan')):
        with pytest.raises(ValueError):
            obj(coefficient=torch.tensor(value))


def test_friction_target_pending_graphs_and_outer_loss():
    class Session(_Session):
        def set_target_settings(self, target):
            self.reference = target['reference']

        def solve_control_objective(self):
            return ((self.values-self.reference)**2).sum()

        def backward_control(self, gradient):
            return 2*(self.values-self.reference)*gradient

    obj = diff.friction(config(), target={'reference': .5},
                        objective=diff.objectives.target(target_state=1).final(),
                        backend=SimpleNamespace(DifferentiableSession=Session))
    x = torch.tensor(.3, dtype=torch.float64, requires_grad=True)
    a, b = obj(coefficient=x), obj(coefficient=2*x)
    (a.square()+b).backward()
    expected = 4*(x.detach()-.5)**3 + 4*(2*x.detach()-.5)
    torch.testing.assert_close(x.grad, expected)


def test_real_friction_target_gradient_and_update(tmp_path, monkeypatch):
    samples = Path(__file__).resolve().parents[1] / 'differentiable_example/samples'
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(samples))
    sample = runpy.run_path(str(samples / 'material_and_contact/friction_matching.py'))
    loss = sample['loss']
    x = torch.tensor(.3, dtype=torch.float64, requires_grad=True)
    value = loss(x)
    gradient, = torch.autograd.grad(value, x)
    assert gradient.abs() > 1e-12
    with torch.no_grad():
        for h in (1e-4, 3e-5):
            fd = (loss(x+h)-loss(x-h))/(2*h)
            torch.testing.assert_close(gradient, fd, rtol=1e-3, atol=1e-10)
        assert loss(x.new_tensor(.5)).item() == pytest.approx(0., abs=1e-14)
        assert loss(sample['x']) < value.detach()
    outer, = torch.autograd.grad(loss(x).square(), x)
    torch.testing.assert_close(outer, 2*value.detach()*gradient)
    invalid = diff.model(sample['config']).as_dicts()[0]
    invalid['geometry'][0]['mesh'] = str(samples.parents[1] / 'differentiability-data/square.obj')
    obj = diff.friction(invalid, target=invalid, objective=diff.objectives.target(target_state=1, selection=1).final())
    def square_mesh_loss(v):
        return obj(coefficient=v)
    value = square_mesh_loss(x)
    derivative, = torch.autograd.grad(value, x)
    h = 1e-4
    direction = torch.ones_like(x)
    with torch.no_grad():
        fd = (square_mesh_loss(x+h*direction) - square_mesh_loss(x-h*direction)) / (2*h)
    torch.testing.assert_close(derivative.sum(), fd, rtol=1e-3, atol=1e-10)
