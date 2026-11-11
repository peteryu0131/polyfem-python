from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest
import torch
from polyfempy import differentiable_api as diff
from test_differentiable_api_controls import _Session


def test_damping_target_validation_copy_and_old_backend():
    objective = diff.objectives.target(target_state=1).final()
    target = {'reference': [.4, .6]}
    obj = diff.damping({}, target=target, objective=objective,
                       backend=SimpleNamespace(DifferentiableSession=_Session))
    target['reference'][0] = 3.
    assert obj._target['reference'][0] == .4
    x = torch.ones(2, dtype=torch.float64)
    with pytest.raises(RuntimeError, match='Rebuild.*set_target_settings'):
        obj(psi=x[0], phi=x[1])
    with pytest.raises(ValueError, match='requires target'):
        diff.damping({}, objective=objective)
    for invalid in (diff.Objective.COMPLIANCE, diff.objectives.target(target_state=1),
                    diff.objectives.target(target_state=0).final()):
        with pytest.raises(ValueError, match='target requires'):
            diff.damping({}, target={}, objective=invalid)


def test_damping_target_pending_graphs_and_outer_loss():
    class Session(_Session):
        def set_target_settings(self, target):
            self.reference = target['reference']

        def solve_control_objective(self):
            return ((self.values-self.reference)**2).sum()

        def backward_control(self, gradient):
            return 2*(self.values-self.reference)*gradient

    obj = diff.damping({}, target={'reference': [.4, .6]},
                       objective=diff.objectives.target(target_state=1).final(),
                       backend=SimpleNamespace(DifferentiableSession=Session))
    x = torch.tensor([.1, .2], dtype=torch.float64, requires_grad=True)
    a, b = obj(psi=x[0], phi=x[1]), obj(psi=2*x[0], phi=2*x[1])
    (a.square()+b).backward()
    reference = x.new_tensor([.4, .6])
    expected = 4*((x.detach()-reference)**2).sum()*(x.detach()-reference) + 4*(2*x.detach()-reference)
    torch.testing.assert_close(x.grad, expected)


def test_real_damping_target_gradient_and_update(tmp_path, monkeypatch):
    samples = Path(__file__).resolve().parents[1] / 'differentiable_example/samples'
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(samples))
    sample = runpy.run_path(str(samples / 'material_and_contact/damping_matching.py'))
    loss = sample['loss']
    x = torch.tensor([.1, .2], dtype=torch.float64, requires_grad=True)
    value = loss(x)
    gradient, = torch.autograd.grad(value, x)
    assert (gradient.abs() > 1e-12).all()
    with torch.no_grad():
        for h in (1e-4, 3e-5):
            fd = torch.stack([(loss(x+h*d)-loss(x-h*d))/(2*h) for d in torch.eye(2, dtype=x.dtype)])
            torch.testing.assert_close(gradient, fd, rtol=1e-3, atol=1e-10)
        assert loss(x.new_tensor([.4, .6])).item() == pytest.approx(0., abs=1e-14)
        assert loss(sample['x']) < value.detach()
    outer, = torch.autograd.grad(loss(x).square(), x)
    torch.testing.assert_close(outer, 2*value.detach()*gradient)
    config = diff.model(sample['config']).as_dicts()[0]
    config['geometry'][0]['mesh'] = str(samples.parents[1] / 'differentiability-data/square.obj')
    obj = diff.damping(config, target=config, objective=diff.objectives.target(target_state=1).final())
    def square_mesh_loss(v):
        return obj(psi=v[0], phi=v[1])
    value = square_mesh_loss(x)
    derivative, = torch.autograd.grad(value, x)
    h = 1e-4
    direction = torch.ones_like(x)
    with torch.no_grad():
        fd = (square_mesh_loss(x+h*direction) - square_mesh_loss(x-h*direction)) / (2*h)
    torch.testing.assert_close(derivative.sum(), fd, rtol=1e-3, atol=1e-10)
