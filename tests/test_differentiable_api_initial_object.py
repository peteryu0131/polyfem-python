from pathlib import Path
import runpy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from polyfempy import differentiable_api as diff


class _InitialSession:
    setups = 0
    uniform = True
    seen = []

    def set_settings(self, settings):
        pass

    def model_metadata(self):
        type(self).setups += 1
        return dict(dimension=2, n_vertices=3, n_elements=1, n_bases=3,
                    value_dim=2, n_dofs=6,
                    supports_uniform_initial_condition=self.uniform)

    def set_initial_condition_parameters(self, initial):
        self.initial = initial.copy()
        type(self).seen.append(self.initial)

    def initial_condition_dof_count(self):
        return 6

    def set_objective(self, objective):
        self.objective = objective

    def solve_initial_condition_objective(self):
        return (self.initial**2).sum()

    def backward_initial_condition(self, gradient):
        return 2 * self.initial * float(gradient)


def _object():
    class Session(_InitialSession):
        setups = 0
        seen = []

    obj = diff.initial(
        {}, objective=diff.objectives.position(dim=0).final(),
        backend=SimpleNamespace(DifferentiableSession=Session),
    )
    return obj, Session


def _fd(loss, x, h=1e-6):
    with torch.no_grad():
        values = []
        for i in range(x.numel()):
            plus, minus = x.clone(), x.clone()
            plus[i] += h
            minus[i] -= h
            values.append((loss(plus) - loss(minus)) / (2 * h))
        return torch.stack(values)


def test_initial_object_parameter_mapping_custom_loss_and_repeated_updates():
    obj, session = _object()
    x = torch.tensor([.8, .3], dtype=torch.float64, requires_grad=True)

    def loss(x):
        velocity = torch.stack((x[0] * x[1].cos(), x[0] * x[1].sin()))
        displacement = torch.stack((x[1].square(), x[1] * 0))
        return (obj(velocity=velocity, displacement=displacement) - .5).square()

    loss(x).backward()
    torch.testing.assert_close(x.grad, _fd(loss, x))
    before = loss(x).detach()
    optimizer = torch.optim.SGD([x], lr=.01)
    for _ in range(2):
        optimizer.zero_grad()
        loss(x).backward()
        optimizer.step()
    assert loss(x).detach() < before
    assert session.setups == 1


def test_initial_object_explicit_fe_layout_and_independent_pending_graphs():
    obj, _ = _object()
    initial = torch.arange(12, dtype=torch.float64).reshape(6, 2).requires_grad_()
    first = obj(initial_condition=initial)
    second = obj(initial_condition=initial * 2)
    (first + second).backward()
    torch.testing.assert_close(initial.grad, 10 * initial.detach())


def test_initial_object_omitted_fields_are_zero_on_each_call():
    obj, session = _object()
    v = torch.tensor([1., 2.], dtype=torch.float64)
    obj(velocity=v, displacement=v * 3)
    obj(velocity=v)
    np.testing.assert_array_equal(session.seen[-1][:6], 0.)
    np.testing.assert_array_equal(session.seen[-1][6:], [1., 2.] * 3)
    obj(displacement=v)
    np.testing.assert_array_equal(session.seen[-1][:6], [1., 2.] * 3)
    np.testing.assert_array_equal(session.seen[-1][6:], 0.)


def test_initial_object_invalid_inputs_and_unsupported_uniform_space():
    obj, _ = _object()
    with pytest.raises(ValueError, match="Supply"):
        obj()
    with pytest.raises(ValueError, match="not both"):
        obj(velocity=torch.zeros(2), initial_condition=torch.zeros(6, 2))
    for bad in (torch.zeros(6), torch.zeros(5, 2), torch.ones(6, 2, dtype=torch.int64),
                torch.full((6, 2), float('nan'))):
        with pytest.raises(ValueError, match="finite floating tensor"):
            obj(initial_condition=bad)
    with pytest.raises(ValueError, match="same dtype"):
        obj(velocity=torch.zeros(2, dtype=torch.float64), displacement=torch.zeros(2))
    with pytest.raises(ValueError, match="finite"):
        obj(velocity=torch.tensor([float('inf'), 0.]))
    with pytest.raises(ValueError, match="shape"):
        obj(velocity=torch.zeros(3))
    with pytest.raises(TypeError, match="torch.Tensor"):
        obj(displacement=[0., 0.])
    from dataclasses import replace
    obj.metadata = replace(obj.metadata, supports_uniform_initial_condition=False)
    with pytest.raises(ValueError, match="unconstrained"):
        obj(velocity=torch.zeros(2))
    assert obj(initial_condition=torch.zeros(6, 2)).item() == 0.
    with pytest.raises(ValueError, match="exactly one"):
        diff.initial(diff.model([{}, {}]), objective=diff.Objective.COMPLIANCE)
    with pytest.raises(ValueError, match="spec itself"):
        diff.initial({}, objective=diff.objectives.position(dim=0), selection=1)


def test_small_fem_initial_velocity_sample_and_composed_gradient(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = Path(__file__).resolve().parents[1] / 'differentiable_example/samples/initial_conditions/initial_velocity.py'
    sample = runpy.run_path(str(path))
    assert sample['x'].item() == pytest.approx(1.02, abs=1e-6)
    obj = sample['obj']
    # Rigid translation: final mean position = u0 + T*v0.
    x = torch.tensor([.7, .03], dtype=torch.float64, requires_grad=True)

    def loss(x):
        v = torch.stack((x[0], x[0] * 0))
        u = torch.stack((x[1], x[1] * 0))
        return (obj(velocity=v, displacement=u) - .2).square()

    value = loss(x)
    value.backward()
    torch.testing.assert_close(value.detach(), x.new_tensor(.01), rtol=1e-4, atol=1e-7)
    torch.testing.assert_close(x.grad, x.new_tensor([-.02, -.2]), rtol=1e-3, atol=1e-6)
    torch.testing.assert_close(x.grad, _fd(loss, x, h=1e-5), rtol=1e-3, atol=1e-6)
    with torch.no_grad():
        after = sample['loss'](sample['x'])
    assert after < sample['value'].detach()
