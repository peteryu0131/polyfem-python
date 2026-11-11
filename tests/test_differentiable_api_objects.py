"""Callable outputs: user parameter mappings, outer losses, and fresh graphs."""

from types import SimpleNamespace

import pytest
import torch

from polyfempy import differentiable_api as diff
from test_differentiable_api_material_enu import _backend, _payload


def _material_object(*, dimension=2, **kwargs):
    backend = _backend(dimension)

    class Session(backend.DifferentiableSession):
        seen = []
        objectives = []
        setups = 0

        def model_metadata(self):
            type(self).setups += 1
            return super().model_metadata()

        def set_material_lame_parameters(self, lame):
            self.lame = lame.copy()
            self.seen.append(self.lame)

        def set_objective(self, objective):
            self.objectives.append(objective)

        def solve_material_objective(self):
            return (self.lame / 1e7).sum()

        def backward_material(self, gradient):
            import numpy as np
            return np.ones_like(self.lame) * float(gradient) / 1e7

    backend.DifferentiableSession = Session
    obj = diff.material(_payload(), body=1, selection=2,
                               objective=diff.Objective.VON_MISES_PEAK,
                               backend=backend, **kwargs)
    return obj, Session


def _finite_difference(loss, x, h=1e-6):
    values = []
    with torch.no_grad():
        for index in range(x.numel()):
            plus, minus = x.clone(), x.clone()
            plus[index] += h
            minus[index] -= h
            values.append(float((loss(plus) - loss(minus)) / (2*h)))
    return x.new_tensor(values)


@pytest.mark.parametrize("dimension", [2, 3])
def test_material_parameter_mapping_custom_loss_and_two_updates(dimension):
    obj, session = _material_object(dimension=dimension)
    x = torch.tensor([2., .25], dtype=torch.float64, requires_grad=True)

    def loss(x):
        prediction = obj(E=x[0].square() * 1e7, nu=x[1])
        return ((prediction - 1.) / 1e5).square() + .01 * x[0].square()

    value = loss(x)
    value.backward()
    torch.testing.assert_close(x.grad, _finite_difference(loss, x), rtol=2e-4, atol=1e-8)
    optimizer = torch.optim.SGD([x], lr=.01)
    for _ in range(2):
        optimizer.zero_grad()
        loss(x).backward()
        optimizer.step()
    assert x[0].item() != 2.
    assert session.setups == 1
    assert session.objectives[0]["volume_selection"] == [2]
    assert session.objectives[0]["weight"] == 1.
    # Backend layout is all lambda, then all mu; body 2 occupies rows 0 and 2.
    for current in session.seen:
        assert (current[[0, 2, 4, 6]] == session.seen[0][[0, 2, 4, 6]]).all()


def test_material_units_inheritance_and_independent_pending_calls():
    internal, _ = _material_object()
    mpa, _ = _material_object(unit="MPa")
    x = torch.tensor([20., .25], dtype=torch.float64, requires_grad=True)
    torch.testing.assert_close(internal(E=x[0]*1e7), mpa(E=x[0]))
    torch.testing.assert_close(internal(nu=x[1]), internal(E=x[0]*1e7, nu=x[1]))
    # One object may have several live forward graphs before backward.
    first = mpa(E=x[0], nu=x[1])
    second = mpa(E=2*x[0], nu=x[1])
    torch.autograd.backward((first, second))
    assert torch.isfinite(x.grad).all() and bool((x.grad != 0).all())
    torch.testing.assert_close(
        x.grad,
        _finite_difference(lambda p: mpa(E=p[0], nu=p[1]) + mpa(E=2*p[0], nu=p[1]), x),
        rtol=2e-5, atol=1e-5,
    )
    with pytest.raises(ValueError, match="Supply"):
        internal()
    with pytest.raises(ValueError, match="finite E"):
        internal(E=-1.)
    # A call omitting nu must inherit config, not retain the previous call's nu.
    with torch.no_grad():
        internal(E=20e7, nu=.4)
        torch.testing.assert_close(internal(E=20e7), mpa(E=20.))


class _ShapeSession:
    objectives = []

    def set_settings(self, config):
        pass

    def set_objective(self, objective):
        self.objectives.append(objective)

    def set_shape_vertices(self, vertices, *, selection):
        assert selection == "all"
        self.vertices = vertices.copy()

    def solve(self):
        raise AssertionError("Expected a scalar objective")

    def solve_objective(self):
        return (self.vertices**2).sum()

    def backward_shape(self, gradient):
        return 2 * self.vertices * float(gradient)


def test_shape_geometry_custom_loss_and_two_updates():
    obj = diff.shape({}, selection=2, objective=diff.Objective.VON_MISES_PEAK,
                            backend=SimpleNamespace(DifferentiableSession=_ShapeSession))
    x = torch.tensor([.8, .3], dtype=torch.float64, requires_grad=True)

    def loss(x):
        V = torch.stack((x[0] * x[1].cos(), x[0] * x[1].sin() + 1)).reshape(1, 2)
        return (obj(V=V) - .5).square() + .1*x[0].square()

    loss(x).backward()
    torch.testing.assert_close(x.grad, _finite_difference(loss, x))
    optimizer = torch.optim.SGD([x], lr=.01)
    before = loss(x).detach()
    for _ in range(2):
        optimizer.zero_grad()
        loss(x).backward()
        optimizer.step()
    assert loss(x).detach() < before
    assert _ShapeSession.objectives[-1]["volume_selection"] == [2]
    with pytest.raises(ValueError, match="V must"):
        obj(V=torch.zeros(2))
    with pytest.raises(TypeError):
        obj(V=torch.zeros(1, 2), F=torch.zeros(1, 3))


def test_objective_spec_and_selection_conflicts():
    spec = diff.objectives.von_mises_peak(selection=2)
    obj = diff.shape(diff.model({}), objective=spec)
    assert obj.objective["volume_selection"] == [2]
    with pytest.raises(ValueError, match="spec itself"):
        diff.shape({}, objective=spec, selection=1)
    with pytest.raises(ValueError, match="exactly one"):
        diff.shape(diff.model([{}, {}]), objective=spec)


def test_small_fem_material_object_composed_loss(tmp_path):
    from test_differentiable_api_von_mises import _model
    model, bodies = _model(tmp_path, 2, "NeoHookean", loaded=True)
    obj = diff.material(model, body=bodies[0], selection=bodies[0],
                               objective=diff.Objective.VON_MISES_PEAK)
    x = torch.tensor([100., .25], dtype=torch.float64, requires_grad=True)

    def loss(x):
        return (obj(E=x[0], nu=x[1]) / 10. - 1.).square()

    loss(x).backward()
    torch.testing.assert_close(x.grad, _finite_difference(loss, x, h=1e-5),
                               rtol=3e-3, atol=1e-6)


def test_small_fem_shape_object_composed_loss(tmp_path):
    import meshio
    import numpy as np
    from test_differentiable_api_von_mises import _model
    model, bodies = _model(tmp_path, 2, "NeoHookean", loaded=True)
    obj = diff.shape(model, selection=bodies[0], objective=diff.Objective.VON_MISES_PEAK)
    base = torch.tensor(np.concatenate([meshio.read(tmp_path / f"body{i}.msh").points[:, :2]
                                        for i in range(2)]), dtype=torch.float64)
    direction = torch.zeros_like(base)
    direction[4] = torch.tensor([.7, -.3])
    x = torch.tensor([.03], dtype=torch.float64, requires_grad=True)

    def loss(x):
        return (obj(V=base + x[0]*direction) / 10. - 1.).square()

    loss(x).backward()
    torch.testing.assert_close(x.grad, _finite_difference(loss, x, h=1e-5),
                               rtol=3e-3, atol=1e-6)
