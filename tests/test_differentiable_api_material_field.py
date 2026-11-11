from pathlib import Path
import runpy

import pytest
import torch

from test_differentiable_api_objects import _material_object, _finite_difference


def test_callable_enum_defaults_to_whole_model_selection():
    from polyfempy import differentiable_api as diff
    from test_differentiable_api_material_enu import _backend, _payload
    obj = diff.material(_payload(), body=1, objective=diff.Objective.COMPLIANCE,
                               backend=_backend())
    assert obj.objective["volume_selection"] == []


@pytest.mark.parametrize("dimension", [2, 3])
def test_material_field_chain_rule_units_and_fixed_body(dimension):
    obj, session = _material_object(dimension=dimension, unit="MPa")
    assert obj.element_indices == (1, 3)
    assert obj.n_elements == 2
    x = torch.tensor([2., 3., .2], dtype=torch.float64, requires_grad=True)

    def loss(x):
        return (obj(E=x[:2].square(), nu=x[2]) / 1e5 - 1.).square()

    loss(x).backward()
    torch.testing.assert_close(x.grad, _finite_difference(loss, x), rtol=3e-4, atol=1e-8)
    for material in session.seen:
        # Full backend layout: lambda[0:4], mu[4:8]. Fixed body rows are 0 and 2.
        assert (material[[0, 2, 4, 6]] == session.seen[0][[0, 2, 4, 6]]).all()
    internal, _ = _material_object(dimension=dimension)
    torch.testing.assert_close(obj(E=x[:2], nu=x[2]), internal(E=x[:2]*1e7, nu=x[2]))


def test_material_field_inheritance_and_independent_pending_graphs():
    obj, _ = _material_object(unit="MPa")
    nu = torch.tensor([.2, .3], dtype=torch.float64, requires_grad=True)
    torch.testing.assert_close(obj(nu=nu), obj(E=20., nu=nu))
    E = torch.tensor([18., 22.], dtype=torch.float64, requires_grad=True)
    torch.testing.assert_close(obj(E=E), obj(E=E, nu=.25))
    first = obj(E=E, nu=nu)
    second = obj(E=E.square(), nu=nu)
    (first + second).backward()
    torch.testing.assert_close(E.grad, _finite_difference(lambda e: obj(E=e, nu=nu) + obj(E=e.square(), nu=nu), E),
                               rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(nu.grad, _finite_difference(lambda n: obj(E=E, nu=n) + obj(E=E.square(), nu=n), nu),
                               rtol=1e-4, atol=1e-4)
    # Each new call restores omitted nu from config, not the preceding call.
    obj(E=E, nu=nu)
    torch.testing.assert_close(obj(E=E), obj(E=E, nu=.25))


def test_material_field_repeated_updates():
    obj, _ = _material_object(unit="MPa")
    x = torch.tensor([1., 2.], dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.SGD([x], lr=.01)
    for _ in range(2):
        optimizer.zero_grad()
        obj(E=x.square()).backward()
        optimizer.step()
    assert bool((x < torch.tensor([1., 2.])).all())


@pytest.mark.parametrize("fields", [
    {"E": torch.ones(3)}, {"E": torch.ones(1, 2)}, {"E": torch.ones(2, 1, 1)},
    {"E": torch.ones(2, dtype=torch.int64)}, {"E": torch.tensor([1., -1.])},
    {"nu": torch.tensor([.3, .5])}, {"nu": torch.tensor([-.999, -1.])},
    {"E": torch.tensor([1., float('nan')])}, {"nu": torch.tensor([float('inf'), .2])},
    {"E": torch.ones(2, dtype=torch.float64), "nu": torch.ones(2, dtype=torch.float32)*.3},
])
def test_material_field_invalid_inputs(fields):
    obj, _ = _material_object()
    with pytest.raises(ValueError):
        obj(**fields)


def test_small_fem_material_field_sample_gradient_and_update(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = Path(__file__).resolve().parents[1] / 'differentiable_example/samples/material_and_contact/material_field.py'
    sample = runpy.run_path(str(path))
    assert sample['obj'].n_elements == 2
    assert bool((sample['x'] > 1.).all())
    x = torch.tensor([.9, 1.1], dtype=torch.float64, requires_grad=True)
    loss = sample['loss']
    loss(x).backward()
    assert bool((x.grad.abs() > 1e-10).all())
    for h in (1e-4, 3e-5):
        torch.testing.assert_close(x.grad, _finite_difference(loss, x, h=h), rtol=2e-3, atol=1e-8)
    with torch.no_grad():
        assert loss(sample['x']) < sample['value'].detach()


@pytest.mark.parametrize("dimension", [2, 3])
def test_material_columns_preserve_values_and_shared_parameter_gradients(dimension):
    obj, _ = _material_object(dimension=dimension, unit="MPa")
    x = torch.tensor([20., .25], dtype=torch.float64, requires_grad=True)

    def loss(p, column):
        ones = torch.ones((obj.n_elements, 1) if column else (obj.n_elements,),
                          dtype=p.dtype)
        return obj(E=p[0] * ones, nu=p[1] * ones)

    flat, column = loss(x, False), loss(x, True)
    torch.testing.assert_close(flat, column)
    flat_grad = torch.autograd.grad(flat, x)[0]
    column_grad = torch.autograd.grad(column, x)[0]
    torch.testing.assert_close(flat_grad, column_grad)
    torch.testing.assert_close(column_grad, _finite_difference(lambda p: loss(p, True), x),
                               rtol=3e-4, atol=1e-8)
