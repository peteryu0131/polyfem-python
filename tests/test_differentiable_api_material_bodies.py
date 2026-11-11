from pathlib import Path
import runpy

import pytest
import torch
from polyfempy import differentiable_api as diff
from test_differentiable_api_objects import _material_object, _finite_difference
from test_differentiable_api_material_enu import _payload


def test_body_mapping_chain_rule_units_and_inheritance():
    single, session = _material_object(unit="MPa")
    obj = diff.material(_payload(), objective=diff.Objective.COMPLIANCE,
                               backend=single._enu.backend)
    x = torch.tensor([20., 30., .2], dtype=torch.float64, requires_grad=True)

    def loss(x):
        return obj(materials={1: {"E": x[0], "unit": "MPa"},
                              2: {"E": x[1], "unit": "MPa", "nu": x[2]}}).square()

    loss(x).backward()
    torch.testing.assert_close(x.grad, _finite_difference(loss, x), rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(obj(materials={1: {"E": x[0], "unit": "MPa"}}), single(E=x[0]))
    with pytest.raises(ValueError):
        obj(E=x[0])
    with pytest.raises(ValueError):
        obj(materials={})
    with pytest.raises(ValueError):
        obj(materials={99: {"E": x[0]}})
    with pytest.raises(ValueError):
        single(materials={1: {"E": x[0]}})
    with pytest.raises(ValueError):
        obj(materials={1: {"E": x}})


def test_small_fem_body_mapping_sample(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sample = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                               "differentiable_example/samples/material_and_contact/material_bodies.py"))
    x = torch.tensor([.9, 1.1], dtype=torch.float64, requires_grad=True)
    loss = sample['loss']
    loss(x).backward()
    assert bool((x.grad.abs() > 1e-10).all())
    for h in (1e-4, 3e-5):
        torch.testing.assert_close(x.grad, _finite_difference(loss, x, h=h),
                                   rtol=2e-3, atol=1e-8)
    with torch.no_grad():
        assert loss(sample['x']) < sample['value'].detach()

@pytest.mark.parametrize('dimension', [2, 3])
def test_body_fields_order_units_shared_parameters_and_defaults(dimension):
    single, session = _material_object(dimension=dimension, unit='MPa')
    obj = diff.material(_payload(), objective=diff.Objective.COMPLIANCE,
                               backend=single._enu.backend)
    assert obj.element_indices_by_body == {1: (1, 3), 2: (0, 2)}
    x = torch.tensor([18., 22., .2], dtype=torch.float64, requires_grad=True)

    def loss(x):
        return obj(materials={2: {'E': x[:2].flip(0), 'unit': 'MPa', 'nu': x[2]},
                              1: {'E': x[:2], 'unit': 'MPa'}}).square()

    loss(x).backward()
    torch.testing.assert_close(x.grad, _finite_difference(loss, x), rtol=2e-4, atol=1e-6)
    torch.testing.assert_close(obj(materials={1: {'E': x[:2], 'unit': 'MPa'}}),
                               single(E=x[:2]))
    torch.testing.assert_close(obj(materials={1: {'nu': x[2].expand(2)}, 2: {'E': 30., 'unit': 'MPa'}}),
                               obj(materials={1: {'nu': x[2]}, 2: {'E': 30., 'unit': 'MPa'}}))


@pytest.mark.parametrize('fields', [
    {1: {'E': torch.ones(3)}},
    {1: {'E': torch.ones(2, dtype=torch.int64)}},
    {1: {'E': torch.tensor([1., float('nan')])}},
    {1: {'nu': torch.tensor([.2, .5])}},
    {1: {'E': torch.ones(2), 'unit': None}},
    {1: {'E': torch.ones(2), 'nu': None}},
    {1: {'E': torch.ones(2)}, 2: {'E': torch.ones(2, dtype=torch.float64)}},
])
def test_body_fields_invalid_inputs(fields):
    single, _ = _material_object()
    obj = diff.material(_payload(), objective=diff.Objective.COMPLIANCE,
                               backend=single._enu.backend)
    with pytest.raises(ValueError):
        obj(materials=fields)


@pytest.mark.parametrize('filename,size', [('material_body_fields.py', 4), ('material_load_cases.py', 2)])
def test_small_fem_fields_and_load_cases(tmp_path, monkeypatch, filename, size):
    monkeypatch.chdir(tmp_path)
    sample = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                               'differentiable_example/samples/material_and_contact' / filename))
    x = torch.linspace(.9, 1.1, size, dtype=torch.float64, requires_grad=True)
    loss = sample['loss']
    loss(x).backward()
    assert bool((x.grad.abs() > 1e-10).all())
    for h in (1e-4, 3e-5):
        torch.testing.assert_close(x.grad, _finite_difference(loss, x, h=h),
                                   rtol=2e-3, atol=1e-8)
    if filename == 'material_load_cases.py':
        a = torch.autograd.grad(sample['vertical'](E=100*x.pow(5)), x)[0]
        b = torch.autograd.grad(sample['horizontal'](E=100*x.pow(5)), x)[0]
        torch.testing.assert_close(x.grad, a + b)
    with torch.no_grad():
        assert loss(sample['x']) < sample['value'].detach()
