"""E/nu conversion checks and tiny static FEM checks; never the impact mesh."""
import json
from types import SimpleNamespace

import pytest
import torch

from polyfempy import differentiable_api as diff
from polyfempy import polyfempy as native
from polyfempy.differentiable_api import torch_ops


def _payload():
    return {"geometry": [{"mesh": "not_loaded_in_mock.msh"}],
            "units": {"length": "cm", "mass": "g", "time": "s"},
            "materials": [{"id": 1, "type": "NeoHookean", "E": {"value": 20., "unit": "MPa"}, "nu": .25},
                          {"id": 2, "type": "NeoHookean", "E": {"value": 200., "unit": "GPa"}, "nu": .3}]}


def _backend(dimension=2):
    class Session:
        def set_settings(self, payload):
            self.payload = payload
        def settings(self):
            return json.dumps(self.payload)
        def model_metadata(self):
            return dict(dimension=dimension, n_vertices=5, n_elements=4, n_bases=5,
                        value_dim=dimension, n_dofs=5*dimension,
                        supports_uniform_initial_condition=False, element_body_ids=[2, 1, 2, 1])
        def set_objective(self, *args): pass
        def set_material_lame_parameters(self, *args): pass
        def solve_material_objective(self): raise AssertionError("No FEM in mapping checks")
        def backward_material(self, *args): raise AssertionError("No FEM in mapping checks")
    return SimpleNamespace(DifferentiableSession=Session,
                           _material_stress_scale=native._material_stress_scale,
                           _material_enu_to_lame=native._material_enu_to_lame,
                           _material_enu_backward=native._material_enu_backward)


def _problem(fields, *, dimension=2, payload=None):
    return diff.material_opt(diff.model(payload or _payload()), materials=fields,
                             objective=diff.Objective.COMPLIANCE, backend=_backend(dimension))


@pytest.mark.parametrize("dimension", [2, 3])
def test_native_conversion_and_both_gradients(dimension):
    values = torch.tensor([[100., .25], [200., .3]], dtype=torch.float64, requires_grad=True)
    converted = torch_ops._MaterialENu.apply(values, dimension, native)
    E, nu = values[:, 0], values[:, 1]
    lam = E*nu / ((1+nu)*(1-2*nu)) if dimension == 3 else E*nu/(1-nu**2)
    expected = torch.stack((lam, E/(2*(1+nu))), dim=1)
    torch.testing.assert_close(converted, expected)
    assert torch.autograd.gradcheck(lambda x: torch_ops._MaterialENu.apply(x, dimension, native), (values,))


@pytest.mark.parametrize("unit,E_value,scale", [("Pa", 20e6, 10.), ("MPa", 20., 1e7), ("GPa", .02, 1e10)])
def test_units_inherit_fixed_body_and_gradient_scale(unit, E_value, scale):
    E = torch.tensor(E_value, dtype=torch.float64, requires_grad=True)
    p = _problem({1: {"E": E, "unit": unit}})
    lame = p._enu.lame()
    torch.testing.assert_close(lame[1], torch.tensor([20e7*.25/(1-.25**2), 20e7/2.5], dtype=torch.float64))
    torch.testing.assert_close(lame[0], torch.tensor([2e12*.3/(1-.3**2), 2e12/2.6], dtype=torch.float64))
    assert torch.autograd.grad(lame[[0, 2]].sum(), E, retain_graph=True)[0] == 0
    lame[[1, 3]].sum().backward()
    assert float(E.grad) == pytest.approx(2*scale*(.25/(1-.25**2)+1/2.5))


def test_nu_only_inherits_unitful_E_and_parameter_order():
    nu = torch.tensor(.25, dtype=torch.float64, requires_grad=True)
    p = _problem({1: {"nu": nu}})
    assert p.parameters == (nu,)
    assert torch.isfinite(torch.autograd.grad(p._enu.lame().sum(), nu)[0])
    E = torch.tensor(20., dtype=torch.float64, requires_grad=True)
    p = _problem({1: {"nu": nu, "E": E, "unit": "MPa"}, 2: {"E": E, "unit": "GPa"}})
    assert len(p.parameters) == 2 and p.parameters[0] is E and p.parameters[1] is nu


def test_body_handle_and_two_updates_rebuild_graph(monkeypatch):
    class Body:
        backend_id = 1
    E = torch.tensor(20., dtype=torch.float64, requires_grad=True)
    nu = torch.tensor(.25, dtype=torch.float64, requires_grad=True)
    p = _problem({Body(): {"E": E, "nu": nu, "unit": "MPa"}})
    seen = []
    def fake_apply(model, lame, **kwargs):
        seen.append(lame.detach().clone())
        return (lame[[1, 3]] / 1e7).square().sum()
    monkeypatch.setattr(torch_ops.MaterialOpt, "apply", fake_apply)
    result = p.optimizer(torch.optim.SGD, lr=1e-5).steps(2).optimize(bounds=[(18., 22.), (.1, .4)])
    assert len(result.history) == len(seen) == 2
    assert not torch.equal(seen[0][1], seen[1][1])
    torch.testing.assert_close(seen[0][[0, 2]], seen[1][[0, 2]])
    assert float(E.detach()) != 20 and float(nu.detach()) != .25
    with torch.no_grad():
        assert not p._enu.lame().requires_grad


@pytest.mark.parametrize("fields", [
    {1: {"E": 20.}}, {1: {"E": -2., "unit": "MPa"}}, {1: {"nu": .5}},
    {1: {"nu": float("nan")}}, {1: {"nu": .2, "unit": "MPa"}},
    {1: {"rho": 100.}}, {1: {"E": "x", "unit": "MPa"}},
    {1: {"E": torch.ones(2), "unit": "MPa"}},
])
def test_invalid_overrides_fail_before_setup(fields):
    with pytest.raises((ValueError, TypeError)):
        _problem(fields)


def test_missing_capability_and_conflicting_input():
    E = torch.tensor(20., dtype=torch.float64)
    with pytest.raises(ValueError, match="not both"):
        diff.material_opt(diff.model(_payload()), torch.ones(4, 2),
                          materials={1: {"E": E, "unit": "MPa"}}, objective=diff.Objective.COMPLIANCE)
    backend = _backend()
    backend._material_enu_to_lame = None
    with pytest.raises(RuntimeError, match="Rebuild"):
        diff.material_opt(diff.model(_payload()), materials={1: {"nu": .3}},
                          objective=diff.Objective.COMPLIANCE, backend=backend)
    with pytest.raises(ValueError, match="Unknown body"):
        _problem({9: {"nu": .3}})
    with pytest.raises((ValueError, RuntimeError)):
        _problem({1: {"E": E, "unit": "kg"}})
    payload = _payload()
    payload["materials"][0]["E"] = "100*x"
    with pytest.raises(ValueError, match="expression"):
        _problem({1: {"nu": .3}}, payload=payload)


@pytest.mark.parametrize("units,expected", [({}, 1e6), ({"length": "cm", "mass": "g", "time": "s"}, 1e7),
                                           ({"length": "mm", "mass": "kg", "time": "s"}, 1000.)])
def test_native_stress_scale(units, expected):
    assert native._material_stress_scale(units, "MPa") == pytest.approx(expected)


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("formulation", ["LinearElasticity", "NeoHookean"])
def test_small_fem_E_nu_total_derivative(tmp_path, dimension, formulation):
    from test_differentiable_api_von_mises import _model
    model, bodies = _model(tmp_path, dimension, formulation, loaded=True)
    E = torch.tensor(100., dtype=torch.float64, requires_grad=True)
    nu = torch.tensor(.25, dtype=torch.float64, requires_grad=True)
    objective = diff.objectives.von_mises_squared(selection=bodies[0])
    problem = diff.material_opt(model, materials={bodies[0]: {"E": E, "nu": nu, "unit": "Pa"}}, objective=objective)
    value = problem()
    value.backward()
    for parameter, h in [(E, .01), (nu, 1e-5)]:
        grad = float(parameter.grad)
        original = float(parameter.detach())
        with torch.no_grad():
            parameter.fill_(original+h)
            plus = float(problem())
            parameter.fill_(original-h)
            minus = float(problem())
            parameter.fill_(original)
        assert grad == pytest.approx((plus-minus)/(2*h), rel=3e-4, abs=1e-6)
    assert not list(tmp_path.rglob('*.vtu'))
