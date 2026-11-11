from copy import deepcopy
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest
import torch

from polyfempy import differentiable_api as diff
from polyfempy.differentiable_api._backend import BackendContractError
from test_differentiable_api_initial_object import _InitialSession


def test_initial_target_validation_copy_and_old_backend():
    objective = diff.objectives.target(target_state=1).final()
    target = {"initial_conditions": {"velocity": [2., 0.]}}
    obj = diff.initial({}, target=target, objective=objective,
                       backend=SimpleNamespace(DifferentiableSession=_InitialSession))
    target["initial_conditions"]["velocity"][0] = 3.
    assert obj._target["initial_conditions"]["velocity"][0] == 2.
    with pytest.raises(BackendContractError, match="Rebuild.*set_target_settings"):
        obj(velocity=torch.ones(2, dtype=torch.float64))
    with pytest.raises(ValueError, match="requires target"):
        diff.initial({}, objective=objective)
    for invalid in (diff.Objective.COMPLIANCE, diff.objectives.target(target_state=1),
                    diff.objectives.target(target_state=0).final()):
        with pytest.raises(ValueError, match="target requires"):
            diff.initial({}, target={}, objective=invalid)


def test_initial_target_pending_graphs_and_outer_loss():
    class Session(_InitialSession):
        def set_target_settings(self, target):
            self.reference = target["reference"]

        def solve_initial_condition_objective(self):
            return ((self.initial - self.reference)**2).sum()

        def backward_initial_condition(self, gradient):
            return 2 * (self.initial - self.reference) * float(gradient)

    obj = diff.initial({}, target={"reference": .3},
                       objective=diff.objectives.target(target_state=1).final(),
                       backend=SimpleNamespace(DifferentiableSession=Session))
    x = torch.ones(6, 2, dtype=torch.float64, requires_grad=True)
    a, b = obj(initial_condition=x), obj(initial_condition=x * 2)
    (a.square() + b).backward()
    expected = 4 * ((x.detach() - .3)**2).sum() * (x.detach() - .3) + 4 * (2*x.detach() - .3)
    torch.testing.assert_close(x.grad, expected)


def test_real_initial_target_sample_value_gradient_and_update(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = Path(__file__).resolve().parents[1] / "differentiable_example/samples/initial_conditions/initial_matching.py"
    sample = runpy.run_path(str(path))
    obj = sample["obj"]
    x = torch.tensor([.7, .3], dtype=torch.float64, requires_grad=True)
    value = obj(velocity=x)
    value.backward()
    # Target integrates over the unit square's boundary (length 4), T=.1.
    expected = .04 * ((x.detach() - x.new_tensor([2., 0.]))**2).sum()
    torch.testing.assert_close(value.detach(), expected, rtol=1e-4, atol=1e-8)
    torch.testing.assert_close(x.grad, .08 * (x.detach() - x.new_tensor([2., 0.])), rtol=1e-3, atol=1e-7)
    with torch.no_grad():
        for h in (1e-4, 3e-5):
            direction = x.new_tensor([.4, -.7])
            fd = (obj(velocity=x+h*direction) - obj(velocity=x-h*direction))/(2*h)
            torch.testing.assert_close(x.grad @ direction, fd, rtol=1e-3, atol=1e-8)
        assert obj(velocity=x.new_tensor([2., 0.])).item() == pytest.approx(0., abs=1e-12)
        assert sample["loss"](sample["x"]) < sample["value"].detach()


def test_real_initial_target_rejects_time_mismatch(tmp_path, monkeypatch):
    from polyfempy import polyfempy as native
    monkeypatch.chdir(tmp_path)
    config = {
        "geometry": {"mesh": str(Path(__file__).resolve().parents[1] / "differentiability-data/square.obj")},
        "materials": {"type": "NeoHookean", "E": 100., "nu": .3, "rho": 1.},
        "time": {"dt": .05, "tend": .1},
        "solver": {"max_threads": 1, "linear": {"solver": "Eigen::SimplicialLDLT"}},
    }
    session = native.DifferentiableSession()
    session.set_settings(config)
    initial = session.initial_condition_parameters()
    assert initial.size == 2 * session.initial_condition_dof_count()
    session.set_initial_condition_parameters(initial)
    target = deepcopy(config)
    target["time"]["tend"] = .2
    with pytest.raises(RuntimeError, match="identical time settings"):
        session.set_target_settings(target)
    session.set_target_settings(config)
    session.set_objective(diff.objectives.target(target_state=1).final().as_dict())
