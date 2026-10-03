from __future__ import annotations

import sys
import types

import pytest


class _FakeInitialCondition:
    pass


class _FakeLoss:
    pass


class _FakeInitialConditionOpt:
    calls: list[tuple] = []

    @classmethod
    def apply(cls, *args, **kwargs):
        cls.calls.append((args, kwargs))
        return _FakeLoss()


def _install_fake_initial_condition_opt(monkeypatch):
    module = types.ModuleType("polyfempy.differentiable_api.torch_ops")
    module.InitialConditionOpt = _FakeInitialConditionOpt
    _FakeInitialConditionOpt.calls = []
    monkeypatch.setitem(
        sys.modules,
        "polyfempy.differentiable_api.torch_ops",
        module,
    )


def _transient_objective():
    return {
        "type": "transient_integral",
        "state": 0,
        "integral_type": "final",
        "steps": [],
        "weight": 1.0,
        "print_energy": "",
        "static_objective": {
            "type": "stress_norm",
            "state": 0,
            "volume_selection": [],
            "power": 2,
            "weight": 1.0,
            "print_energy": "",
        },
    }


def test_initial_condition_opt_builds_callable_problem_without_optimizer_helpers(
    monkeypatch,
):
    _install_fake_initial_condition_opt(monkeypatch)

    from polyfempy import differentiable_api as diff

    payload = {"geometry": [{"mesh": "beam.msh"}]}
    diff_model = diff.model([payload])
    initial_condition = _FakeInitialCondition()
    objective = _transient_objective()

    initial_opt = diff.initial_condition_opt(
        diff_model,
        initial_condition,
        objective=objective,
    )

    assert initial_opt.model is diff_model
    assert initial_opt.initial_condition is initial_condition
    assert initial_opt.objective is objective
    assert initial_opt.objective_params is None
    assert not hasattr(initial_opt, "optimizer")
    assert not hasattr(initial_opt, "steps")
    assert not hasattr(initial_opt, "optimize")

    loss = initial_opt()

    assert isinstance(loss, _FakeLoss)
    assert _FakeInitialConditionOpt.calls == [
        (
            (diff_model, initial_condition),
            {
                "objective": objective,
            },
        ),
    ]


def test_initial_condition_opt_accepts_objective_params_for_enum_objective():
    from polyfempy import differentiable_api as diff

    payload = {"geometry": [{"mesh": "beam.msh"}]}
    diff_model = diff.model([payload])
    initial_condition = _FakeInitialCondition()

    initial_opt = diff.initial_condition_opt(
        diff_model,
        initial_condition,
        objective=diff.Objective.STRESS_NORM,
        objective_params={"power": 4},
    )

    assert initial_opt.objective is diff.Objective.STRESS_NORM
    assert initial_opt.objective_params == {"power": 4}


def test_initial_condition_opt_requires_objective_initial_condition_and_single_model():
    from polyfempy import differentiable_api as diff

    diff_model = diff.model([{"geometry": [{"mesh": "beam.msh"}]}])
    initial_condition = _FakeInitialCondition()

    with pytest.raises(ValueError, match="objective"):
        diff.initial_condition_opt(diff_model, initial_condition, objective=None)

    with pytest.raises(ValueError, match="exactly one model"):
        diff.initial_condition_opt(
            diff.model([
                {"geometry": [{"mesh": "a.msh"}]},
                {"geometry": [{"mesh": "b.msh"}]},
            ]),
            initial_condition,
            objective=diff.Objective.STRESS_NORM,
        )

    with pytest.raises(ValueError, match="initial_condition"):
        diff.initial_condition_opt(
            diff_model,
            None,
            objective=diff.Objective.STRESS_NORM,
        )
