from __future__ import annotations

import sys
import types

import pytest


class _FakeLame:
    pass


class _FakeLoss:
    pass


class _FakeMaterialOpt:
    calls: list[tuple] = []

    @classmethod
    def apply(cls, *args, **kwargs):
        cls.calls.append((args, kwargs))
        return _FakeLoss()


def _install_fake_materialopt(monkeypatch):
    module = types.ModuleType("polyfempy.differentiable_api.torch_ops")
    module.MaterialOpt = _FakeMaterialOpt
    _FakeMaterialOpt.calls = []
    monkeypatch.setitem(
        sys.modules,
        "polyfempy.differentiable_api.torch_ops",
        module,
    )


def test_material_opt_builds_callable_problem_without_optimizer_helpers(monkeypatch):
    _install_fake_materialopt(monkeypatch)

    from polyfempy import differentiable_api as diff

    payload = {"geometry": [{"mesh": "beam.msh"}]}
    diff_model = diff.model([payload])
    lame = _FakeLame()

    material_opt = diff.material_opt(
        diff_model,
        lame,
        objective=diff.Objective.COMPLIANCE,
        objective_params={"selection": 1},
    )

    assert material_opt.model is diff_model
    assert material_opt.lame is lame
    assert material_opt.objective is diff.Objective.COMPLIANCE
    assert material_opt.objective_params == {"selection": 1}
    assert not hasattr(material_opt, "optimizer")
    assert not hasattr(material_opt, "steps")
    assert not hasattr(material_opt, "optimize")

    loss = material_opt()

    assert isinstance(loss, _FakeLoss)
    assert _FakeMaterialOpt.calls == [
        (
            (diff_model, lame),
            {
                "objective": diff.Objective.COMPLIANCE,
                "objective_params": {"selection": 1},
            },
        ),
    ]


def test_material_opt_requires_objective_and_single_model():
    from polyfempy import differentiable_api as diff

    diff_model = diff.model([{"geometry": [{"mesh": "beam.msh"}]}])
    lame = _FakeLame()

    with pytest.raises(ValueError, match="objective"):
        diff.material_opt(diff_model, lame, objective=None)

    with pytest.raises(ValueError, match="exactly one model"):
        diff.material_opt(
            diff.model([
                {"geometry": [{"mesh": "a.msh"}]},
                {"geometry": [{"mesh": "b.msh"}]},
            ]),
            lame,
            objective=diff.Objective.COMPLIANCE,
        )

    with pytest.raises(ValueError, match="lame"):
        diff.material_opt(diff_model, None, objective=diff.Objective.COMPLIANCE)
