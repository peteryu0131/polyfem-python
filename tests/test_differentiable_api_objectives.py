from __future__ import annotations

import importlib
import sys

import pytest


class _SelectionRef:
    def __init__(self, backend_id: int):
        self.backend_id = backend_id


class _BodyHandle:
    def __init__(self, backend_id: int):
        self.volume_ref = _SelectionRef(backend_id)


def test_objective_namespace_builds_stress_norm_spec_from_body_selection():
    from polyfempy import differentiable_api as D

    objective = D.objectives.stress_norm(
        selection=_BodyHandle(3),
        power=8,
        weight=2.5,
    )

    assert objective.as_dict() == {
        "type": "stress_norm",
        "state": 0,
        "volume_selection": [3],
        "power": 8,
        "weight": 2.5,
        "print_energy": "",
    }


def test_objective_namespace_builds_common_backend_objective_specs():
    from polyfempy import differentiable_api as D

    assert D.objectives.max_stress(selection=2).as_dict() == {
        "type": "max_stress",
        "state": 0,
        "volume_selection": [2],
        "weight": 1.0,
        "print_energy": "",
    }
    assert D.objectives.compliance(selection=[1, 4], state=0).as_dict() == {
        "type": "compliance",
        "state": 0,
        "volume_selection": [1, 4],
        "weight": 1.0,
        "print_energy": "",
    }
    assert D.objectives.volume(selection=None, weight=0.1).as_dict() == {
        "type": "volume",
        "state": 0,
        "weight": 0.1,
        "print_energy": "",
    }


def test_top_level_max_stress_helper_matches_meeting_api_shape():
    from polyfempy import differentiable_api as D

    objective = D.MaxStress(selection=_BodyHandle(5), weight=3.0)

    assert objective.as_dict() == {
        "type": "max_stress",
        "state": 0,
        "volume_selection": [5],
        "weight": 3.0,
        "print_energy": "",
    }


def test_top_level_stress_norm_helper_matches_first_differentiable_objective_shape():
    from polyfempy import differentiable_api as D

    objective = D.StressNorm(selection=_BodyHandle(6), power=8, weight=2.0)

    assert objective.as_dict() == {
        "type": "stress_norm",
        "state": 0,
        "volume_selection": [6],
        "power": 8,
        "weight": 2.0,
        "print_energy": "",
    }


def test_objective_specs_are_immutable_and_json_copy_safe():
    from polyfempy import differentiable_api as D

    objective = D.objectives.stress_norm(selection=1, print_energy="stress.csv")
    payload = objective.as_dict()
    payload["volume_selection"].append(9)

    assert objective.as_dict() == {
        "type": "stress_norm",
        "state": 0,
        "volume_selection": [1],
        "power": 2,
        "weight": 1.0,
        "print_energy": "stress.csv",
    }


def test_objective_namespace_rejects_bad_selection_ids():
    from polyfempy import differentiable_api as D

    with pytest.raises(ValueError, match="positive"):
        D.objectives.stress_norm(selection=0)

    with pytest.raises(TypeError, match="backend selection id"):
        D.objectives.stress_norm(selection=object())


def test_objectives_module_import_does_not_load_torch_or_old_reference_package():
    torch_was_loaded = "torch" in sys.modules
    sys.modules.pop("polyfempy.differentiable", None)

    importlib.import_module("polyfempy.differentiable_api.objectives")

    assert "polyfempy.differentiable" not in sys.modules
    if not torch_was_loaded:
        assert "torch" not in sys.modules
