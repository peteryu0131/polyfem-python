"""Schema alignment and payload tests; no simulation or native solve."""
import importlib.util
import json
from pathlib import Path

import pytest

from polyfempy import differentiable_api as diff

ROOT = Path(__file__).resolve().parents[1]
ROWS = json.loads((ROOT / "polyfem/json-specs/objective-spec.json").read_text())
DEFINITIONS = {r["type_name"]: r for r in ROWS if r["pointer"] == "/" and "type_name" in r}


def test_snapshot_matches_source():
    spec = importlib.util.spec_from_file_location("generate_objective_schema", ROOT / "tools/generate_objective_schema.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.OUTPUT.read_text() == module.render()
    names = next(r["options"] for r in ROWS if r["pointer"] == "/type")
    assert set(diff.objectives.standard_types) == set(names)
    assert not set(names) & set(diff.objectives.extension_types)


@pytest.mark.parametrize("kind", list(DEFINITIONS))
def test_every_standard_kind_can_construct_configuration(kind):
    required_values = {
        "delta": .1, "mesh_path": "reference.obj", "target_data_path": "target.txt",
        "dim": 0, "value": 1., "parametrization": [], "soft_bound": [0., 1.], "dmin": .01,
        "objective": {"type": "volume", "state": 0},
        "static_objective": {"type": "volume", "state": 0},
    }
    fields = {key: required_values[key] for key in DEFINITIONS[kind]["required"] if key not in ("type", "state")}
    if kind == "divide":
        fields["objective"] = [required_values["objective"], required_values["objective"]]
    value = diff.objectives.create(kind, **fields).as_dict()
    assert value["type"] == kind
    assert set(DEFINITIONS[kind]["required"]) <= value.keys()
    assert set(value) <= set(DEFINITIONS[kind]["required"] + DEFINITIONS[kind].get("optional", []))
    json.dumps(value, allow_nan=False)


def test_surface_volume_selection_and_aliases():
    class Body:
        volume_ref = type("Ref", (), {"backend_id": 2})()
    surface = diff.objectives.displacement_target(surface_selection=[4], target_displacement=[0., 0.])
    assert surface.as_dict()["type"] == "displacement-target"
    assert "volume_selection" not in surface.as_dict()
    assert diff.objectives.create("volume", volume_selection=Body()).as_dict()["volume_selection"] == [2]
    assert diff.objectives.create("volume", volume_selection="ids.txt").as_dict()["volume_selection"] == "ids.txt"
    assert diff.objectives.create("volume", selection=2).as_dict()["volume_selection"] == [2]
    assert diff.objectives.create("target", surface_selection=4).as_dict()["surface_selection"] == 4
    with pytest.raises(TypeError):
        diff.objectives.create("volume", volume_selection=2)
    with pytest.raises(TypeError, match="boundary"):
        diff.objectives.create("target", surface_selection=Body())
    with pytest.raises(ValueError, match="not in"):
        diff.objectives.create("volume", surface_selection=[4])
    with pytest.raises(ValueError, match="not both"):
        diff.objectives.create("volume", volume_selection=[1], selection=[2])


def test_recursive_composition_and_defaults_are_copy_safe():
    spatial = diff.objectives.create("stress_norm", volume_selection=[2])
    time = diff.objectives.transient_integral(static_objective=spatial, integral_type="final")
    result = diff.objectives.power(objective=time, power=2).as_dict()
    assert "state" not in result  # power has no state in the standard.
    assert result["objective"]["static_objective"]["power"] == 2
    assert result["objective"]["steps"] == []
    assert result["weight"] == 1
    result["objective"]["static_objective"]["volume_selection"].append(99)
    assert spatial.as_dict()["volume_selection"] == [2]
    assert time.as_dict()["static_objective"]["volume_selection"] == [2]


@pytest.mark.parametrize("kind,fields", [
    ("position", {}), ("volume", {"typo": 1}),
    ("transient_integral", {"static_objective": {"type": "volume"}, "integral_type": "max"}),
    ("divide", {"objective": {"type": "volume"}}),
    ("power", {"objective": {"type": "not_a_type"}}),
    ("von_mises_peak", {}),
])
def test_invalid_standard_payloads_fail_early(kind, fields):
    with pytest.raises((ValueError, TypeError)):
        diff.objectives.create(kind, **fields)


def test_legacy_helpers_validate_fields_and_remain_compatible():
    assert diff.objectives.max_stress(selection=2).as_dict()["volume_selection"] == [2]
    assert diff.objectives.von_mises_peak(selection=2).as_dict()["type"] == "von_mises_peak"
    assert diff.objectives.von_mises_squared(selection=2).final().as_dict()["integral_type"] == "final"
    with pytest.raises(ValueError, match="not in"):
        diff.objectives.stress_norm(extra={"typo": 3})
