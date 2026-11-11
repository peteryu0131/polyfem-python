"""Staging must preserve the reference problem, including empty initial arrays."""

import json

import pytest

from tools.validation.compare_original_json import stage


@pytest.mark.parametrize("kind", ["material", "shape"])
def test_reference_staging_preserves_physics_and_optimization_mapping(kind, tmp_path):
    source = stage(kind, tmp_path)
    original = json.loads(source.read_text())
    staged = json.loads((tmp_path / "opt.json").read_text())
    for key in ("parameters", "variable_to_simulation", "functionals"):
        assert staged[key] == original[key]
    assert staged["compute_objective"] is True
    original_state_path = source.parent / original["states"][0]["path"]
    original_state = json.loads(original_state_path.read_text())
    staged_state = json.loads((tmp_path / "state.json").read_text())
    for key in ("materials", "space", "boundary_conditions"):
        assert staged_state[key] == original_state[key]
    for before, after in zip(original_state["geometry"], staged_state["geometry"], strict=True):
        assert after.pop("mesh") == str((original_state_path.parent / before.pop("mesh")).resolve())
        assert after == before
    assert staged_state["solver"]["nonlinear"] == original_state["solver"]["nonlinear"]
    if kind == "material":
        assert staged["parameters"][0]["initial"] == []
