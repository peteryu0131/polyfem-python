from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
DIFF_EXAMPLES = ROOT / "differentiable_example"
EXAMPLES = (
    "shape/neohookean_stress_3d_opt.py",
    "shape/neohookean_stress_3d_optimization.py",
    "shape/neohookean_stress_3d_chain_rule.py",
    "material/linear_elasticity_compliance.py",
    "initial_condition/transient_elastic_initial_condition.py",
    "initial_condition/projectile_angle.py",
)


@pytest.mark.parametrize("relative_path", EXAMPLES)
def test_examples_compile_and_use_public_objective_api(relative_path):
    path = DIFF_EXAMPLES / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    compile(tree, str(path), "exec")
    calls = [ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]
    kind = relative_path.split("/")[0]
    assert f"diff.{kind}_opt" in calls
    # Keep examples on public wrappers; check numerical behavior in backend tests.
    assert not any(call in {"diff.ShapeOpt.apply", "diff.MaterialOpt.apply",
                            "diff.InitialConditionOpt.apply"} for call in calls)
    if kind in {"material", "initial_condition"}:
        assert "diff_model.metadata" in calls
        helper = "uniform_material" if kind == "material" else "uniform_initial_condition"
        assert f"metadata.{helper}" in calls


def test_neohookean_stress_3d_opt_example_matches_diffdata_reference_spec():
    opt_spec = json.loads(
        (ROOT / "differentiability-data" / "input" / "neohookean-stress-3d-opt.json")
        .read_text(encoding="utf-8")
    )

    assert opt_spec["parameters"] == "auto"
    assert opt_spec["variable_to_simulation"] == [
        {
            "type": "shape",
            "state": 0,
            "composition": [],
        }
    ]
    assert opt_spec["functionals"] == [
        {
            "type": "stress_norm",
            "state": 0,
        }
    ]
    assert opt_spec["states"] == [{"path": "neohookean-stress-3d.json"}]
