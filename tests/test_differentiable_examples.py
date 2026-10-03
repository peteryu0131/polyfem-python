from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIFF_EXAMPLES = ROOT / "differentiable_example"
SHAPE_EXAMPLES = DIFF_EXAMPLES / "shape"
MATERIAL_EXAMPLES = DIFF_EXAMPLES / "material"
INITIAL_CONDITION_EXAMPLES = DIFF_EXAMPLES / "initial_condition"


def _example_text(name: str) -> str:
    return (SHAPE_EXAMPLES / name).read_text(encoding="utf-8")


def _material_example_text(name: str) -> str:
    return (MATERIAL_EXAMPLES / name).read_text(encoding="utf-8")


def _initial_condition_example_text(name: str) -> str:
    return (INITIAL_CONDITION_EXAMPLES / name).read_text(encoding="utf-8")


def test_differentiable_example_directory_contains_expected_files():
    root_files = sorted(path.name for path in DIFF_EXAMPLES.iterdir() if path.is_file())
    shape_files = sorted(path.name for path in SHAPE_EXAMPLES.iterdir() if path.is_file())
    material_files = sorted(
        path.name for path in MATERIAL_EXAMPLES.iterdir() if path.is_file()
    )
    initial_condition_files = sorted(
        path.name for path in INITIAL_CONDITION_EXAMPLES.iterdir() if path.is_file()
    )

    assert root_files == []
    assert shape_files == [
        "_shape_common.py",
        "neohookean_stress_3d_chain_rule.py",
        "neohookean_stress_3d_opt.py",
        "neohookean_stress_3d_optimization.py",
    ]
    assert material_files == [
        "linear_elasticity_compliance.py",
    ]
    assert initial_condition_files == [
        "transient_elastic_initial_condition.py",
    ]


def test_shape_examples_are_direct_user_facing_scripts():
    for name in [
        "neohookean_stress_3d_opt.py",
        "neohookean_stress_3d_optimization.py",
    ]:
        text = _example_text(name)

        assert "# Forward model" in text
        assert "# Differentiable model" in text
        assert "mesh = polyfem.mesh(" in text
        assert "body = polyfem.body(model=model)" in text
        assert "body.mesh(mesh)" in text
        assert "model.rhs([10, 100, 0])" in text
        assert "polyfem_config = polyfem.config(" in text
        assert "model=model" in text
        assert "space=polyfem.space(" not in text
        assert "polyfem.space_advanced(" not in text
        assert "discr_order=1" not in text
        assert "quadrature_order=4" not in text
        assert "diff_model = diff.model([polyfem_config])" in text
        assert "shape_opt = diff.shape_opt(" in text
        assert "objective=diff.Objective.STRESS_NORM" in text

        assert text.index("polyfem_config = polyfem.config(") < text.index(
            "diff_model = diff.model([polyfem_config])"
        )
        assert text.index("diff_model = diff.model([polyfem_config])") < text.index(
            "shape_opt = diff.shape_opt("
        )

        assert "import argparse" not in text
        assert 'if __name__ == "__main__":' not in text
        assert "def " not in text
        assert "example_output_dir" not in text
        assert "diff.ShapeOpt.apply" not in text
        assert "model=diff_model" not in text
        assert "tensor=vertices" not in text
        assert "body = model.mesh(" not in text
        assert "rhs=[10, 100, 0]" not in text


def test_neohookean_stress_3d_opt_example_is_single_gradient_script():
    text = _example_text("neohookean_stress_3d_opt.py")

    assert "loss = shape_opt()" in text
    assert "loss.backward()" in text
    assert "gradient_shape" in text
    assert "gradient_norm" in text
    assert "torch.optim" not in text
    assert "for step in range" not in text


def test_neohookean_stress_3d_optimization_example_uses_plain_pytorch_loop():
    text = _example_text("neohookean_stress_3d_optimization.py")

    assert "# PyTorch optimization" in text
    assert "steps = 2" in text
    assert "lr = 1e-7" in text
    assert "optimizer = torch.optim.Adam([vertices], lr=lr)" in text
    assert "for step in range(steps):" in text
    assert "optimizer.zero_grad()" in text
    assert "loss = shape_opt()" in text
    assert "loss.backward()" in text
    assert "history.append({" in text
    assert "optimizer.step()" in text

    assert ".optimizer(" not in text
    assert ".steps(" not in text
    assert ".optimize(" not in text


def test_linear_elasticity_compliance_material_example_is_direct_user_script():
    text = _material_example_text("linear_elasticity_compliance.py")

    assert "# Forward model" in text
    assert "# Differentiable model" in text
    assert "mesh = polyfem.mesh(" in text
    assert "material = polyfem.linear_elasticity(" in text
    assert "body = polyfem.body(model=model)" in text
    assert "body.mesh(mesh)" in text
    assert "body.material(material)" in text
    assert "body.surface_box(" in text
    assert "model.rhs([10, 100])" in text
    assert "polyfem_config = polyfem.config(" in text
    assert "diff_model = diff.model([polyfem_config])" in text
    assert "material_opt = diff.material_opt(" in text
    assert "objective=diff.Objective.COMPLIANCE" in text
    assert "loss = material_opt()" in text
    assert "loss.backward()" in text
    assert "lame.requires_grad_(True)" in text
    assert "gradient_shape" in text
    assert "gradient_norm" in text

    assert "finite_difference" not in text
    assert "torch.optim" not in text
    assert ".optimizer(" not in text
    assert ".steps(" not in text
    assert ".optimize(" not in text
    assert "diff.MaterialOpt.apply" not in text


def test_transient_elastic_initial_condition_example_uses_high_level_speed_parameter():
    text = _initial_condition_example_text("transient_elastic_initial_condition.py")

    assert "# Forward model" in text
    assert "# Differentiable model" in text
    assert "polyfem_config = {" in text
    assert "diff_model = diff.model([polyfem_config])" in text
    assert "dof_count = initial_condition_dof_count(" in text
    assert "speed = torch.tensor(" in text
    assert "requires_grad=True" in text
    assert "initial_velocity = base_velocity * speed" in text
    assert "initial_condition = torch.stack(" in text
    assert "initial_opt = diff.initial_condition_opt(" in text
    assert "loss = initial_opt()" in text
    assert "loss.backward()" in text
    assert "speed.grad" in text
    assert "initial_condition_shape" in text
    assert "speed_grad" in text

    assert "finite_difference" not in text
    assert "torch.optim" not in text
    assert ".optimizer(" not in text
    assert ".steps(" not in text
    assert ".optimize(" not in text
    assert "diff.InitialConditionOpt.apply" not in text


def test_neohookean_stress_3d_chain_rule_example_uses_high_level_parameter():
    text = _example_text("neohookean_stress_3d_chain_rule.py")

    assert "scale = torch.tensor(" in text
    assert "requires_grad=True" in text
    assert "base_vertices = gmsh_vertices(MESH_PATH, dimension=3).detach()" in text
    assert "vertices = build_vertices(base_vertices, scale)" in text
    assert "vertices[:, 0]" in text
    assert "shape_opt = diff.shape_opt(" in text
    assert "base_loss.backward()" in text
    assert "scale.grad" in text
    assert "vertices_shape" in text
    assert "finite_difference_gradient" in text
    assert "composed_loss = composition_base_loss ** 2" in text
    assert "composition_scale.grad" in text

    assert "gradient_shape" not in text
    assert "optimizer = torch.optim.Adam([vertices]" not in text
    assert "vertices.requires_grad_()" not in text


def test_shape_common_only_contains_mesh_loading_helpers():
    text = _example_text("_shape_common.py")

    assert "MESH_PATH" in text
    assert "OPT_SPEC_PATH" in text
    assert "def gmsh_vertices(" in text
    assert "TemporaryDirectory" not in text
    assert "example_output_dir" not in text


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
