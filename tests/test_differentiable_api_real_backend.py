from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _run_shape_example(module_name: str):
    example_path = ROOT / "differentiable_example" / "shape" / f"{module_name}.py"
    completed = subprocess.run(
        [sys.executable, str(example_path)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    json_start = completed.stdout.rfind("\n{")
    if json_start == -1:
        json_start = completed.stdout.find("{")
    else:
        json_start += 1
    assert json_start >= 0, completed.stdout
    return json.loads(completed.stdout[json_start:])


def _cube_vertices(torch):
    return torch.tensor(
        [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, -0.5, 0.5],
            [-0.5, -0.5, 0.5],
            [-0.5, 0.5, -0.5],
            [0.5, 0.5, -0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )


def _laplacian_shape_smoke_settings(mesh_path: Path, output_dir: Path) -> dict:
    return {
        "geometry": {
            "mesh": str(mesh_path),
            "surface_selection": [
                {
                    "id": 1,
                    "box": [
                        [-0.501, -0.501, -0.501],
                        [-0.499, 0.501, 0.501],
                    ],
                    "relative": False,
                },
                {
                    "id": 3,
                    "box": [
                        [0.499, -0.501, -0.501],
                        [0.501, 0.501, 0.501],
                    ],
                    "relative": False,
                },
            ],
            "advanced": {"normalize_mesh": False},
        },
        "materials": {"type": "Laplacian"},
        "preset_problem": {"type": "Franke"},
        "solver": {
            "max_threads": 1,
            "linear": {"solver": "Eigen::SimplicialLDLT"},
            "nonlinear": {"line_search": {"use_grad_norm_tol": 1e-5}},
        },
        "time": {"tend": 1, "dt": 1},
        "output": {
            "directory": str(output_dir),
            "json": "",
            "paraview": {"file_name": ""},
            "advanced": {"save_time_sequence": False},
        },
    }


def _bar_face_count(mesh_path: Path) -> int:
    return sum(
        1
        for line in mesh_path.read_text(encoding="utf-8").splitlines()
        if line.startswith("f ")
    )


def _linear_elasticity_compliance_settings(mesh_path: Path, output_dir: Path) -> dict:
    return {
        "geometry": [
            {
                "mesh": str(mesh_path),
                "surface_selection": {"threshold": 0.0001},
                "volume_selection": 1,
            }
        ],
        "space": {
            "discr_order": 1,
            "advanced": {
                "quadrature_order": 2,
                "mass_quadrature_order": 2,
            },
        },
        "solver": {
            "max_threads": 1,
            "linear": {"solver": "Eigen::SimplicialLDLT"},
            "advanced": {
                "characteristic_force_density": 1,
                "characteristic_length": 1,
            },
            "nonlinear": {
                "norm_type": "Euclidean",
                "rel_grad_norm_tol": 0,
                "first_grad_norm_tol": 1e-10,
                "grad_norm_tol": 1e-8,
            },
        },
        "boundary_conditions": {
            "rhs": [10, 100],
            "dirichlet_boundary": [
                {
                    "id": 1,
                    "value": [0.0, 0.0],
                }
            ],
        },
        "materials": {
            "type": "LinearElasticity",
            "lambda": 1.0e5,
            "mu": 5.0e4,
        },
        "output": {
            "directory": str(output_dir),
            "json": "",
            "paraview": {"file_name": ""},
            "advanced": {"save_time_sequence": False},
        },
    }


def _material_loss(diff, diff_model, lame, backend):
    return diff.material_opt(
        diff_model,
        lame,
        objective=diff.Objective.COMPLIANCE,
        objective_params={"selection": 1},
        backend=backend,
    )()


def test_shapeopt_real_backend_forward_backward_smoke(tmp_path):
    backend = pytest.importorskip("polyfempy.polyfempy")
    torch = pytest.importorskip("torch")

    mesh_path = ROOT / "polyfem-data" / "contact" / "meshes" / "3D" / "simple" / "cube.msh"
    if not mesh_path.exists():
        pytest.skip(
            "polyfem-data submodule is not initialized; run "
            "`git submodule update --init polyfem-data`"
        )

    from polyfempy import differentiable_api as diff

    diff_model = diff.model([
        _laplacian_shape_smoke_settings(mesh_path, tmp_path),
    ])
    x = _cube_vertices(torch)

    u = diff.ShapeOpt.apply(diff_model, "all", x, backend)
    loss = u.square().mean()
    loss.backward()

    assert tuple(u.shape) == (8, 1)
    assert u.dtype is torch.float64
    assert u.device == x.device
    assert torch.isfinite(u).all()
    assert x.grad is not None
    assert tuple(x.grad.shape) == tuple(x.shape)
    assert torch.isfinite(x.grad).all()


def test_shapeopt_real_backend_objective_backward_smoke(tmp_path):
    backend = pytest.importorskip("polyfempy.polyfempy")
    torch = pytest.importorskip("torch")

    mesh_path = ROOT / "polyfem-data" / "contact" / "meshes" / "3D" / "simple" / "cube.msh"
    if not mesh_path.exists():
        pytest.skip(
            "polyfem-data submodule is not initialized; run "
            "`git submodule update --init polyfem-data`"
        )

    from polyfempy import differentiable_api as diff

    diff_model = diff.model([
        _laplacian_shape_smoke_settings(mesh_path, tmp_path),
    ])
    x = _cube_vertices(torch)

    loss = diff.ShapeOpt.apply(
        model=diff_model,
        selection="all",
        tensor=x,
        objective=diff.Objective.STRESS_NORM,
        objective_params={
            "selection": 1,
            "power": 8,
        },
        backend=backend,
    )
    loss.backward()

    assert tuple(loss.shape) == ()
    assert loss.dtype is torch.float64
    assert loss.device == x.device
    assert torch.isfinite(loss)
    assert x.grad is not None
    assert tuple(x.grad.shape) == tuple(x.shape)
    assert torch.isfinite(x.grad).all()
    assert not list(tmp_path.rglob("*.vtu"))
    assert not list(tmp_path.rglob("*.pvd"))


def test_materialopt_real_backend_compliance_gradient_and_chain_rule(tmp_path):
    backend = pytest.importorskip("polyfempy.polyfempy")
    torch = pytest.importorskip("torch")

    mesh_path = (
        ROOT
        / "polyfem-data"
        / "contact"
        / "meshes"
        / "2D"
        / "simple"
        / "bar"
        / "bar40.obj"
    )
    if not mesh_path.exists():
        pytest.skip(
            "polyfem-data submodule is not initialized; run "
            "`git submodule update --init polyfem-data`"
        )

    from polyfempy import differentiable_api as diff

    diff_model = diff.model([
        _linear_elasticity_compliance_settings(mesh_path, tmp_path),
    ])
    n_elements = _bar_face_count(mesh_path)
    base_lame = torch.empty((n_elements, 2), dtype=torch.float64)
    base_lame[:, 0] = 1.0e5
    base_lame[:, 1] = 5.0e4

    lame_leaf = base_lame.clone().requires_grad_(True)
    primitive_loss = _material_loss(diff, diff_model, lame_leaf, backend)
    primitive_loss.backward()

    assert tuple(primitive_loss.shape) == ()
    assert primitive_loss.dtype is torch.float64
    assert torch.isfinite(primitive_loss)
    assert lame_leaf.grad is not None
    assert tuple(lame_leaf.grad.shape) == (n_elements, 2)
    assert torch.isfinite(lame_leaf.grad).all()

    direction = torch.zeros_like(base_lame)
    direction[:, 0] = 1.0 / math.sqrt(n_elements)
    primitive_directional = float((lame_leaf.grad * direction).sum())
    primitive_eps = 1e2
    primitive_plus = _material_loss(
        diff,
        diff_model,
        base_lame + primitive_eps * direction,
        backend,
    )
    primitive_minus = _material_loss(
        diff,
        diff_model,
        base_lame - primitive_eps * direction,
        backend,
    )
    primitive_fd = float((primitive_plus - primitive_minus) / (2.0 * primitive_eps))

    assert math.isfinite(primitive_directional)
    assert math.isfinite(primitive_fd)
    assert primitive_directional == pytest.approx(primitive_fd, rel=1e-3, abs=1e-6)

    scale = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    lame = torch.stack(
        (
            base_lame[:, 0] * scale,
            base_lame[:, 1],
        ),
        dim=1,
    )
    loss = _material_loss(diff, diff_model, lame, backend)
    loss.backward()

    eps = 1e-4
    loss_plus = _material_loss(
        diff,
        diff_model,
        torch.stack(
            (
                base_lame[:, 0] * (float(scale.detach()) + eps),
                base_lame[:, 1],
            ),
            dim=1,
        ),
        backend,
    )
    loss_minus = _material_loss(
        diff,
        diff_model,
        torch.stack(
            (
                base_lame[:, 0] * (float(scale.detach()) - eps),
                base_lame[:, 1],
            ),
            dim=1,
        ),
        backend,
    )
    finite_difference = float((loss_plus - loss_minus) / (2.0 * eps))

    assert scale.grad is not None
    assert math.isfinite(float(scale.grad))
    assert math.isfinite(finite_difference)
    assert float(scale.grad) == pytest.approx(finite_difference, rel=1e-3, abs=1e-5)

    composition_scale = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    composition_lame = torch.stack(
        (
            base_lame[:, 0] * composition_scale,
            base_lame[:, 1],
        ),
        dim=1,
    )
    base_loss = _material_loss(diff, diff_model, composition_lame, backend)
    composed_loss = base_loss**2
    composed_loss.backward()
    expected_composition_grad = 2.0 * float(base_loss.detach()) * finite_difference

    assert composition_scale.grad is not None
    assert math.isfinite(float(composition_scale.grad))
    assert float(composition_scale.grad) == pytest.approx(
        expected_composition_grad,
        rel=1e-3,
        abs=1e-5,
    )
    assert not list(tmp_path.rglob("*.vtu"))
    assert not list(tmp_path.rglob("*.pvd"))


def test_neohookean_stress_3d_opt_reference_example_runs_without_persistent_outputs(
    tmp_path,
):
    pytest.importorskip("polyfempy.polyfempy")
    pytest.importorskip("torch")

    opt_spec_path = (
        ROOT
        / "differentiability-data"
        / "input"
        / "neohookean-stress-3d-opt.json"
    )
    if not opt_spec_path.exists():
        pytest.skip(
            "differentiability-data submodule is not initialized; run "
            "`git submodule update --init differentiability-data`"
        )

    summary = _run_shape_example("neohookean_stress_3d_opt")
    output_dir = Path(summary["output_dir"])

    assert summary["objective"] == "stress_norm"
    assert summary["source_opt_spec"].endswith("neohookean-stress-3d-opt.json")
    assert summary["vertices_shape"][1] == 3
    assert math.isfinite(summary["gradient_norm"])
    assert "objective_value" in summary
    assert output_dir == (
        ROOT
        / "differentiable_example"
        / "shape"
        / "runs"
        / "neohookean_stress_3d_opt"
    )
    assert not list(output_dir.rglob("*.vtu"))
    assert not list(output_dir.rglob("*.pvd"))


def test_neohookean_stress_3d_optimization_example_runs_one_step(tmp_path):
    pytest.importorskip("polyfempy.polyfempy")
    pytest.importorskip("torch")

    opt_spec_path = (
        ROOT
        / "differentiability-data"
        / "input"
        / "neohookean-stress-3d-opt.json"
    )
    if not opt_spec_path.exists():
        pytest.skip(
            "differentiability-data submodule is not initialized; run "
            "`git submodule update --init differentiability-data`"
        )

    summary = _run_shape_example("neohookean_stress_3d_optimization")
    output_dir = Path(summary["output_dir"])

    assert summary["objective"] == "stress_norm"
    assert summary["source_opt_spec"].endswith("neohookean-stress-3d-opt.json")
    assert summary["steps"] == 2
    assert summary["lr"] == 1e-7
    assert summary["initial_loss"] == summary["history"][0]["loss"]
    assert summary["last_loss"] == summary["history"][-1]["loss"]
    assert len(summary["history"]) == 2
    assert summary["history"][1]["loss"] != summary["history"][0]["loss"]
    assert all(math.isfinite(item["gradient_norm"]) for item in summary["history"])
    assert output_dir == (
        ROOT
        / "differentiable_example"
        / "shape"
        / "runs"
        / "neohookean_stress_3d_optimization"
    )
    assert not list(output_dir.rglob("*.vtu"))
    assert not list(output_dir.rglob("*.pvd"))


def test_neohookean_stress_3d_chain_rule_example_validates_parameter_grad(tmp_path):
    pytest.importorskip("polyfempy.polyfempy")
    pytest.importorskip("torch")

    opt_spec_path = (
        ROOT
        / "differentiability-data"
        / "input"
        / "neohookean-stress-3d-opt.json"
    )
    if not opt_spec_path.exists():
        pytest.skip(
            "differentiability-data submodule is not initialized; run "
            "`git submodule update --init differentiability-data`"
        )

    summary = _run_shape_example("neohookean_stress_3d_chain_rule")
    output_dir = Path(summary["output_dir"])

    assert summary["objective"] == "stress_norm"
    assert summary["parameter"] == "scale_x"
    assert summary["mapping"] == "vertices[:, 0] = base_vertices[:, 0] * scale"
    assert summary["scale_value"] == 1.0
    assert summary["vertices_shape"][1] == 3
    assert math.isfinite(summary["parameter_grad"])
    assert math.isfinite(summary["finite_difference_gradient"])
    assert summary["finite_difference_abs_error"] < 1e-2
    assert summary["finite_difference_rel_error"] < 1e-5
    assert math.isfinite(summary["composition_gradient"])
    assert math.isfinite(summary["expected_composition_gradient"])
    assert summary["composition_abs_error"] < 1e-6
    assert summary["composition_rel_error"] < 1e-10
    assert output_dir == (
        ROOT
        / "differentiable_example"
        / "shape"
        / "runs"
        / "neohookean_stress_3d_chain_rule"
    )
    assert not list(output_dir.rglob("*.vtu"))
    assert not list(output_dir.rglob("*.pvd"))
