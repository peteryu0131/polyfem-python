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
    assert summary["gradient_shape"][1] == 3
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
    assert summary["gradient_shape"][1] == 3
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
