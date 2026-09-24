from __future__ import annotations

from pathlib import Path
import importlib.util

import pytest


ROOT = Path(__file__).resolve().parents[1]


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


def test_shapeopt_laplacian_smoke_example_runs_without_persistent_outputs(tmp_path):
    pytest.importorskip("polyfempy.polyfempy")
    pytest.importorskip("torch")

    example_path = ROOT / "differentiable_example" / "shapeopt_laplacian_smoke.py"
    spec = importlib.util.spec_from_file_location(
        "shapeopt_laplacian_smoke",
        example_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    summary = module.run_smoke(tmp_path)

    assert summary["solution_shape"] == [67, 1]
    assert summary["gradient_shape"] == [242, 2]
    assert summary["gradient_norm"] > 0
    assert not list(tmp_path.rglob("*.vtu"))
    assert not list(tmp_path.rglob("*.pvd"))


def test_shapeopt_objective_smoke_example_runs_without_persistent_outputs(tmp_path):
    pytest.importorskip("polyfempy.polyfempy")
    pytest.importorskip("torch")

    example_path = ROOT / "differentiable_example" / "shapeopt_objective_smoke.py"
    spec = importlib.util.spec_from_file_location(
        "shapeopt_objective_smoke",
        example_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    summary = module.run_smoke(tmp_path)

    assert summary["objective"]["type"] == "stress_norm"
    assert summary["gradient_shape"] == [8, 3]
    assert summary["gradient_norm"] >= 0
    assert "objective_value" in summary
    assert not list(tmp_path.rglob("*.vtu"))
    assert not list(tmp_path.rglob("*.pvd"))


def test_shapeopt_stress_norm_optimization_example_runs_without_persistent_outputs(
    tmp_path,
):
    pytest.importorskip("polyfempy.polyfempy")
    pytest.importorskip("torch")

    example_path = ROOT / "differentiable_example" / "shapeopt_stress_norm_optimization.py"
    spec = importlib.util.spec_from_file_location(
        "shapeopt_stress_norm_optimization",
        example_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    summary = module.run_optimization(tmp_path, steps=2)

    assert summary["objective"]["type"] == "stress_norm"
    assert summary["objective"]["selection"] == "all"
    assert summary["steps"] == 2
    assert len(summary["loss_history"]) == 2
    assert len(summary["gradient_norm_history"]) == 2
    assert summary["vertex_shape"] == [8, 3]
    assert not list(tmp_path.rglob("*.vtu"))
    assert not list(tmp_path.rglob("*.pvd"))
