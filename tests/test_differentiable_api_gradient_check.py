from __future__ import annotations

import os
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _gmsh22_vertices_2d(torch, path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    node_marker = lines.index("$Nodes")
    node_count = int(lines[node_marker + 1])
    vertices = []
    for line in lines[node_marker + 2 : node_marker + 2 + node_count]:
        _node_id, x, y, _z = line.split()
        vertices.append([float(x), float(y)])
    return torch.tensor(vertices, dtype=torch.float64, requires_grad=True)


def _solution_interior_mask(torch, vertices, solution_size: int):
    solution_vertices = vertices.detach()[:solution_size]
    radius = torch.linalg.vector_norm(solution_vertices, dim=1)
    return radius < radius.max() * 0.9


def _finite_difference_direction(torch, like, active_rows):
    direction = torch.zeros_like(like)
    active_indices = torch.nonzero(active_rows, as_tuple=False).flatten()
    direction[active_indices] = torch.arange(
        1,
        active_indices.numel() * like.shape[1] + 1,
        dtype=like.dtype,
    ).reshape(-1, like.shape[1])
    return direction / direction.norm()


def _laplacian_shape_settings(mesh_path: Path, output_dir: Path) -> dict:
    return {
        "geometry": {
            "mesh": str(mesh_path),
            "advanced": {"normalize_mesh": False},
        },
        "materials": {"type": "Laplacian"},
        "preset_problem": {"type": "Franke"},
        "space": {"advanced": {"isoparametric": True}},
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


def _loss_with_backend_shape_solve(diff, backend, diff_model, vertices, solution_mask):
    solution = diff.shape_solve(
        model=diff_model,
        selection="all",
        tensor=vertices,
        backend=backend,
    )
    return float((solution[solution_mask] * solution[solution_mask]).mean())


def test_shapeopt_shape_gradient_matches_finite_difference(tmp_path):
    if os.environ.get("POLYFEMPY_RUN_DIFF_GRAD_CHECK") != "1":
        pytest.skip("set POLYFEMPY_RUN_DIFF_GRAD_CHECK=1 to run the backend gradient check")

    backend = pytest.importorskip("polyfempy.polyfempy")
    torch = pytest.importorskip("torch")

    mesh_path = ROOT / "polyfem-data" / "circle2.msh"
    if not mesh_path.exists():
        pytest.skip(
            "polyfem-data submodule is not initialized; run "
            "`git submodule update --init polyfem-data`"
        )

    from polyfempy import differentiable_api as diff

    diff_model = diff.model([
        _laplacian_shape_settings(mesh_path, tmp_path),
    ])
    vertices = _gmsh22_vertices_2d(torch, mesh_path)

    u = diff.ShapeOpt.apply(diff_model, "all", vertices, backend)
    solution_mask = _solution_interior_mask(torch, vertices, u.shape[0])
    loss = u[solution_mask].square().mean()
    direction = _finite_difference_direction(torch, vertices, solution_mask)
    loss.backward()

    assert vertices.grad is not None
    adjoint_directional = float((vertices.grad * direction).sum())

    eps = 1e-6
    vertices_np = vertices.detach().numpy()
    direction_np = direction.detach().numpy()
    loss_plus = _loss_with_backend_shape_solve(
        diff,
        backend,
        diff_model,
        vertices_np + eps * direction_np,
        solution_mask.detach().numpy(),
    )
    loss_minus = _loss_with_backend_shape_solve(
        diff,
        backend,
        diff_model,
        vertices_np - eps * direction_np,
        solution_mask.detach().numpy(),
    )
    finite_difference_directional = (loss_plus - loss_minus) / (2 * eps)

    assert adjoint_directional == pytest.approx(
        finite_difference_directional,
        rel=5e-2,
        abs=1e-5,
    )
