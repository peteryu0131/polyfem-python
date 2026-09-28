from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
DIFFERENTIABILITY_DATA_ROOT = ROOT / "differentiability-data"


def _gmsh_vertices(torch, path: Path, *, dimension: int | None = None):
    lines = path.read_text(encoding="utf-8").splitlines()
    node_marker = lines.index("$Nodes")
    header = lines[node_marker + 1].split()
    vertices_by_tag = {}

    if len(header) == 1:
        node_count = int(header[0])
        for line in lines[node_marker + 2 : node_marker + 2 + node_count]:
            node_id, x, y, z = line.split()
            vertices_by_tag[int(node_id)] = [float(x), float(y), float(z)]
    elif len(header) == 4:
        block_count = int(header[0])
        cursor = node_marker + 2
        for _block in range(block_count):
            _entity_dim, _entity_tag, _parametric, block_node_count = (
                int(value) for value in lines[cursor].split()
            )
            cursor += 1
            tags = [int(lines[cursor + i]) for i in range(block_node_count)]
            cursor += block_node_count
            for tag in tags:
                coordinates = [float(value) for value in lines[cursor].split()]
                cursor += 1
                vertices_by_tag[tag] = coordinates[:3]
    else:
        raise ValueError(f"Unsupported Gmsh $Nodes header: {lines[node_marker + 1]!r}")

    vertices = [vertices_by_tag[tag] for tag in sorted(vertices_by_tag)]
    if dimension is not None:
        vertices = [coords[:dimension] for coords in vertices]
    return torch.tensor(vertices, dtype=torch.float64, requires_grad=True)


def _gmsh22_vertices_2d(torch, path: Path):
    return _gmsh_vertices(torch, path, dimension=2)


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


def _finite_difference_prefix_direction(torch, like, row_count: int):
    direction = torch.zeros_like(like)
    rows = min(row_count, like.shape[0])
    direction[:rows] = torch.arange(
        1,
        rows * like.shape[1] + 1,
        dtype=like.dtype,
        device=like.device,
    ).reshape(rows, like.shape[1])
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


def _differentiability_data_root() -> Path:
    override = os.environ.get("POLYFEMPY_DIFFDATA_ROOT")
    root = _differentiability_data_root_path()

    required = [
        root / "bunny.msh",
        root / "input" / "neohookean-stress-3d.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        joined = ", ".join(missing)
        if override:
            pytest.skip(f"missing differentiability-data fixture file(s): {joined}")
        pytest.skip(
            "differentiability-data submodule is not initialized; run "
            "`git submodule update --init differentiability-data`; "
            f"missing fixture file(s): {joined}"
        )
    return root


def _differentiability_data_root_path() -> Path:
    value = os.environ.get("POLYFEMPY_DIFFDATA_ROOT")
    return Path(value) if value else DIFFERENTIABILITY_DATA_ROOT


def test_differentiability_data_root_defaults_to_submodule(monkeypatch):
    monkeypatch.delenv("POLYFEMPY_DIFFDATA_ROOT", raising=False)

    assert _differentiability_data_root_path() == DIFFERENTIABILITY_DATA_ROOT


def _neohookean_stress_3d_settings(diffdata_root: Path, output_dir: Path) -> dict:
    return _neohookean_stress_3d_reference_case(diffdata_root, output_dir)["settings"]


def _neohookean_stress_3d_reference_case(
    diffdata_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    opt_spec_path = diffdata_root / "input" / "neohookean-stress-3d-opt.json"
    with opt_spec_path.open(encoding="utf-8") as handle:
        opt_spec = json.load(handle)

    if opt_spec.get("parameters") != "auto":
        raise ValueError("neohookean reference expects parameters='auto'")
    if opt_spec.get("variable_to_simulation") != [
        {
            "type": "shape",
            "state": 0,
            "composition": [],
        }
    ]:
        raise ValueError("neohookean reference expects direct shape")
    if opt_spec.get("functionals") != [
        {
            "type": "stress_norm",
            "state": 0,
        }
    ]:
        raise ValueError("neohookean reference expects stress_norm")

    state_path = diffdata_root / "input" / opt_spec["states"][0]["path"]
    with state_path.open(encoding="utf-8") as handle:
        settings = json.load(handle)

    geometry = settings["geometry"]
    if isinstance(geometry, list):
        mesh_path = (state_path.parent / geometry[0]["mesh"]).resolve()
        geometry[0]["mesh"] = str(mesh_path)
    else:
        mesh_path = (state_path.parent / geometry["mesh"]).resolve()
        geometry["mesh"] = str(mesh_path)
    settings.setdefault("solver", {})["max_threads"] = 1

    output = settings.setdefault("output", {})
    output["directory"] = str(output_dir)
    output["json"] = ""
    output.setdefault("paraview", {})["file_name"] = ""
    output.setdefault("advanced", {})["save_time_sequence"] = False
    return {
        "settings": settings,
        "mesh_path": mesh_path,
        "objective": opt_spec["functionals"][0]["type"],
    }


def test_neohookean_reference_case_is_loaded_from_diffdata_opt_spec(tmp_path):
    diffdata_root = _differentiability_data_root()

    case = _neohookean_stress_3d_reference_case(
        diffdata_root,
        tmp_path,
    )

    assert case["objective"] == "stress_norm"
    assert case["mesh_path"] == diffdata_root / "bunny.msh"
    assert case["settings"]["geometry"][0]["mesh"] == str(diffdata_root / "bunny.msh")
    assert case["settings"]["output"]["json"] == ""
    assert case["settings"]["output"]["paraview"]["file_name"] == ""
    assert case["settings"]["output"]["advanced"]["save_time_sequence"] is False


def _loss_with_backend_shape_objective(diff, backend, settings, vertices):
    diff_model = diff.model([settings])
    loss = diff.shape_opt(
        diff_model,
        vertices,
        objective=diff.Objective.STRESS_NORM,
        backend=backend,
    )()
    detach = getattr(loss, "detach", None)
    if callable(detach):
        loss = detach()
    return float(loss)


def test_shapeopt_shape_gradient_matches_finite_difference(tmp_path):
    if os.environ.get("POLYFEMPY_RUN_DIFF_GRAD_CHECK") != "1":
        pytest.skip(
            "set POLYFEMPY_RUN_DIFF_GRAD_CHECK=1 to run the backend gradient check"
        )

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


def test_shapeopt_objective_gradient_matches_differentiability_data_finite_difference(
    tmp_path,
):
    if os.environ.get("POLYFEMPY_RUN_DIFF_GRAD_CHECK") != "1":
        pytest.skip(
            "set POLYFEMPY_RUN_DIFF_GRAD_CHECK=1 to run the backend gradient check"
        )

    backend = pytest.importorskip("polyfempy.polyfempy")
    torch = pytest.importorskip("torch")
    diffdata_root = _differentiability_data_root()

    from polyfempy import differentiable_api as diff

    case = _neohookean_stress_3d_reference_case(
        diffdata_root,
        tmp_path / "adjoint",
    )
    vertices = _gmsh_vertices(torch, case["mesh_path"], dimension=3)
    diff_model = diff.model([case["settings"]])

    loss = diff.shape_opt(
        diff_model,
        vertices,
        objective=diff.Objective(case["objective"]),
        backend=backend,
    )()
    direction = _finite_difference_prefix_direction(torch, vertices, row_count=20)
    loss.backward()

    assert vertices.grad is not None
    adjoint_directional = float((vertices.grad * direction).sum())

    eps = 1e-6
    vertices_np = vertices.detach().numpy()
    direction_np = direction.detach().numpy()
    loss_plus = _loss_with_backend_shape_objective(
        diff,
        backend,
        _neohookean_stress_3d_settings(diffdata_root, tmp_path / "plus"),
        vertices_np + eps * direction_np,
    )
    loss_minus = _loss_with_backend_shape_objective(
        diff,
        backend,
        _neohookean_stress_3d_settings(diffdata_root, tmp_path / "minus"),
        vertices_np - eps * direction_np,
    )
    finite_difference_directional = (loss_plus - loss_minus) / (2 * eps)

    assert adjoint_directional == pytest.approx(
        finite_difference_directional,
        rel=1e-4,
        abs=1e-4,
    )
    assert not list(tmp_path.rglob("*.pvd"))
    assert not list(tmp_path.rglob("*.vtu"))
