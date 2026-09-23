#!/usr/bin/env python3
"""Run a minimal differentiable shape smoke test.

This example intentionally uses a tiny Laplacian solve and writes outputs to a
temporary directory by default, so it does not leave VTU/PVD files in the repo.
Use ``--keep-output`` only when you want to inspect the backend workspace.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import torch

import polyfempy.polyfempy as backend
from polyfempy import differentiable_api as diff


ROOT = Path(__file__).resolve().parents[1]
MESH_PATH = ROOT / "polyfem-data" / "circle2.msh"
KEEP_OUTPUT_DIR = ROOT / "differentiable_example" / "runs" / "shapeopt_laplacian_smoke"


def mesh_vertices(path: Path) -> torch.Tensor:
    lines = path.read_text(encoding="utf-8").splitlines()
    node_marker = lines.index("$Nodes")
    node_count = int(lines[node_marker + 1])
    vertices = []
    for line in lines[node_marker + 2 : node_marker + 2 + node_count]:
        _node_id, x, y, _z = line.split()
        vertices.append([float(x), float(y)])
    return torch.tensor(vertices, dtype=torch.float64, requires_grad=True)


def solution_interior_mask(vertices: torch.Tensor, solution_size: int) -> torch.Tensor:
    solution_vertices = vertices.detach()[:solution_size]
    radius = torch.linalg.vector_norm(solution_vertices, dim=1)
    return radius < radius.max() * 0.9


def forward_settings(output_dir: Path) -> dict[str, Any]:
    return {
        "geometry": {
            "mesh": str(MESH_PATH),
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


def run_smoke(output_dir: Path) -> dict[str, Any]:
    vertices = mesh_vertices(MESH_PATH)
    diff_model = diff.model([forward_settings(output_dir)])

    u = diff.ShapeOpt.apply(
        model=diff_model,
        selection="all",
        tensor=vertices,
        backend=backend,
    )
    mask = solution_interior_mask(vertices, u.shape[0])
    loss = u[mask].square().mean()
    loss.backward()

    gradient_norm = float(vertices.grad.norm())
    return {
        "loss": float(loss.detach()),
        "solution_shape": list(u.shape),
        "gradient_shape": list(vertices.grad.shape),
        "gradient_norm": gradient_norm,
        "output_dir": str(output_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--keep-output",
        action="store_true",
        help="Keep the backend workspace under differentiable_example/runs/.",
    )
    args = parser.parse_args()

    if not MESH_PATH.exists():
        raise FileNotFoundError(
            f"Missing mesh fixture: {MESH_PATH}. "
            "Initialize submodules before running this example."
        )

    if args.keep_output:
        KEEP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        summary = run_smoke(KEEP_OUTPUT_DIR)
    else:
        with TemporaryDirectory(prefix="polyfempy-diff-smoke-") as tmp:
            summary = run_smoke(Path(tmp))

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
