#!/usr/bin/env python3
"""Run a tiny stress-norm shape optimization loop.

This example is intentionally small. It shows the API shape for a user-facing
PyTorch optimization loop without writing VTU/PVD output files by default.
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
MESH_PATH = ROOT / "polyfem-data" / "contact" / "meshes" / "3D" / "simple" / "cube.msh"
KEEP_OUTPUT_DIR = ROOT / "differentiable_example" / "runs" / "shapeopt_stress_norm"


def mesh_vertices(path: Path) -> torch.Tensor:
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
                coords = [float(value) for value in lines[cursor].split()]
                cursor += 1
                vertices_by_tag[tag] = coords[:3]
    else:
        raise ValueError(f"Unsupported Gmsh $Nodes header: {lines[node_marker + 1]!r}")

    vertices = [vertices_by_tag[tag] for tag in sorted(vertices_by_tag)]
    return torch.tensor(vertices, dtype=torch.float64, requires_grad=True)


def forward_settings(output_dir: Path) -> dict[str, Any]:
    return {
        "geometry": {
            "mesh": str(MESH_PATH),
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


def run_optimization(
    output_dir: Path,
    *,
    steps: int = 3,
    learning_rate: float = 1e-4,
) -> dict[str, Any]:
    vertices = mesh_vertices(MESH_PATH)
    diff_model = diff.model([forward_settings(output_dir)])
    optimizer = torch.optim.Adam([vertices], lr=learning_rate)
    loss_history = []
    gradient_norm_history = []

    for _step in range(steps):
        optimizer.zero_grad()

        loss = diff.ShapeOpt.apply(
            model=diff_model,
            selection="all",
            tensor=vertices,
            objective=diff.Objective.STRESS_NORM,
            backend=backend,
        )
        loss.backward()

        loss_history.append(float(loss.detach()))
        gradient_norm_history.append(float(vertices.grad.norm()))
        optimizer.step()

    return {
        "objective": {
            "type": diff.Objective.STRESS_NORM.value,
            "selection": "all",
        },
        "steps": steps,
        "learning_rate": learning_rate,
        "loss_history": loss_history,
        "gradient_norm_history": gradient_norm_history,
        "vertex_shape": list(vertices.shape),
        "output_dir": str(output_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
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
        summary = run_optimization(
            KEEP_OUTPUT_DIR,
            steps=args.steps,
            learning_rate=args.learning_rate,
        )
    else:
        with TemporaryDirectory(prefix="polyfempy-diff-shapeopt-") as tmp:
            summary = run_optimization(
                Path(tmp),
                steps=args.steps,
                learning_rate=args.learning_rate,
            )

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
