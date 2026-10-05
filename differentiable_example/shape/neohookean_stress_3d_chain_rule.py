#!/usr/bin/env python3
"""PyTorch chain-rule ShapeOpt example with a high-level scale parameter."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from polyfempy.generated_api import generated_api as polyfem
from polyfempy import differentiable_api as diff


EXAMPLE_DIR = Path(__file__).resolve().parent
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from _shape_common import MESH_PATH, OPT_SPEC_PATH, gmsh_vertices  # noqa: E402


def build_vertices(base_vertices: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    vertices = base_vertices.clone()
    vertices[:, 0] = base_vertices[:, 0] * scale
    return vertices


# Forward model
model = polyfem.model()

mesh = polyfem.mesh(mesh=str(MESH_PATH))
material = polyfem.neo_hookean(E=100000.0, nu=0.3)
body = polyfem.body(model=model)
body.mesh(mesh)
body.material(material)
body.surface_all(id=1).dirichlet(value=[0.0, 0.0, 0.0])
model.rhs([10, 100, 0])

solver = polyfem.solver(
    max_threads=1,
    linear=polyfem.linear(solver="Eigen::PardisoLDLT"),
    nonlinear=polyfem.nonlinear(
        norm_type="Euclidean", rel_grad_norm_tol=0,
        first_grad_norm_tol=1e-10, grad_norm_tol=1e-8,
    ),
    advanced=polyfem.solver_advanced(characteristic_force_density=1, characteristic_length=1),
)

output_dir = EXAMPLE_DIR / "runs" / "neohookean_stress_3d_chain_rule"
output_dir.mkdir(parents=True, exist_ok=True)

output = polyfem.output(
    directory=str(output_dir), json="",
    paraview=polyfem.output_paraview(file_name=""),
    advanced=polyfem.output_advanced(save_time_sequence=False),
)

polyfem_config = polyfem.config(model=model, solver=solver, output=output)


# Differentiable model
diff_model = diff.model([polyfem_config])
base_vertices = gmsh_vertices(MESH_PATH, dimension=3).detach()

scale = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
vertices = build_vertices(base_vertices, scale)
shape_opt = diff.shape_opt(diff_model, vertices, objective=diff.Objective.STRESS_NORM)

base_loss = shape_opt()
loss = base_loss ** 2
loss.backward()

summary = {
    "source_opt_spec": str(OPT_SPEC_PATH),
    "objective": diff.Objective.STRESS_NORM.value,
    "objective_transform": "square",
    "parameter": "scale_x",
    "mapping": "vertices[:, 0] = base_vertices[:, 0] * scale",
    "scale_value": float(scale.detach()),
    "base_objective_value": float(base_loss.detach()),
    "loss": float(loss.detach()),
    "parameter_grad": float(scale.grad.detach()),
    "vertices_shape": list(vertices.shape),
    "output_dir": str(output_dir),
}

print(json.dumps(summary, indent=2, sort_keys=True))
