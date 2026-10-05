#!/usr/bin/env python3
"""Small PyTorch optimizer loop for the neohookean-stress-3d ShapeOpt case."""

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

output_dir = EXAMPLE_DIR / "runs" / "neohookean_stress_3d_optimization"
output_dir.mkdir(parents=True, exist_ok=True)

output = polyfem.output(
    directory=str(output_dir), json="",
    paraview=polyfem.output_paraview(file_name=""),
    advanced=polyfem.output_advanced(save_time_sequence=False),
)

polyfem_config = polyfem.config(model=model, solver=solver, output=output)


# Differentiable model
vertices = gmsh_vertices(MESH_PATH, dimension=3)

diff_model = diff.model([polyfem_config])
shape_opt = diff.shape_opt(diff_model, vertices, objective=diff.Objective.STRESS_NORM)


# PyTorch optimization
steps = 2
lr = 1e-7
optimizer = torch.optim.Adam([vertices], lr=lr)
history = []
loss = None

for step in range(steps):
    optimizer.zero_grad()

    loss = shape_opt()
    loss.backward()

    history.append({
        "step": step,
        "loss": float(loss.detach()),
        "gradient_norm": float(vertices.grad.norm()),
    })

    optimizer.step()

summary = {
    "source_opt_spec": str(OPT_SPEC_PATH),
    "objective": diff.Objective.STRESS_NORM.value,
    "steps": steps,
    "lr": lr,
    "initial_loss": history[0]["loss"],
    "last_loss": float(loss.detach()),
    "history": history,
    "output_dir": str(output_dir),
}

print(json.dumps(summary, indent=2, sort_keys=True))
