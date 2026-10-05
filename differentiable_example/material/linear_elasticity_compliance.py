#!/usr/bin/env python3
"""Minimal MaterialOpt compliance example with per-element Lame parameters."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from polyfempy.generated_api import generated_api as polyfem
from polyfempy import differentiable_api as diff


EXAMPLE_DIR = Path(__file__).resolve().parent
ROOT = EXAMPLE_DIR.parents[1]
MESH_PATH = ROOT / "polyfem-data" / "contact" / "meshes" / "2D" / "simple" / "bar" / "bar40.obj"


def obj_face_count(path: Path) -> int:
    # TODO: replace this OBJ-only helper with mesh/diff_model metadata.
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("f "))


# Forward model
model = polyfem.model()

mesh = polyfem.mesh(mesh=str(MESH_PATH))
# MaterialOpt overwrites the elastic Lame parameters before each solve.
material = polyfem.linear_elasticity(E=1.0e5, nu=0.3)
body = polyfem.body(model=model)
body.mesh(mesh)
body.material(material)
body.surface_box(
    box=[[-0.5001, -0.006], [-0.4999, 0.006]],
    relative=False, id=1,
).dirichlet(value=[0.0, 0.0])
model.rhs([10, 100])

space = polyfem.space(
    discr_order=1,
    advanced=polyfem.space_advanced(quadrature_order=2, mass_quadrature_order=2),
)

solver = polyfem.solver(
    max_threads=1,
    linear=polyfem.linear(solver="Eigen::SimplicialLDLT"),
    nonlinear=polyfem.nonlinear(
        norm_type="Euclidean", rel_grad_norm_tol=0,
        first_grad_norm_tol=1e-10, grad_norm_tol=1e-8,
    ),
    advanced=polyfem.solver_advanced(characteristic_force_density=1, characteristic_length=1),
)

output_dir = EXAMPLE_DIR / "runs" / "linear_elasticity_compliance"
output_dir.mkdir(parents=True, exist_ok=True)

output = polyfem.output(
    directory=str(output_dir), json="",
    paraview=polyfem.output_paraview(file_name=""),
    advanced=polyfem.output_advanced(save_time_sequence=False),
)

polyfem_config = polyfem.config(model=model, space=space, solver=solver, output=output)


# Differentiable model
n_elements = obj_face_count(MESH_PATH)
lame = torch.empty((n_elements, 2), dtype=torch.float64)
lame[:, 0] = 1.0e5
lame[:, 1] = 5.0e4
lame.requires_grad_(True)

diff_model = diff.model([polyfem_config])
material_opt = diff.material_opt(
    diff_model, lame,
    objective=diff.Objective.COMPLIANCE,
    objective_params={"selection": 1},
)

loss = material_opt()
loss.backward()

summary = {
    "objective": diff.Objective.COMPLIANCE.value,
    "material_parameter": "lame",
    "lame_shape": list(lame.shape),
    "loss": float(loss.detach()),
    "gradient_shape": list(lame.grad.shape),
    "gradient_norm": float(lame.grad.norm()),
    "output_dir": str(output_dir),
}

print(json.dumps(summary, indent=2, sort_keys=True))
