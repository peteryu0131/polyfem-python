#!/usr/bin/env python3
"""PyTorch chain-rule InitialConditionOpt example."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from polyfempy.generated_api import generated_api as polyfem
from polyfempy import differentiable_api as diff
from polyfempy import polyfempy as backend


EXAMPLE_DIR = Path(__file__).resolve().parent
ROOT = EXAMPLE_DIR.parents[1]
MESH_PATH = ROOT / "polyfem-data" / "contact" / "meshes" / "2D" / "simple" / "square.obj"


def initial_condition_dof_count(polyfem_config: dict) -> int:
    session = backend.DifferentiableSession()
    session.set_settings(polyfem_config)
    return int(session.initial_condition_dof_count())


# Forward model
model = polyfem.model()

mesh = polyfem.mesh(mesh=str(MESH_PATH))
material = polyfem.neo_hookean(E=2.55e7, nu=0.48, rho=1700)
body = polyfem.body(model=model)
body.mesh(mesh)
body.material(material)
model.rhs([0, 0])

time = polyfem.time(polyfem.object1(dt=0.04, tend=0.04, integrator="BDF"))
contact = polyfem.contact(enabled=True, dhat=0.001)

solver = polyfem.solver(
    max_threads=1,
    linear=polyfem.linear(solver="Eigen::SimplicialLDLT"),
    nonlinear=polyfem.nonlinear(
        solver="Newton", iterations_per_strategy=1,
        line_search=polyfem.line_search(method="RobustArmijo"),
    ),
    advanced=polyfem.solver_advanced(lump_mass_matrix=True),
)

output_dir = EXAMPLE_DIR / "runs" / "transient_elastic_initial_condition"
output_dir.mkdir(parents=True, exist_ok=True)

output = polyfem.output(
    directory=str(output_dir), json="",
    paraview=polyfem.output_paraview(file_name=""),
    advanced=polyfem.output_advanced(save_solve_sequence_debug=False, save_time_sequence=False),
)

polyfem_config = polyfem.config(model=model, time=time, contact=contact, solver=solver, output=output)

objective = {
    "type": "transient_integral",
    "state": 0,
    "integral_type": "final",
    "steps": [],
    "weight": 1.0,
    "print_energy": "",
    "static_objective": {
        "type": "stress_norm",
        "state": 0,
        "volume_selection": [],
        "power": 2,
        "weight": 1.0,
        "print_energy": "",
    },
}


# Differentiable model
diff_model = diff.model([polyfem_config])
dof_count = initial_condition_dof_count(diff_model.as_dicts()[0])

initial_solution = torch.zeros(dof_count, dtype=torch.float64)
base_velocity = torch.linspace(0.0, 1.0e-2, dof_count, dtype=torch.float64)

speed = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
initial_velocity = base_velocity * speed
initial_condition = torch.stack((initial_solution, initial_velocity), dim=1)

initial_opt = diff.initial_condition_opt(diff_model, initial_condition, objective=objective)

loss = initial_opt()
loss.backward()

summary = {
    "objective": diff.Objective.STRESS_NORM.value,
    "parameter": "initial_velocity_speed",
    "mapping": "initial_velocity = base_velocity * speed",
    "speed_value": float(speed.detach()),
    "speed_grad": float(speed.grad.detach()),
    "loss": float(loss.detach()),
    "dof_count": dof_count,
    "initial_condition_shape": list(initial_condition.shape),
    "initial_condition_requires_grad": bool(initial_condition.requires_grad),
    "output_dir": str(output_dir),
}

print(json.dumps(summary, indent=2, sort_keys=True))
