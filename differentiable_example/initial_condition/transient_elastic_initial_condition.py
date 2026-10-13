#!/usr/bin/env python3
"""Uniform initial velocity and angle optimization at a fixed final time.

This targets a horizontal position, not landing range or a 45-degree launch.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from polyfempy.generated_api import generated_api as polyfem
from polyfempy import differentiable_api as diff


EXAMPLE_DIR = Path(__file__).resolve().parent
ROOT = EXAMPLE_DIR.parents[1]
MESH_PATH = ROOT / "polyfem-data" / "contact" / "meshes" / "2D" / "simple" / "square.obj"


# Forward model
model = polyfem.model()

mesh = polyfem.mesh(mesh=str(MESH_PATH))
material = polyfem.neo_hookean(E=2.55e7, nu=0.48, rho=1700)
body = polyfem.body(model=model)
body.mesh(mesh)
body.material(material)
body.rhs(value=[0, 0])

time = polyfem.time(polyfem.object1(dt=0.04, tend=0.04, integrator="BDF"))
contact = polyfem.contact(enabled=False)
space = polyfem.space(discr_order=1)

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

polyfem_config = polyfem.config(model=model, time=time, space=space, contact=contact, solver=solver, output=output)

objective = diff.objectives.position(dim=0, selection=body).final()


# Differentiable model
diff_model = diff.model([polyfem_config])
metadata = diff_model.metadata()
dof_count = metadata.n_dofs

target_x = 0.02
speed = 1.0


def loss(velocity: torch.Tensor) -> torch.Tensor:
    """R^2 -> R: squared error in final horizontal mean position."""
    initial_condition = metadata.uniform_initial_condition(velocity)
    initial_opt = diff.initial_condition_opt(diff_model, initial_condition, objective=objective)
    # PositionForm integrates x + displacement over reference area.
    # This centered square has area 1, so the integral equals mean position.
    final_x = initial_opt()
    return (final_x - target_x) ** 2


def angle_loss(theta: torch.Tensor) -> torch.Tensor:
    """R -> R: fixed speed, one launch angle in radians."""
    velocity = speed * torch.stack((torch.cos(theta), torch.sin(theta)))
    return loss(velocity)


def main():
    velocity = torch.tensor([0.8, 0.6], dtype=torch.float64, requires_grad=True)
    velocity_loss = loss(velocity)
    velocity_loss.backward()

    theta = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    value = angle_loss(theta)
    value.backward()
    initial_loss = float(value.detach())
    initial_gradient = float(theta.grad)

    # Re-evaluate angle -> velocity -> simulation on every iteration.
    result = (
        diff.optimization(lambda: angle_loss(theta), parameters=[theta])
        .optimizer(torch.optim.Adam, lr=0.05)
        .steps(8)
        .optimize(record=lambda step, value: {"theta": float(theta.detach())})
    )
    history = result.history

    # Equivalent full loop: replace the result/history block above, then rerun.
    # optimizer = torch.optim.Adam([theta], lr=0.05)
    # history = []
    # for step in range(8):
    #     optimizer.zero_grad()
    #     value = angle_loss(theta)
    #     value.backward()
    #     history.append({
    #         "step": step, "theta": float(theta.detach()), "loss": float(value.detach()),
    #         "gradient_norm": float(theta.grad.norm()),
    #     })
    #     optimizer.step()
    final_loss = float(angle_loss(theta).detach())

    summary = {
        "objective": "fixed_time_horizontal_position_error",
        "parameter": "launch_angle_radians",
        "velocity_gradient": velocity.grad.tolist(),
        "velocity_gradient_shape": list(velocity.grad.shape),
        "theta_initial": 0.5,
        "theta_final": float(theta.detach()),
        "theta_grad": initial_gradient,
        "gradient_shape": list(theta.grad.shape),
        "loss": initial_loss,
        "final_loss": final_loss,
        "history": history,
        "dof_count": dof_count,
        "initial_condition_shape": list(metadata.uniform_initial_condition(velocity).shape),
        "output_dir": str(output_dir),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
