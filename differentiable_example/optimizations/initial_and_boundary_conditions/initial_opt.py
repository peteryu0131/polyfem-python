"""Match surface 1 to its reference position at the final time.

x is the 2D initial velocity of body 1. We chose this loss and control for the supplied forward models.

Source: differentiability-data/optimizations/initial-opt
Run: python -m differentiable_example.optimizations.initial_and_boundary_conditions.initial_opt
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import torch

import polyfem.diff as diff
from polyfempy import polyfempy as backend
import polyfem

from differentiable_example._scene import OPTIMIZATIONS, example_output, solver_config

ASSETS = OPTIMIZATIONS / "initial-opt"


def build_scene(output, *, initial_velocity):
    model = polyfem.model()
    model.obstacle_mesh(
        mesh=str((ASSETS / "wave.msh").resolve()),
        transformation=polyfem.transformation(scale=[2, 1]),
        volume_selection=3,
        surface_selection=3,
    )
    model.mesh(
        mesh=str((ASSETS / "puzzle.obj").resolve()),
        transformation=polyfem.transformation(translation=[-1, 0.2], scale=0.001),
        volume_selection=1,
        surface_selection=1,
    )
    config = model.config(
        materials=[
            polyfem.neo_hookean(id=1, E=1000000.0, nu=0.3, rho=1000),
            polyfem.neo_hookean(id=3, E=1000000.0, nu=0.3, rho=1000),
        ],
        boundary_conditions=polyfem.boundary_conditions(
            rhs=[0, 9.8], dirichlet_boundary=[{"id": 3, "value": [0, 0]}]
        ),
        initial_conditions=polyfem.initial_conditions(
            velocity=[{"id": 1, "value": initial_velocity}]
        ),
        space=polyfem.space(discr_order=1, advanced=polyfem.space_advanced(quadrature_order=4)),
        time=polyfem.time(tend=0.6, dt=0.02),
        contact=polyfem.contact(
            enabled=True, use_convergent_formulation=True, friction_coefficient=0.5
        ),
        solver=polyfem.solver(
            contact=polyfem.solver_contact(barrier_stiffness=100000.0),
            advanced=polyfem.solver_advanced(
                characteristic_force_density=1, characteristic_length=1
            ),
            nonlinear=polyfem.nonlinear(
                norm_type="Euclidean",
                rel_grad_norm_tol=0,
                first_grad_norm_tol=2e-12,
                grad_norm_tol=2e-10,
                line_search=polyfem.line_search(use_grad_norm_tol=2e-08),
            ),
            max_threads=1,
            linear=polyfem.linear(solver="Eigen::SimplicialLDLT"),
        ),
        output=example_output(output),
    )
    return solver_config(config)


def problem(config, target):
    initial = config["initial_conditions"]["velocity"][0]["value"]
    obj = diff.initial(
        config,
        target=target,
        objective=diff.objectives.target(target_state=1, selection=1).final(),
    )
    session = backend.DifferentiableSession()
    session.set_settings(config)
    velocity = (
        torch.as_tensor(session.initial_condition_parameters())
        .reshape(2, -1)[1]
        .reshape(-1, 2)
    )
    mask = velocity[:, :1] / initial[0]

    def loss(x):
        values = (mask * x).flatten()
        return obj(
            initial_condition=torch.stack((torch.zeros_like(values), values), dim=1)
        )

    return initial, loss


if __name__ == "__main__":
    with TemporaryDirectory() as directory:
        output = Path(directory)
        config = build_scene(output=output / "forward0", initial_velocity=[0.5, 0])
        target = build_scene(output=output / "forward1", initial_velocity=[2, 1])
        initial, loss = problem(config, target)
        x = torch.tensor(initial, dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.Adam([x], lr=1e-4)
        optimizer.zero_grad()
        before = loss(x)
        before.backward()
        optimizer.step()
        with torch.no_grad():
            after = loss(x)
        print(f"Loss after one update: {before.item():.10g} -> {after.item():.10g}")
