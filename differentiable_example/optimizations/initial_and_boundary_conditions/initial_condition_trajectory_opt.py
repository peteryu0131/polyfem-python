"""Match the center of body 1 to its reference trajectory.

x is the 2D initial velocity shared by all nodes on body 1.

Source: differentiability-data/optimizations/initial-condition-trajectory-opt
Run: python -m differentiable_example.optimizations.initial_and_boundary_conditions.initial_condition_trajectory_opt
"""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

import polyfem.diff as diff
from polyfempy import polyfempy as backend
import polyfem

from differentiable_example._scene import OPTIMIZATIONS, example_output, solver_config

ASSETS = OPTIMIZATIONS / "initial-condition-trajectory-opt"


def build_scene(output, *, initial_velocity):
    model = polyfem.model()
    model.mesh(
        mesh=str((ASSETS / "../../circle.msh").resolve()),
        transformation=polyfem.transformation(translation=[-1.5, 0.3], scale=0.5),
        volume_selection=1,
        surface_selection=1,
    )
    model.mesh(
        mesh=str((ASSETS / "square.obj").resolve()),
        transformation=polyfem.transformation(scale=[3, 0.02]),
        volume_selection=2,
        surface_selection=2,
    )
    config = model.config(
        materials=polyfem.neo_hookean(E=1000000.0, nu=0.3, rho=1000),
        boundary_conditions=polyfem.boundary_conditions(
            rhs=[0, 9.8], dirichlet_boundary=[{"id": 2, "value": [0, 0]}]
        ),
        initial_conditions=polyfem.initial_conditions(
            velocity=[{"id": 1, "value": initial_velocity}]
        ),
        space=polyfem.space(discr_order=1, advanced=polyfem.space_advanced(quadrature_order=5)),
        time=polyfem.time(tend=0.4, dt=0.02, integrator={"type": "BDF", "steps": 1}),
        contact=polyfem.contact(
            enabled=True, use_convergent_formulation=True, friction_coefficient=0.4
        ),
        solver=polyfem.solver(
            linear=polyfem.linear(solver="Eigen::SimplicialLDLT"),
            nonlinear=polyfem.nonlinear(
                norm_type="Euclidean",
                rel_grad_norm_tol=0,
                first_grad_norm_tol=2e-12,
                grad_norm_tol=2e-10,
                line_search=polyfem.line_search(use_grad_norm_tol=2e-08),
            ),
            contact=polyfem.solver_contact(barrier_stiffness=100000.0),
            advanced=polyfem.solver_advanced(
                characteristic_force_density=1, characteristic_length=1
            ),
            max_threads=1,
        ),
        output=example_output(output),
    )
    return solver_config(config)


def problem(config, target):
    obj = diff.initial(
        config,
        objective=diff.objectives.transient_integral(
            static_objective=diff.objectives.create(
                "center-target", target_state=1, volume_selection=[1]
            ),
            integral_type="uniform",
            weight=10,
        ),
        target=target,
    )
    # Query a unit horizontal velocity on body 1 in the solver's DOF order.
    mask_config = deepcopy(config)
    mask_config["initial_conditions"]["velocity"] = [{"id": 1, "value": [1.0, 0.0]}]
    session = backend.DifferentiableSession()
    session.set_settings(mask_config)
    initial = torch.as_tensor(
        session.initial_condition_parameters(), dtype=torch.float64
    )
    moving = initial.reshape(2, -1)[1].reshape(-1, 2)[:, :1]
    assert moving.numel() * 2 == 146

    def loss(x):
        velocity = (moving * x).flatten()
        return obj(
            initial_condition=torch.stack((torch.zeros_like(velocity), velocity), dim=1)
        )

    return [4, 0], loss


if __name__ == "__main__":
    with TemporaryDirectory() as directory:
        output = Path(directory)
        config = build_scene(output=output / "forward0", initial_velocity=[4, 0])
        target = build_scene(output=output / "forward1", initial_velocity=[3, -3])
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
