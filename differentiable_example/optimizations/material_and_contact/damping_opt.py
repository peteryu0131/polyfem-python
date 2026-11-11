"""Fit the simulated surface trajectory to the reference simulation.

x contains the two damping coefficients, psi and phi.

Source: differentiability-data/optimizations/damping-opt
Run: python -m differentiable_example.optimizations.material_and_contact.damping_opt
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import torch

import polyfem.diff as diff
import polyfem

from differentiable_example._scene import OPTIMIZATIONS, bdf1_option, example_output, solver_config

ASSETS = OPTIMIZATIONS / "damping-opt"


def build_scene(output, *, phi, psi, bdf_steps=2):
    model = polyfem.model()
    model.obstacle_mesh(
        mesh=str((ASSETS / "wave.msh").resolve()),
        transformation=polyfem.transformation(rotation=30, scale=[6, 3]),
        volume_selection=3,
        surface_selection=3,
    )
    model.mesh(
        mesh=str((ASSETS / "micro.msh").resolve()),
        transformation=polyfem.transformation(translation=[1, 1], rotation=30, scale=0.005),
        volume_selection=1,
        surface_selection=1,
    )
    config = model.config(
        materials=polyfem.neo_hookean(E=100000.0, nu=0.5, rho=100, phi=phi, psi=psi),
        boundary_conditions=polyfem.boundary_conditions(
            rhs=[0, 9.8], dirichlet_boundary=[{"id": 3, "value": [0, 0]}]
        ),
        space=polyfem.space(discr_order=1, advanced=polyfem.space_advanced(quadrature_order=5)),
        time=polyfem.time(tend=0.4, dt=0.04, integrator={"type": "BDF", "steps": bdf_steps}),
        contact=polyfem.contact(
            enabled=True, use_convergent_formulation=True, friction_coefficient=1
        ),
        solver=polyfem.solver(
            linear=polyfem.linear(solver="Eigen::SimplicialLDLT"),
            contact=polyfem.solver_contact(barrier_stiffness=10000.0),
            advanced=polyfem.solver_advanced(
                characteristic_force_density=1, characteristic_length=1
            ),
            nonlinear=polyfem.nonlinear(
                norm_type="Euclidean",
                rel_grad_norm_tol=0,
                first_grad_norm_tol=4e-12,
                grad_norm_tol=4e-10,
                line_search=polyfem.line_search(use_grad_norm_tol=4e-08),
            ),
            max_threads=1,
        ),
        output=example_output(output),
    )
    return solver_config(config)


def build_problem(config, target, objective):
    obj = diff.damping(config, target=target, objective=objective)

    def loss(x):
        return obj(psi=x[0], phi=x[1])

    return [config["materials"]["psi"], config["materials"]["phi"]], loss


if __name__ == "__main__":
    with TemporaryDirectory() as directory:
        output = Path(directory)
        bdf_steps = 1 if bdf1_option() else 2
        config = build_scene(
            output=output / "forward0", phi=0.01, psi=0.01, bdf_steps=bdf_steps
        )
        target = build_scene(
            output=output / "forward1", phi=1, psi=0.5, bdf_steps=bdf_steps
        )
        initial, loss = build_problem(
            config,
            target,
            diff.objectives.transient_integral(
                static_objective=diff.objectives.target(target_state=1),
                integral_type="uniform",
            ),
        )
        x = torch.tensor(initial, dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.Adam([x], lr=1e-4)
        optimizer.zero_grad()
        before = loss(x)
        before.backward()
        optimizer.step()
        with torch.no_grad():
            after = loss(x)
        print(f"Loss after one update: {before.item():.10g} -> {after.item():.10g}")
