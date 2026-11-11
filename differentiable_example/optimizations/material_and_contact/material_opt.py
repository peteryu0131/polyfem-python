"""Fit the simulated surface trajectory to the reference simulation.

x contains log(lambda) and log(mu), shared by every element.

Source: differentiability-data/optimizations/material-opt
Run: python -m differentiable_example.optimizations.material_and_contact.material_opt
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import torch

import polyfem.diff as diff
import polyfem

from differentiable_example._scene import OPTIMIZATIONS, bdf1_option, example_output, solver_config

ASSETS = OPTIMIZATIONS / "material-opt"


def build_scene(output, *, young_modulus, poisson_ratio, bdf_steps=2):
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
        materials=polyfem.neo_hookean(E=young_modulus, nu=poisson_ratio, rho=100),
        boundary_conditions=polyfem.boundary_conditions(
            rhs=[0, 9.8], dirichlet_boundary=[{"id": 3, "value": [0, 0]}]
        ),
        space=polyfem.space(discr_order=1, advanced=polyfem.space_advanced(quadrature_order=5)),
        time=polyfem.time(tend=0.4, dt=0.04, integrator={"type": "BDF", "steps": bdf_steps}),
        contact=polyfem.contact(
            enabled=True, friction_coefficient=1, use_convergent_formulation=True
        ),
        solver=polyfem.solver(
            linear=polyfem.linear(solver="Eigen::SimplicialLDLT"),
            nonlinear=polyfem.nonlinear(
                max_iterations=200,
                norm_type="Euclidean",
                rel_grad_norm_tol=0,
                first_grad_norm_tol=4e-12,
                grad_norm_tol=4e-05,
                line_search=polyfem.line_search(use_grad_norm_tol=4e-08),
            ),
            contact=polyfem.solver_contact(barrier_stiffness=10000.0),
            advanced=polyfem.solver_advanced(
                characteristic_force_density=1, characteristic_length=1
            ),
            max_threads=1,
        ),
        output=example_output(output),
    )
    return solver_config(config)


def build_loss(config, target, objective):
    model = diff.model(config)
    meta = model.metadata()
    assert len(set(meta.element_body_ids)) == 1

    def loss(x):
        lame = x.exp().expand(meta.n_elements, 2).contiguous()
        return diff.MaterialOpt.apply(model, lame, objective=objective, target=target)

    return loss


if __name__ == "__main__":
    with TemporaryDirectory() as directory:
        output = Path(directory)
        bdf_steps = 1 if bdf1_option() else 2
        config = build_scene(
            output=output / "forward0",
            young_modulus=10000.0,
            poisson_ratio=0.8,
            bdf_steps=bdf_steps,
        )
        target = build_scene(
            output=output / "forward1",
            young_modulus=100000.0,
            poisson_ratio=0.5,
            bdf_steps=bdf_steps,
        )
        loss = build_loss(
            config,
            target,
            diff.objectives.transient_integral(
                static_objective=diff.objectives.target(target_state=1, weight=100),
                integral_type="uniform",
            ),
        )
        initial = [10.0088, 7.92941]
        x = torch.tensor(initial, dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.Adam([x], lr=1e-4)
        optimizer.zero_grad()
        before = loss(x)
        before.backward()
        optimizer.step()
        with torch.no_grad():
            after = loss(x)
        print(f"Loss after one update: {before.item():.10g} -> {after.item():.10g}")
