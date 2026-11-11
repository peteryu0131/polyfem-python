"""Minimize the stress norm accumulated over the simulation.

x holds the undeformed vertex coordinates of the frictional contact model.

Source: differentiability-data/input/shape-transient-friction-opt.json
Run: python -m differentiable_example.samples.shape_contact.shape_transient_friction
"""

import torch

from differentiable_example._scene import (
    DATA_DIR,
    example_output,
    mesh_vertices,
    simulation_directory,
    solver_config,
)
import polyfem.diff as diff
import polyfem

with simulation_directory() as output:
    model = polyfem.model()
    model.mesh(
        mesh=str(DATA_DIR / "square.obj"),
        transformation=polyfem.transformation(rotation=-30, scale=[5, 0.02]),
        volume_selection=1,
        surface_selection=1,
    )
    model.mesh(
        mesh=str(DATA_DIR / "circle.msh"),
        transformation=polyfem.transformation(
            translation=[0.26, 0.5], rotation=-30, scale=1.0
        ),
        volume_selection=2,
        surface_selection=2,
    )
    config = solver_config(
        model.config(
            materials=polyfem.neo_hookean(E=1000000.0, nu=0.48, rho=1000, phi=10, psi=10),
            boundary_conditions=polyfem.boundary_conditions(
                rhs=[0, 9.81], dirichlet_boundary=[{"id": 1, "value": [0, 0]}]
            ),
            space=polyfem.space(
                discr_order=2,
                advanced=polyfem.space_advanced(n_boundary_samples=5, quadrature_order=5),
            ),
            time=polyfem.time(t0=0, tend=0.25, time_steps=10),
            contact=polyfem.contact(
                enabled=True,
                use_convergent_formulation=True,
                dhat=0.001,
                friction_coefficient=0.2,
            ),
            solver=polyfem.solver(
                linear=polyfem.linear(solver="Eigen::SimplicialLDLT"),
                nonlinear=polyfem.nonlinear(
                    line_search=polyfem.line_search(use_grad_norm_tol=2.5e-05),
                    norm_type="Euclidean",
                    rel_grad_norm_tol=0,
                    first_grad_norm_tol=2.5000000000000003e-12,
                    grad_norm_tol=2.5e-10,
                ),
                contact=polyfem.solver_contact(barrier_stiffness=100000.0),
                advanced=polyfem.solver_advanced(
                    characteristic_force_density=1, characteristic_length=1
                ),
                max_threads=1,
            ),
            output=example_output(output),
        )
    )

    objective = diff.objectives.transient_integral(
        static_objective=diff.objectives.stress_norm(selection=[]),
        integral_type="uniform",
    )

    obj = diff.shape(config, objective=objective)
    vertices = mesh_vertices(config)

    x = vertices.clone().requires_grad_()

    def loss(x):
        return obj(V=x)

    optimizer = torch.optim.Adam([x], lr=1e-5)
    optimizer.zero_grad()
    before = loss(x)
    before.backward()
    optimizer.step()

    with torch.no_grad():
        after = loss(x)
    print(f"Loss after one update: {before.item():.10g} -> {after.item():.10g}")
