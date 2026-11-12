"""Fit surface 2 to the target curve at the final step.

x holds the undeformed vertex coordinates of the frictional contact model.

Source: differentiability-data/input/shape-transient-friction-sdf-opt.json
Run: python -m differentiable_example.samples.shape_contact.shape_transient_friction_sdf
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

with simulation_directory(__file__) as output:
    model = polyfem.model()
    model.mesh(
        mesh=str(DATA_DIR / "square.obj"),
        transformation=polyfem.transformation(
            translation=[0, 0], rotation=-30, scale=[5, 0.02]
        ),
        volume_selection=1,
        surface_selection=1,
        n_refs=0,
        advanced={"normalize_mesh": False},
    )
    model.mesh(
        mesh=str(DATA_DIR / "circle.msh"),
        transformation=polyfem.transformation(
            translation=[0.26, 0.5], rotation=-30, scale=1.0
        ),
        volume_selection=2,
        surface_selection=[
            {"id": 2, "relative": False, "box": [[0.11, 0.55], [0.8, 1]]}
        ],
        n_refs=0,
        advanced={"normalize_mesh": False},
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
                contact=polyfem.solver_contact(barrier_stiffness=100000.0),
                advanced=polyfem.solver_advanced(
                    characteristic_force_density=1, characteristic_length=1
                ),
                nonlinear=polyfem.nonlinear(
                    norm_type="Euclidean",
                    rel_grad_norm_tol=0,
                    first_grad_norm_tol=2.5000000000000003e-12,
                    grad_norm_tol=2.5e-10,
                    line_search=polyfem.line_search(use_grad_norm_tol=2.5e-08),
                ),
                max_threads=1,
            ),
            output=example_output(output),
        )
    )

    objective = diff.objectives.sdf_target(
        surface_selection=[2],
        control_points=[[1, 0.4], [0.66666667, 0.73333333], [0.43333333, 1], [0.1, 1]],
        knots=[0, 0, 0, 0, 1, 1, 1, 1],
        delta=0.05,
    ).final()

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
