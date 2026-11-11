"""Minimize the quadratic contact force measure on those surfaces at step 30.

x holds the vertex coordinates on surfaces 3 and 4.

Source: differentiability-data/input/shape-contact-force-norm-opt.json
Run: python -m differentiable_example.samples.shape_contact.shape_contact_force_norm
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
        transformation=polyfem.transformation(translation=[0.4, 1.02]),
        surface_selection=[
            {"id": 1, "axis": "y", "position": 1.5, "relative": False},
            {"id": 3, "axis": "-y", "position": 1.0, "relative": False},
        ],
        volume_selection=1,
        n_refs=2,
    )
    model.mesh(
        mesh=str(DATA_DIR / "square.obj"),
        surface_selection=[
            {"id": 2, "axis": "-y", "position": -0.4, "relative": False},
            {"id": 4, "axis": "y", "position": 0.0, "relative": False},
        ],
        volume_selection=2,
        n_refs=2,
    )
    config = solver_config(
        model.config(
            materials=[
                polyfem.neo_hookean(id=1, E=100000.0, nu=0.3),
                polyfem.neo_hookean(id=2, E=100000.0, nu=0.3),
            ],
            boundary_conditions=polyfem.boundary_conditions(
                rhs=[0, 9.81],
                dirichlet_boundary=[
                    {"id": 1, "value": ["-0.2 * t", "-0.1 * t"]},
                    {"id": 2, "value": [0.0, 0.0]},
                ],
            ),
            space=polyfem.space(
                discr_order=1,
                advanced=polyfem.space_advanced(quadrature_order=5, n_boundary_samples=5),
            ),
            time=polyfem.time(
                t0=0, tend=1.0, time_steps=30, integrator={"type": "BDF", "steps": 1}
            ),
            contact=polyfem.contact(
                enabled=True,
                use_convergent_formulation=True,
                dhat=0.01,
                friction_coefficient=0.4,
            ),
            solver=polyfem.solver(
                linear=polyfem.linear(solver="Eigen::SimplicialLDLT"),
                contact=polyfem.solver_contact(
                    barrier_stiffness=10000.0, friction_iterations=1
                ),
                nonlinear=polyfem.nonlinear(
                    norm_type="Euclidean",
                    rel_grad_norm_tol=0,
                    first_grad_norm_tol=3.3333333333333335e-12,
                    grad_norm_tol=3.3333333333333335e-11,
                    line_search=polyfem.line_search(
                        use_grad_norm_tol=3.3333333333333334e-08
                    ),
                ),
                advanced=polyfem.solver_advanced(
                    characteristic_force_density=1, characteristic_length=1
                ),
                max_threads=1,
            ),
            output=example_output(output),
        )
    )

    objective = diff.objectives.transient_integral(
        static_objective=diff.objectives.quadratic_contact_force_norm(
            surface_selection=[3, 4], dhat=0.1
        ),
        integral_type="steps",
        steps=[30],
    )

    obj = diff.shape(config, objective=objective)
    vertices = mesh_vertices(config)

    ids = torch.tensor(diff.model(config).boundary_vertex_ids([3, 4]))

    x = vertices[ids].clone().requires_grad_()

    def loss(x):
        return obj(V=vertices.index_copy(0, ids, x))

    optimizer = torch.optim.Adam([x], lr=1e-5)
    optimizer.zero_grad()
    before = loss(x)
    before.backward()
    optimizer.step()

    with torch.no_grad():
        after = loss(x)
    print(f"Loss after one update: {before.item():.10g} -> {after.item():.10g}")
