"""Minimize the logarithmic contact force measure on surfaces 2 and 3 at the final step.

x holds boundary vertex coordinates outside surfaces 1 and 2.

Source: differentiability-data/input/shape-contact-force-norm-3d-opt.json
Run: python -m differentiable_example.samples.shape_contact.shape_contact_force_norm_3d
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
        mesh=str(DATA_DIR / "sphere-P1.msh"),
        transformation=polyfem.transformation(translation=[0, 1.81, 0], scale=0.98),
        surface_selection=[
            {"id": 1, "axis": "y", "position": 0.8, "relative": True},
            {"id": 3, "axis": "-y", "position": 0.8, "relative": True},
        ],
        volume_selection=1,
        n_refs=0,
    )
    model.mesh(
        mesh=str(DATA_DIR / "sphere-P1.msh"),
        surface_selection=[
            {"id": 2, "axis": "y", "position": 0.8, "relative": True},
            {"id": 1, "axis": "-y", "position": 0.2, "relative": True},
        ],
        volume_selection=2,
        n_refs=0,
    )
    config = solver_config(
        model.config(
            materials=[
                polyfem.neo_hookean(id=1, E=100000.0, nu=0.3),
                polyfem.neo_hookean(id=2, E=100000.0, nu=0.3),
            ],
            boundary_conditions=polyfem.boundary_conditions(
                rhs=[0, 0, 0],
                dirichlet_boundary=[{"id": 1, "value": [0, 0, 0]}],
                pressure_boundary=[{"id": 3, "value": "-5000*t"}],
            ),
            space=polyfem.space(
                discr_order=2,
                advanced=polyfem.space_advanced(quadrature_order=5, n_boundary_samples=5),
            ),
            time=polyfem.time(
                t0=0, tend=0.7, time_steps=7, integrator={"type": "BDF", "steps": 1}
            ),
            contact=polyfem.contact(
                enabled=True,
                dhat=0.02,
                friction_coefficient=0.0,
                use_gcp_formulation=True,
                use_convergent_formulation=False,
                alpha_t=0.95,
            ),
            solver=polyfem.solver(
                linear=polyfem.linear(solver="Eigen::SimplicialLDLT"),
                contact=polyfem.solver_contact(
                    barrier_stiffness=10000.0, friction_iterations=1
                ),
                nonlinear=polyfem.nonlinear(
                    norm_type="Euclidean",
                    rel_grad_norm_tol=0,
                    first_grad_norm_tol=1e-11,
                    grad_norm_tol=9.999999999999999e-14,
                    line_search=polyfem.line_search(use_grad_norm_tol=9.999999999999998e-08),
                ),
                advanced=polyfem.solver_advanced(
                    characteristic_force_density=1, characteristic_length=1
                ),
                max_threads=1,
            ),
            output=example_output(output),
        )
    )

    objective = diff.objectives.log_contact_force_norm(
        dhat=0.1, surface_selection=[2, 3]
    ).final()

    obj = diff.shape(config, objective=objective)
    vertices = mesh_vertices(config)

    ids = torch.tensor(diff.model(config).boundary_vertex_ids([1, 2], exclude=True))

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
