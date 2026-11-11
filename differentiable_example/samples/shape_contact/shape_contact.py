"""Minimize the stress norm plus collision barriers on the initial and deformed meshes.

x holds the undeformed vertex coordinates of the bodies in contact.

Source: differentiability-data/input/shape-contact-opt.json
Run: python -m differentiable_example.samples.shape_contact.shape_contact
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
        mesh=str(DATA_DIR / "cube_dense.msh"),
        transformation=polyfem.transformation(translation=[0, 1.5001], scale=1.0),
        volume_selection=1,
        surface_selection=[
            {"id": 1, "axis": "y", "position": 1.99},
            {"id": 2, "axis": "-y", "position": 0.01},
        ],
    )
    model.mesh(
        mesh=str(DATA_DIR / "cube_dense.msh"),
        transformation=polyfem.transformation(translation=[0, 0.5], scale=1.0),
        volume_selection=2,
        surface_selection=[
            {"id": 1, "axis": "y", "position": 1.99},
            {"id": 2, "axis": "-y", "position": 0.01},
        ],
    )
    config = solver_config(
        model.config(
            materials=polyfem.neo_hookean(E=200, nu=0.3, rho=1),
            boundary_conditions=polyfem.boundary_conditions(
                dirichlet_boundary=[
                    {"id": 1, "value": [-0.1, 0]},
                    {"id": 2, "value": [0, 0]},
                ]
            ),
            space=polyfem.space(
                discr_order=1, advanced=polyfem.space_advanced(quadrature_order=5)
            ),
            contact=polyfem.contact(
                enabled=True, use_convergent_formulation=True, dhat=0.001
            ),
            solver=polyfem.solver(
                linear=polyfem.linear(solver="Eigen::SimplicialLDLT"),
                nonlinear=polyfem.nonlinear(
                    line_search=polyfem.line_search(use_grad_norm_tol=0.001),
                    norm_type="Euclidean",
                    rel_grad_norm_tol=0,
                    first_grad_norm_tol=1e-10,
                    grad_norm_tol=1e-08,
                ),
                contact=polyfem.solver_contact(barrier_stiffness=20),
                advanced=polyfem.solver_advanced(
                    characteristic_force_density=1, characteristic_length=1
                ),
                max_threads=1,
            ),
            output=example_output(output),
        )
    )

    stress = diff.objectives.stress_norm(selection=[])
    rest_barrier = diff.objectives.collision_barrier()
    deformed_barrier = diff.objectives.deformed_collision_barrier()

    objective = diff.objectives.sum(stress, rest_barrier, deformed_barrier)

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
