"""Minimize the integral of the deformed y-coordinate over the body.

x holds the undeformed 2D mesh vertex coordinates.

Source: differentiability-data/input/linear_elasticity-surface-opt.json
Run: python -m differentiable_example.samples.shape_design.linear_elasticity_surface
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
    model.geometry_mesh(
        mesh=str(DATA_DIR / "cube_dense.msh"),
        surface_selection=[{"threshold": 0.0001}],
        transformation=polyfem.transformation(translation=[0.5, 0.5]),
    )
    config = solver_config(
        model.config(
            materials=polyfem.linear_elasticity(lambda_=17284000.0, mu=7407410.0),
            boundary_conditions=polyfem.boundary_conditions(
                neumann_boundary=[{"id": 3, "value": [5, 3]}],
                dirichlet_boundary=[{"id": 1, "value": [0, 0]}],
            ),
            space=polyfem.space(
                discr_order=1,
                advanced=polyfem.space_advanced(n_boundary_samples=5, quadrature_order=5),
            ),
            solver=polyfem.solver(
                linear=polyfem.linear(solver="Eigen::SimplicialLDLT"),
                advanced=polyfem.solver_advanced(
                    characteristic_force_density=1, characteristic_length=1
                ),
                nonlinear=polyfem.nonlinear(
                    norm_type="Euclidean",
                    rel_grad_norm_tol=0,
                    first_grad_norm_tol=1e-10,
                    grad_norm_tol=1e-08,
                ),
                max_threads=1,
            ),
            output=example_output(output),
        )
    )

    objective = diff.objectives.position(dim=1)

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
