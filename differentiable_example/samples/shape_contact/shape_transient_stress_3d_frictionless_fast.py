"""Minimize the final stress norm in body 2.

x holds the vertex coordinates on surface 2 of the frictionless 3D model.

Source: differentiability-data/input/shape-transient-stress-3d-frictionless-fast-opt.json
Run: python -m differentiable_example.samples.shape_contact.shape_transient_stress_3d_frictionless_fast
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
        mesh=str(DATA_DIR / "plank.msh"),
        transformation=polyfem.transformation(
            rotation=30, rotation_mode="x", scale=[2.0, 2.0, 2.0]
        ),
        volume_selection=1,
        surface_selection=1,
    )
    model.mesh(
        mesh=str(DATA_DIR / "sphere600.msh"),
        transformation=polyfem.transformation(
            translation=[0, 0.5, -0.2], scale=[0.5, 0.5, 0.5]
        ),
        volume_selection=2,
        surface_selection=2,
    )
    config = solver_config(
        model.config(
            materials=polyfem.neo_hookean(E=100000.0, nu=0.3, rho=1000),
            boundary_conditions=polyfem.boundary_conditions(
                rhs=[0, 9.81, 0], dirichlet_boundary=[{"id": 1, "value": [0, 0, 0]}]
            ),
            space=polyfem.space(
                discr_order=2,
                advanced=polyfem.space_advanced(n_boundary_samples=5, quadrature_order=5),
            ),
            time=polyfem.time(t0=0, tend=0.24, time_steps=12),
            contact=polyfem.contact(
                enabled=True,
                dhat=0.001,
                friction_coefficient=0.0,
                use_convergent_formulation=True,
            ),
            solver=polyfem.solver(
                linear=polyfem.linear(solver="Eigen::SimplicialLDLT"),
                nonlinear=polyfem.nonlinear(
                    line_search=polyfem.line_search(use_grad_norm_tol=0.001)
                ),
                contact=polyfem.solver_contact(barrier_stiffness=10000000000.0),
                max_threads=1,
            ),
            output=example_output(output),
        )
    )

    objective = diff.objectives.stress_norm(power=2, selection=[2]).final()

    obj = diff.shape(config, objective=objective)
    vertices = mesh_vertices(config)

    ids = torch.tensor(diff.model(config).boundary_vertex_ids([2]))

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
