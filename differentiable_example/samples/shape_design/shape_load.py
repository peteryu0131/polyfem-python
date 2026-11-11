"""Minimize the final stress norm in body 1 under the prescribed Neumann load.

x is a single 2D offset shared by all vertices on boundary 2.

Source: differentiability-data/input/shape-neumann-nodes-opt.json
Run: python -m differentiable_example.samples.shape_design.shape_load
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
        transformation=polyfem.transformation(translation=[0, 0.56]),
        surface_selection=[
            {"id": 2, "box": [[-0.6, 0.8], [0.6, 1.2]], "relative": False}
        ],
        volume_selection=1,
        n_refs=0,
    )
    model.mesh(
        mesh=str(DATA_DIR / "square.obj"),
        transformation=polyfem.transformation(scale=[3, 0.1]),
        surface_selection=1,
        volume_selection=2,
    )
    config = solver_config(
        model.config(
            materials=[
                polyfem.neo_hookean(id=1, E=100000.0, nu=0.3),
                polyfem.neo_hookean(id=2, E=100000.0, nu=0.3),
            ],
            boundary_conditions=polyfem.boundary_conditions(
                rhs=[0, 9.81],
                dirichlet_boundary=[{"id": 1, "value": [0.0, 0.0]}],
                neumann_boundary=[{"id": 2, "value": ["5.0*t", -10.0]}],
            ),
            space=polyfem.space(
                discr_order=1,
                advanced=polyfem.space_advanced(quadrature_order=5, n_boundary_samples=5),
            ),
            time=polyfem.time(t0=0, tend=0.25, time_steps=10),
            contact=polyfem.contact(
                enabled=True, use_convergent_formulation=True, dhat=0.001
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
                contact=polyfem.solver_contact(barrier_stiffness=10000.0),
                advanced=polyfem.solver_advanced(
                    characteristic_force_density=1, characteristic_length=1
                ),
                max_threads=1,
            ),
            output=example_output(output),
        )
    )

    vertices = mesh_vertices(config)
    movable_ids = torch.tensor(diff.model(config).boundary_vertex_ids([2]))
    objective = diff.objectives.stress_norm(selection=1).final()
    obj = diff.shape(config, objective=objective)
    x = torch.zeros(2, dtype=torch.float64, requires_grad=True)

    def loss(x):
        moved_vertices = vertices[movable_ids] + x
        mesh = vertices.index_copy(0, movable_ids, moved_vertices)
        return obj(V=mesh)

    optimizer = torch.optim.Adam([x], lr=1e-5)
    optimizer.zero_grad()
    value = loss(x)
    value.backward()
    optimizer.step()

    with torch.no_grad():
        after = loss(x)
    print(f"Loss after one update: {value.item():.10g} -> {after.item():.10g}")
