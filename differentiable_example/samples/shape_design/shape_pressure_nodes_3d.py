"""Under the prescribed pressure, fit surface 5 to the target surface at the final step.

x holds boundary vertex coordinates outside surfaces 1, 5 and 3.

Source: differentiability-data/input/shape-pressure-nodes-3d-opt.json
Run: python -m differentiable_example.samples.shape_design.shape_pressure_nodes_3d
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
        mesh=str(DATA_DIR / "mat20x20.msh"),
        surface_selection=[
            {"id": 2, "box": [[0.01, -5, 0.01], [0.99, 0, 0.99]], "relative": True},
            {"id": 3, "axis": "-x", "position": 0.01, "relative": True},
            {"id": 3, "axis": "x", "position": 0.99, "relative": True},
            {"id": 3, "axis": "-z", "position": 0.01, "relative": True},
            {"id": 3, "axis": "z", "position": 0.99, "relative": True},
            {"id": 5, "box": [[0.01, 0, 0.01], [0.99, 5, 0.99]], "relative": True},
        ],
        volume_selection=1,
        n_refs=0,
    )
    model.mesh(
        mesh=str(DATA_DIR / "cube.msh"),
        transformation=polyfem.transformation(translation=[0, 0.06, 0], scale=0.1),
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
                rhs=[0, 9.81, 0],
                dirichlet_boundary=[
                    {"id": 1, "value": [0.0, 0.0, 0.0]},
                    {"id": 3, "value": [0.0, 0.0, 0.0]},
                ],
                pressure_boundary=[{"id": 2, "value": "-20.0 * t"}],
            ),
            space=polyfem.space(
                discr_order=1,
                advanced=polyfem.space_advanced(quadrature_order=5, n_boundary_samples=5),
            ),
            time=polyfem.time(
                t0=0, tend=0.25, time_steps=10, integrator={"type": "BDF", "steps": 1}
            ),
            contact=polyfem.contact(
                enabled=True, use_convergent_formulation=True, dhat=0.001
            ),
            solver=polyfem.solver(
                linear=polyfem.linear(solver="Eigen::SimplicialLDLT"),
                contact=polyfem.solver_contact(barrier_stiffness=10000.0),
                nonlinear=polyfem.nonlinear(
                    line_search=polyfem.line_search(use_grad_norm_tol=2.5e-09),
                    norm_type="Euclidean",
                    rel_grad_norm_tol=0,
                    first_grad_norm_tol=2.5000000000000003e-12,
                    grad_norm_tol=2.5e-10,
                ),
                advanced=polyfem.solver_advanced(
                    characteristic_force_density=1, characteristic_length=1
                ),
                max_threads=1,
            ),
            output=example_output(output),
        )
    )

    objective = diff.objectives.sdf_target(
        surface_selection=[5],
        delta=0.05,
        control_points_grid=[
            [-0.5, 0.0065, -0.5],
            [-0.16666666666666669, 0.0065, -0.5],
            [0.16666666666666663, 0.0065, -0.5],
            [0.5, 0.0065, -0.5],
            [-0.5, 0.0065, -0.16666666666666669],
            [-0.16666666666666669, 0.4065, -0.16666666666666669],
            [0.16666666666666663, 0.4065, -0.16666666666666669],
            [0.5, 0.0065, -0.16666666666666669],
            [-0.5, 0.0065, 0.16666666666666663],
            [-0.16666666666666669, 0.4065, 0.16666666666666663],
            [0.16666666666666663, 0.4065, 0.16666666666666663],
            [0.5, 0.0065, 0.16666666666666663],
            [-0.5, 0.0065, 0.5],
            [-0.16666666666666669, 0.0065, 0.5],
            [0.16666666666666663, 0.0065, 0.5],
            [0.5, 0.0065, 0.5],
        ],
        knots_u=[0, 0, 0, 0, 1, 1, 1, 1],
        knots_v=[0, 0, 0, 0, 1, 1, 1, 1],
    ).final()

    obj = diff.shape(config, objective=objective)
    vertices = mesh_vertices(config)

    ids = torch.tensor(diff.model(config).boundary_vertex_ids([1, 5, 3], exclude=True))

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
