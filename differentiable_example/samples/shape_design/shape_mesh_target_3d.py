"""Fit surface 6 to both a target mesh and a spline surface.

x holds the boundary vertex coordinates outside surfaces 1 through 4.

Source: differentiability-data/input/3d-shape-mesh-target-opt.json
Run: python -m differentiable_example.samples.shape_design.shape_mesh_target_3d
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
        mesh=str(DATA_DIR / "mat20x20.msh"),
        transformation=polyfem.transformation(scale=1),
        volume_selection=1,
        surface_selection=[
            {"id": 1, "axis": "-x", "position": 0.01, "relative": True},
            {"id": 2, "axis": "x", "position": 0.99, "relative": True},
            {"id": 3, "axis": "-z", "position": 0.01, "relative": True},
            {"id": 4, "axis": "z", "position": 0.99, "relative": True},
            {"id": 5, "box": [[0.01, 0, 0.01], [0.99, 0.5, 0.99]], "relative": True},
            {"id": 6, "box": [[0.01, 0.5, 0.01], [0.99, 1, 0.99]], "relative": True},
        ],
        n_refs=0,
        advanced={"normalize_mesh": False},
    )
    config = solver_config(
        model.config(
            materials=polyfem.neo_hookean(E=1000000.0, nu=0.48, rho=1000, phi=10, psi=10),
            boundary_conditions=polyfem.boundary_conditions(
                rhs=[0, 9.81, 0],
                dirichlet_boundary=[
                    {"id": 1, "value": [0, 0, 0]},
                    {"id": 2, "value": [0, 0, 0]},
                    {"id": 3, "value": [0, 0, 0]},
                    {"id": 4, "value": [0, 0, 0]},
                ],
                pressure_boundary=[{"id": 5, "value": -500}],
            ),
            space=polyfem.space(
                discr_order=2,
                advanced=polyfem.space_advanced(n_boundary_samples=5, quadrature_order=5),
            ),
            solver=polyfem.solver(
                linear=polyfem.linear(solver="Eigen::SimplicialLDLT"),
                nonlinear=polyfem.nonlinear(
                    solver=[{"type": "Newton"}, {"type": "GradientDescent"}],
                    norm_type="Euclidean",
                    rel_grad_norm_tol=0,
                    first_grad_norm_tol=1e-10,
                    grad_norm_tol=1e-08,
                ),
                advanced=polyfem.solver_advanced(
                    characteristic_force_density=1, characteristic_length=1
                ),
                max_threads=1,
            ),
            output=example_output(output),
        )
    )

    mesh_distance = diff.objectives.mesh_target(
        surface_selection=[6],
        mesh_path=str((DATA_DIR / "3d-shape-mesh-target.obj").resolve()),
        delta=0.05,
    )
    surface_distance = diff.objectives.sdf_target(
        surface_selection=[6],
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
    )

    objective = diff.objectives.sum(mesh_distance, surface_distance)

    obj = diff.shape(config, objective=objective)
    vertices = mesh_vertices(config)

    ids = torch.tensor(
        diff.model(config).boundary_vertex_ids([1, 2, 3, 4], exclude=True)
    )

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
