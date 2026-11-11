"""Minimize the weighted stress norm plus the mesh smoothing energy.

x holds the undeformed mesh vertex coordinates.

Source: differentiability-data/input/laplacian-opt.json
Run: python -m differentiable_example.samples.shape_design.laplacian
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
    model.geometry_mesh(mesh=str(DATA_DIR / "circle2.msh"), surface_selection=1)
    config = solver_config(
        model.config(
            materials=polyfem.laplacian(type="Laplacian"),
            space=polyfem.space(
                discr_order=1, advanced=polyfem.space_advanced(quadrature_order=4)
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

    # The scalar Laplacian uses scalar load and boundary values.
    config["boundary_conditions"] = {
        "rhs": -20,
        "dirichlet_boundary": [{"id": 1, "value": 0}],
    }

    stress = diff.objectives.stress_norm(selection=[], weight=10000.0)
    smoothing = diff.objectives.boundary_smoothing(scale_invariant=True, power=3)

    objective = diff.objectives.sum(stress, smoothing)

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
