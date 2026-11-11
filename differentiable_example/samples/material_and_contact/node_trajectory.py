"""Minimize the mismatch between simulated node positions and prescribed targets.

Each row of x holds one element's two Lamé parameters, lambda and mu.

Source: differentiability-data/input/node-trajectory-opt.json
Run: python -m differentiable_example.samples.material_and_contact.node_trajectory
"""

from pathlib import Path

import numpy as np
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
        volume_selection=1,
        surface_selection=[
            {"id": 1, "axis": "y", "position": 0.499},
            {"id": 2, "axis": "-y", "position": -0.499},
        ],
    )
    config = solver_config(
        model.config(
            materials=polyfem.linear_elasticity(E=200, nu=0.3, rho=1),
            boundary_conditions=polyfem.boundary_conditions(
                dirichlet_boundary=[
                    {"id": 1, "value": [0, 0]},
                    {"id": 2, "value": [0.2, 0]},
                ]
            ),
            space=polyfem.space(advanced=polyfem.space_advanced(quadrature_order=4)),
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

    # The upstream test supplies its node targets at runtime.
    vertices = mesh_vertices(config)
    targets = 10 * torch.rand(
        vertices.shape,
        dtype=vertices.dtype,
        generator=torch.Generator().manual_seed(42),
    )
    target_file = Path.cwd() / "node_targets.txt"
    np.savetxt(target_file, torch.cat((vertices, targets), dim=1).numpy())
    obj = diff.material(
        config,
        body=1,
        objective=diff.objectives.create(
            "node-target", target_data_path=str(target_file)
        ),
    )
    E, nu = config["materials"]["E"], config["materials"]["nu"]
    x = (
        torch.tensor([E * nu / (1 - nu**2), E / (2 * (1 + nu))], dtype=torch.float64)
        .repeat(obj.n_elements, 1)
        .requires_grad_()
    )

    def loss(x):
        # Convert the 2D Lamé parameters to the E and nu accepted by the material API.
        lam, mu = x.unbind(dim=1)
        nu = lam / (lam + 2 * mu)
        return obj(E=2 * mu * (1 + nu), nu=nu)

    optimizer = torch.optim.Adam([x], lr=1e-5)
    optimizer.zero_grad()
    before = loss(x)
    before.backward()
    optimizer.step()

    with torch.no_grad():
        after = loss(x)
    print(f"Loss after one update: {before.item():.10g} -> {after.item():.10g}")
