"""Minimize the stress norm in body 1.

Each row of x prescribes a 3D displacement for one of four selected nodes.

Source: differentiability-data/input/dirichlet-nodes-3d-opt.json
Run: python -m differentiable_example.samples.boundary_controls.dirichlet_nodes
"""

import torch

from differentiable_example._scene import (
    DATA_DIR,
    example_output,
    simulation_directory,
    solver_config,
)
import polyfem.diff as diff
import polyfem

with simulation_directory() as output:
    model = polyfem.model()
    model.mesh(mesh=str(DATA_DIR / "cube.msh"), volume_selection=1, n_refs=0)
    config = solver_config(
        model.config(
            materials=[polyfem.neo_hookean(id=1, E=730000.0, nu=0.47, rho=1000)],
            boundary_conditions=polyfem.boundary_conditions(
                dirichlet_boundary=[str(DATA_DIR / "input" / "dirichlet_constraints.txt")]
            ),
            space=polyfem.space(
                discr_order=1,
                advanced=polyfem.space_advanced(quadrature_order=5, n_boundary_samples=5),
            ),
            contact=polyfem.contact(enabled=False),
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

    obj = diff.dirichlet(
        config,
        node_ids=[0, 1, 4, 6],
        objective=diff.objectives.stress_norm(selection=1),
    )
    x = torch.zeros((4, 3), dtype=torch.float64, requires_grad=True)

    def loss(x):
        return obj(values=x)

    optimizer = torch.optim.Adam([x], lr=0.0001)
    optimizer.zero_grad()
    value = loss(x)
    value.backward()
    optimizer.step()

    with torch.no_grad():
        after = loss(x)
    print(f"Loss after one update: {value.item():.10g} -> {after.item():.10g}")
