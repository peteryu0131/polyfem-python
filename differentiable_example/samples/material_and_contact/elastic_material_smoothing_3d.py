"""Minimize the material smoothing energy with Poisson's ratio fixed at 0.3.

x holds the Young's moduli of six tetrahedra.

Source: differentiability-data/input/elastic-material-smoothing-3d-opt.json
Run: python -m differentiable_example.samples.material_and_contact.elastic_material_smoothing_3d
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
    model.mesh(
        mesh=str(DATA_DIR / "cube.msh"),
        volume_selection=1,
        surface_selection={"id": 1, "axis": "-x", "position": 0.01, "relative": True},
    )
    config = solver_config(
        model.config(
            materials=polyfem.linear_elasticity(E=1.0, nu=0.3),
            boundary_conditions=polyfem.boundary_conditions(
                rhs=[0, 0, 0], dirichlet_boundary=[{"id": 1, "value": [0, 0, 0]}]
            ),
            space=polyfem.space(discr_order=1),
            solver=polyfem.solver(
                linear=polyfem.linear(solver="Eigen::SimplicialLDLT"), max_threads=1
            ),
            output=example_output(output),
        )
    )

    smoothing = diff.objectives.elastic_material_smoothing()
    obj = diff.material(config, body=1, objective=smoothing)
    x = torch.tensor(
        [1.0, 1.2, 1.4, 1.6, 1.8, 2.0], dtype=torch.float64, requires_grad=True
    )

    def loss(x):
        return obj(E=x, nu=0.3)

    optimizer = torch.optim.Adam([x], lr=1e-5)
    optimizer.zero_grad()
    before = loss(x)
    before.backward()
    optimizer.step()

    with torch.no_grad():
        after = loss(x)
    print(f"Loss after one update: {before.item():.10g} -> {after.item():.10g}")
