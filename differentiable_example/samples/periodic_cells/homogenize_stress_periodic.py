"""Minimize the cell's yy stress objective while maintaining periodicity.

x contains the reduced vertex coordinates and affine parameters of the periodic cell.

Source: differentiability-data/input/homogenize-stress-periodic-opt.json
Run: python -m differentiable_example.samples.periodic_cells.homogenize_stress_periodic
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
    model.geometry_mesh(
        mesh=str(DATA_DIR / "periodic105.msh"), n_refs=0,
        surface_selection=[{"threshold": 1e-06}],
    )
    config = solver_config(
        model.config(
            materials=polyfem.neo_hookean(E=1000.0, nu=0.5),
            boundary_conditions=polyfem.boundary_conditions(
                periodic=[{"boundary_ids": [1, 3]}, {"boundary_ids": [2, 4]}]
            ),
            space=polyfem.space(
                discr_order=1, advanced=polyfem.space_advanced(quadrature_order=4)
            ),
            contact=polyfem.contact(
                enabled=True, use_convergent_formulation=True, dhat=0.005, periodic=True
            ),
            constraints=polyfem.constraints(
                macro_displacement_gradient={
                    "value": [[0, 0], [0, -0.3]],
                    "fixed_components": [3],
                }
            ),
            solver=polyfem.solver(
                nonlinear=polyfem.nonlinear(
                    norm_type="Euclidean",
                    rel_grad_norm_tol=0,
                    first_grad_norm_tol=1e-10,
                    grad_norm_tol=1e-09,
                ),
                contact=polyfem.solver_contact(barrier_stiffness=1),
                advanced=polyfem.solver_advanced(
                    characteristic_force_density=1, characteristic_length=1
                ),
                max_threads=1,
            ),
            output=example_output(output),
        )
    )

    # SPQR is supported by the solver but is missing from the generated enum.
    config["solver"]["linear"] = {"solver": "Eigen::SPQR"}
    config["solver"]["adjoint_linear"] = {"solver": "Eigen::SPQR"}

    stress = diff.objectives.stress(dimensions=[1, 1])
    obj = diff.periodic_shape(config, objective=stress)
    x = obj.initial_parameters.requires_grad_()

    def loss(x):
        return obj(parameters=x)

    optimizer = torch.optim.Adam([x], lr=1e-5)
    optimizer.zero_grad()
    before = loss(x)
    before.backward()
    optimizer.step()

    with torch.no_grad():
        after = loss(x)
    print(f"Loss after one update: {before.item():.10g} -> {after.item():.10g}")
