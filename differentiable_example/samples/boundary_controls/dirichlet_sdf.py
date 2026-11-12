"""Bring surface 4 closer to the target curve at step 10.

x stores 2D displacements for three boundaries at successive times in the source table.

Source: differentiability-data/input/dirichlet-sdf-opt.json
Run: python -m differentiable_example.samples.boundary_controls.dirichlet_sdf
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

with simulation_directory(__file__) as output:
    model = polyfem.model()
    model.mesh(
        mesh=str(DATA_DIR / "bar198.obj"),
        transformation=polyfem.transformation(translation=[0, 0], rotation=0, scale=[2, 10]),
        volume_selection=1,
        n_refs=0,
        advanced={"normalize_mesh": False},
        surface_selection=[
            {"id": 1, "axis": "-x", "position": -0.8, "relative": False},
            {"id": 2, "axis": "x", "position": 0.8, "relative": False},
            {"id": 4, "relative": False, "box": [[-2, -1], [2, 0]]},
        ],
    )
    model.mesh(
        mesh=str(DATA_DIR / "circle36.obj"),
        transformation=polyfem.transformation(translation=[0, 0.5], rotation=0, scale=0.5),
        volume_selection=3,
        n_refs=0,
        advanced={"normalize_mesh": False},
        surface_selection=[{"id": 3, "axis": "y", "position": 0.5, "relative": False}],
    )
    config = solver_config(
        model.config(
            materials=polyfem.neo_hookean(E=100000.0, nu=0.44, rho=1000),
            boundary_conditions=polyfem.boundary_conditions(
                rhs=[0, 9.8],
                dirichlet_boundary=[
                    {
                        "id": 1,
                        "time_reference": [
                            0.02,
                            0.04,
                            0.06,
                            0.08,
                            0.1,
                            0.12,
                            0.14,
                            0.16,
                            0.18,
                            0.2,
                            0.22,
                        ],
                        "value": [
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        ],
                    },
                    {
                        "id": 2,
                        "time_reference": [
                            0.02,
                            0.04,
                            0.06,
                            0.08,
                            0.1,
                            0.12,
                            0.14,
                            0.16,
                            0.18,
                            0.2,
                            0.22,
                        ],
                        "value": [
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        ],
                    },
                    {
                        "id": 3,
                        "time_reference": [
                            0.02,
                            0.04,
                            0.06,
                            0.08,
                            0.1,
                            0.12,
                            0.14,
                            0.16,
                            0.18,
                            0.2,
                            0.22,
                        ],
                        "value": [
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [
                                0.0,
                                -0.04,
                                -0.08,
                                -0.12,
                                -0.16,
                                -0.17,
                                -0.18,
                                -0.19,
                                -0.2,
                                -0.21,
                                -0.22,
                            ],
                        ],
                    },
                ],
            ),
            space=polyfem.space(
                discr_order=1,
                advanced=polyfem.space_advanced(n_boundary_samples=5, quadrature_order=5),
            ),
            time=polyfem.time(tend=0.2, dt=0.02, integrator={"type": "BDF", "steps": 1}),
            contact=polyfem.contact(
                enabled=True,
                use_convergent_formulation=True,
                dhat=0.001,
                friction_coefficient=0.5,
            ),
            solver=polyfem.solver(
                linear=polyfem.linear(solver="Eigen::SimplicialLDLT"),
                nonlinear=polyfem.nonlinear(
                    norm_type="Euclidean",
                    rel_grad_norm_tol=0,
                    first_grad_norm_tol=2e-12,
                    grad_norm_tol=2e-08,
                    line_search=polyfem.line_search(use_grad_norm_tol=2e-08),
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

    target_distance = diff.objectives.sdf_target(
        surface_selection=[4],
        delta=0.05,
        control_points=[[-2.5, -0.1], [-1, -1], [1, 1], [2.5, -0.1]],
        knots=[0, 0, 0, 0, 1, 1, 1, 1],
    )
    objective = diff.objectives.transient_integral(
        static_objective=target_distance,
        integral_type="steps",
        steps=[10],
    )
    boundaries = config["boundary_conditions"]["dirichlet_boundary"]
    series = torch.tensor(
        [b["value"] for b in boundaries], dtype=torch.float64
    ).permute(2, 0, 1)
    # The source starts at dt. Preserve its timestamps and prepend the zero initial displacement.
    for b in boundaries:
        b["value"] = [[0.0, *row] for row in b["value"]]
        b["time_reference"] = [0.0, *b["time_reference"]]
    obj = diff.dirichlet(
        config, boundary_ids=[b["id"] for b in boundaries], objective=objective
    )
    x = series[1:].clone().requires_grad_()

    def loss(x):
        # The first simulated step uses the fixed first entry; the last table entry falls after the simulation.
        step_displacements = torch.cat((series[:1], x[:-1]))
        return obj(values=step_displacements)

    optimizer = torch.optim.Adam([x], lr=1e-5)
    optimizer.zero_grad()
    before = loss(x)
    before.backward()
    optimizer.step()

    with torch.no_grad():
        after = loss(x)
    print(f"Loss after one update: {before.item():.10g} -> {after.item():.10g}")
