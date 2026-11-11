"""Match the final horizontal displacement of surface 4 to 0.1.

Each row of x holds the pressures on boundaries 2 and 3 at one time step.

Source: differentiability-data/input/walker-opt.json
Run: python -m differentiable_example.samples.boundary_controls.walker
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
        mesh=str(DATA_DIR / "walker.obj"),
        volume_selection=1,
        transformation=polyfem.transformation(scale=0.1),
        n_refs=0,
        surface_selection=[
            {"id": 2, "box": [[0.1, -0.1], [0.2, 0.15]], "relative": False},
            {"id": 3, "box": [[-0.2, -0.1], [-0.1, 0.15]], "relative": False},
            {"id": 4, "axis": "y", "position": 0.16, "relative": False},
        ],
    )
    model.mesh(
        mesh=str(DATA_DIR / "square.obj"),
        volume_selection=1,
        transformation=polyfem.transformation(scale=[2, 0.01], translation=[0, -0.13]),
        n_refs=0,
        surface_selection=1,
    )
    config = solver_config(
        model.config(
            materials=[polyfem.neo_hookean(id=1, E=100000.0, nu=0.35, rho=1000)],
            boundary_conditions=polyfem.boundary_conditions(
                rhs=[0, 1],
                dirichlet_boundary=[{"id": 1, "value": [0, 0]}],
                pressure_boundary=[
                    {
                        "id": 2,
                        "value": [
                            0,
                            -0.148,
                            -5.262,
                            -17.195,
                            -34.781,
                            -56.3,
                            -79.647,
                            -102.54,
                            -122.74,
                            -138.272,
                            -147.616,
                            -149.86,
                            -144.782,
                            -132.881,
                            -115.32,
                            -93.816,
                            -70.472,
                            -47.571,
                            -27.352,
                            -11.792,
                            -2.414,
                        ],
                        "time_reference": [
                            0,
                            0.1,
                            0.2,
                            0.3,
                            0.4,
                            0.5,
                            0.6,
                            0.7,
                            0.8,
                            0.9,
                            1,
                            1.1,
                            1.2,
                            1.3,
                            1.4,
                            1.5,
                            1.6,
                            1.7,
                            1.8,
                            1.9,
                            2,
                        ],
                    },
                    {
                        "id": 3,
                        "value": [
                            0,
                            -0.148,
                            -5.262,
                            -17.195,
                            -34.781,
                            -56.3,
                            -79.647,
                            -102.54,
                            -122.74,
                            -138.272,
                            -147.616,
                            -149.86,
                            -144.782,
                            -132.881,
                            -115.32,
                            -93.816,
                            -70.472,
                            -47.571,
                            -27.352,
                            -11.792,
                            -2.414,
                        ],
                        "time_reference": [
                            0,
                            0.1,
                            0.2,
                            0.3,
                            0.4,
                            0.5,
                            0.6,
                            0.7,
                            0.8,
                            0.9,
                            1,
                            1.1,
                            1.2,
                            1.3,
                            1.4,
                            1.5,
                            1.6,
                            1.7,
                            1.8,
                            1.9,
                            2,
                        ],
                    },
                ],
            ),
            space=polyfem.space(
                discr_order=1,
                advanced=polyfem.space_advanced(quadrature_order=5, n_boundary_samples=5),
            ),
            time=polyfem.time(t0=0, tend=2.0, dt=0.1),
            contact=polyfem.contact(
                enabled=True,
                friction_coefficient=0.8,
                dhat=0.0001,
                use_convergent_formulation=True,
            ),
            solver=polyfem.solver(
                linear=polyfem.linear(solver="Eigen::SimplicialLDLT"),
                contact=polyfem.solver_contact(
                    barrier_stiffness=100000.0, friction_iterations=1
                ),
                nonlinear=polyfem.nonlinear(
                    max_iterations=1500,
                    solver=[
                        {"type": "Newton"},
                        {"type": "RegularizedNewton"},
                        {"type": "L-BFGS"},
                        {"type": "GradientDescent"},
                    ],
                    Newton={
                        "use_psd_projection": False,
                        "use_psd_projection_in_regularized": False,
                    },
                    norm_type="Euclidean",
                    rel_grad_norm_tol=0,
                    first_grad_norm_tol=1.0000000000000001e-11,
                    grad_norm_tol=1.0000000000000002e-10,
                    line_search=polyfem.line_search(use_grad_norm_tol=1e-07),
                ),
                advanced=polyfem.solver_advanced(
                    characteristic_force_density=1, characteristic_length=1
                ),
                max_threads=1,
            ),
            output=example_output(output),
        )
    )

    objective = diff.objectives.transient_integral(
        static_objective=diff.objectives.displacement_target(
            target_displacement=[0.1, 0],
            active_dimension=[True, False],
            surface_selection=[4],
        ),
        weight=100.0,
        print_energy="target_match",
        integral_type="final",
    )
    boundaries = config["boundary_conditions"]["pressure_boundary"]
    obj = diff.pressure(
        config, boundary_ids=[b["id"] for b in boundaries], objective=objective
    )

    # One row per simulated step, one column per controlled boundary.
    x = (
        torch.tensor([b["value"][1:] for b in boundaries], dtype=torch.float64)
        .T.contiguous()
        .requires_grad_()
    )

    def loss(x):
        return obj(values=x)

    optimizer = torch.optim.Adam([x], lr=1e-5)
    optimizer.zero_grad()
    before = loss(x)
    before.backward()
    optimizer.step()

    with torch.no_grad():
        after = loss(x)
    print(f"Loss after one update: {before.item():.10g} -> {after.item():.10g}")
