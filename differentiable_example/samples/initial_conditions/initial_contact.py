"""Match surface 1 to the reference trajectory during contact.

Each row of x holds the initial displacement and velocity of one degree of freedom.

Source: differentiability-data/input/initial-contact-opt.json
Run: python -m differentiable_example.samples.initial_conditions.initial_contact
"""

import torch

from differentiable_example._scene import (
    DATA_DIR,
    example_output,
    initial_state,
    simulation_directory,
    solver_config,
)
import polyfem.diff as diff
import polyfem


def build_scene(output, *, initial_velocity):
    model = polyfem.model()
    model.mesh(
        mesh=str(DATA_DIR / "square.obj"),
        transformation=polyfem.transformation(scale=[3, 0.02]),
        volume_selection=3,
        surface_selection=3,
    )
    model.mesh(
        mesh=str(DATA_DIR / "circle.msh"),
        transformation=polyfem.transformation(translation=[-1.5, 0.3], scale=0.5),
        volume_selection=1,
        surface_selection=1,
    )
    config = solver_config(
        model.config(
            materials=polyfem.neo_hookean(E=100000.0, nu=0.3, rho=1000),
            boundary_conditions=polyfem.boundary_conditions(
                rhs=[0, 9.8], dirichlet_boundary=[{"id": 3, "value": [0, 0]}]
            ),
            initial_conditions=polyfem.initial_conditions(
                velocity=[{"id": 1, "value": initial_velocity}]
            ),
            space=polyfem.space(
                discr_order=2, advanced=polyfem.space_advanced(quadrature_order=5)
            ),
            time=polyfem.time(tend=0.2, dt=0.04, integrator={"type": "BDF", "steps": 2}),
            contact=polyfem.contact(
                enabled=True, use_convergent_formulation=True, friction_coefficient=0.2
            ),
            solver=polyfem.solver(
                linear=polyfem.linear(solver="Eigen::SimplicialLDLT"),
                contact=polyfem.solver_contact(barrier_stiffness=10000.0),
                advanced=polyfem.solver_advanced(
                    characteristic_force_density=1, characteristic_length=1
                ),
                nonlinear=polyfem.nonlinear(
                    norm_type="Euclidean",
                    rel_grad_norm_tol=0,
                    first_grad_norm_tol=4e-12,
                    grad_norm_tol=4e-10,
                    line_search=polyfem.line_search(use_grad_norm_tol=4e-08),
                ),
                max_threads=1,
            ),
            output=example_output(output),
        )
    )

    return config


with simulation_directory() as output:
    config = build_scene(output=output / "config", initial_velocity=[5, 0])
    target = build_scene(output=output / "target", initial_velocity=[4, -1])

    matching = diff.objectives.transient_integral(
        static_objective=diff.objectives.target(target_state=1, surface_selection=[1])
    )

    trajectory = diff.initial(config, target=target, objective=matching)

    x = initial_state(config).requires_grad_()

    def loss(x):
        return trajectory(initial_condition=x)

    optimizer = torch.optim.Adam([x], lr=1e-5)
    optimizer.zero_grad()
    before = loss(x)
    before.backward()
    optimizer.step()

    with torch.no_grad():
        after = loss(x)
    print(f"Loss after one update: {before.item():.10g} -> {after.item():.10g}")
