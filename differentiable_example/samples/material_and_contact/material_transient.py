"""Match the motion of surface 1 to the reference simulation.

x holds one Young's modulus per element across the bodies in contact.

Source: differentiability-data/input/material-transient-opt.json
Run: python -m differentiable_example.samples.material_and_contact.material_transient
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


def build_scene(output, *, young_modulus):
    model = polyfem.model()
    model.mesh(
        mesh=str(DATA_DIR / "square.obj"),
        transformation=polyfem.transformation(
            translation=[0, 0], rotation=0, scale=[3, 0.02]
        ),
        volume_selection=3,
        surface_selection=3,
    )
    model.mesh(
        mesh=str(DATA_DIR / "circle.msh"),
        transformation=polyfem.transformation(
            translation=[-1.5, 0.3], rotation=0, scale=0.5
        ),
        volume_selection=1,
        surface_selection=1,
    )
    config = solver_config(
        model.config(
            materials=polyfem.neo_hookean(E=young_modulus, nu=0.3, rho=1000, phi=10, psi=10),
            boundary_conditions=polyfem.boundary_conditions(
                rhs=[0, 9.8], dirichlet_boundary=[{"id": 3, "value": [0, 0]}]
            ),
            initial_conditions=polyfem.initial_conditions(
                velocity=[{"id": 1, "value": [5, -2]}]
            ),
            space=polyfem.space(
                discr_order=1, advanced=polyfem.space_advanced(quadrature_order=5)
            ),
            time=polyfem.time(tend=0.1, dt=0.01, integrator={"type": "BDF", "steps": 2}),
            contact=polyfem.contact(
                enabled=True,
                use_convergent_formulation=True,
                dhat=0.01,
                friction_coefficient=0.2,
            ),
            solver=polyfem.solver(
                linear=polyfem.linear(solver="Eigen::SimplicialLDLT"),
                nonlinear=polyfem.nonlinear(
                    max_iterations=100,
                    norm_type="Euclidean",
                    rel_grad_norm_tol=0,
                    grad_norm_tol=3.0000000000000004e-08,
                    line_search=polyfem.line_search(use_grad_norm_tol=1e-08),
                ),
                contact=polyfem.solver_contact(barrier_stiffness=100000.0),
                advanced=polyfem.solver_advanced(
                    characteristic_force_density=1, characteristic_length=1
                ),
                max_threads=1,
            ),
            output=example_output(output),
        )
    )

    return config


with simulation_directory() as output:
    config = build_scene(output=output / "config", young_modulus=1000000.0)
    target = build_scene(output=output / "target", young_modulus=100000.0)

    matching = diff.objectives.transient_integral(
        static_objective=diff.objectives.target(
            target_state=1, surface_selection=[1], reference_cached_body_ids=[1, 3]
        )
    )
    obj = diff.material(config, target=target, objective=matching)
    x = torch.full(
        (obj.n_elements,),
        config["materials"]["E"],
        dtype=torch.float64,
        requires_grad=True,
    )

    def loss(x):
        materials = {
            body: {"E": x[list(ids)], "nu": 0.3}
            for body, ids in obj.element_indices_by_body.items()
        }
        return obj(materials=materials)

    # E starts at 1e6; use a step in the same stiffness units.
    optimizer = torch.optim.Adam([x], lr=1000.0)
    optimizer.zero_grad()
    before = loss(x)
    before.backward()
    optimizer.step()

    with torch.no_grad():
        after = loss(x)
    print(f"Loss after one update: {before.item():.10g} -> {after.item():.10g}")
