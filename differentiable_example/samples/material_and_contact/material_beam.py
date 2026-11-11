"""Minimize the compliance of a beam fixed at its left end.

Source: differentiability-data/input/topology-compliance-opt.json
Run: python -m differentiable_example.samples.material_and_contact.material_beam
"""

import torch
import polyfem
import polyfem.diff as diff

from differentiable_example._scene import DATA_DIR, example_output, simulation_directory

initial_E = 100000.0

with simulation_directory() as output:
    model = polyfem.model()
    beam = model.mesh(mesh=str(DATA_DIR / "bar198.obj"), volume_selection=1)
    beam.material(polyfem.linear_elasticity(E=initial_E, nu=0.3))
    beam.surface_axis(axis=-1, position=0.0001, relative=True, id=1).dirichlet(value=[0, 0])

    config = model.config(
        boundary_conditions=polyfem.boundary_conditions(rhs=[10, 100]),
        space=polyfem.space(
            discr_order=1,
            advanced=polyfem.space_advanced(quadrature_order=5, mass_quadrature_order=5),
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

    obj = diff.material(config, body=beam, selection=1, objective=diff.Objective.COMPLIANCE)

    # Each element starts with E = initial_E and has its own design variable.
    x = torch.full((obj.n_elements,), initial_E ** (1 / 5), dtype=torch.float64, requires_grad=True)

    def loss(x):
        E = x.pow(5)
        return obj(E=E, nu=0.3)

    optimizer = torch.optim.Adam([x], lr=0.001)
    optimizer.zero_grad()
    before = loss(x)
    before.backward()
    optimizer.step()

    with torch.no_grad():
        after = loss(x)
    print(f"Compliance after one update: {before.item():.10f} -> {after.item():.10f}")
