"""Reduce the stress norm and penalize mesh distortion (AMIPS).

x stores the handle translations, followed by the interior vertex coordinates.

Source: differentiability-data/optimizations/shape-stress-bbw-opt
Run: python -m differentiable_example.optimizations.shape_design.shape_stress_bbw_opt
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import torch

import polyfem.diff as diff
import polyfem

from .._bbw import BAR_WEIGHTS, load_handle_data
from differentiable_example._scene import OPTIMIZATIONS, example_output, solver_config

ASSETS = OPTIMIZATIONS / "shape-stress-bbw-opt"


def build_scene(output):
    model = polyfem.model()
    model.mesh(
        mesh=str((ASSETS / "../../bar-2523.msh").resolve()),
        volume_selection=1,
        n_refs=0,
        surface_selection=[
            {"id": 1, "axis": "-x", "position": 2.5},
            {"id": 2, "axis": "+x", "position": 2.5},
        ],
    )
    config = model.config(
        materials=polyfem.neo_hookean(E=100000000.0, nu=0.47, rho=1000),
        boundary_conditions=polyfem.boundary_conditions(
            rhs=[0, 9.81, 0], dirichlet_boundary=[{"id": 1, "value": [0, 0, 0]}]
        ),
        space=polyfem.space(discr_order=1, advanced=polyfem.space_advanced(quadrature_order=4)),
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
    return solver_config(config)


def problem(config):
    # The original tolerance stalls near 2e-7 on this mesh.
    config["solver"]["nonlinear"]["grad_norm_tol"] = 5e-7
    vertices, skin, inside, weights, pivots = load_handle_data(config, BAR_WEIGHTS)
    n_controls = len(pivots) * 3
    initial = torch.cat((vertices.new_zeros(n_controls), vertices[inside].flatten())).tolist()

    def geometry(x):
        translations = x[:n_controls].reshape(-1, 3)
        shift = weights @ translations
        mesh = vertices.index_copy(0, skin, vertices[skin] + shift)
        return mesh.index_copy(0, inside, x[n_controls:].reshape(-1, 3))

    stress = diff.shape(
        config,
        objective=diff.objectives.create(
            "stress_norm", power=2, weight=1e-12, volume_selection=[1]
        ),
    )
    distortion = diff.shape(
        config,
        objective=diff.objectives.create("AMIPS", weight=1, volume_selection=[1]),
    )

    def loss(x):
        vertices = geometry(x)
        return stress(V=vertices) + distortion(V=vertices)

    return initial, loss


if __name__ == "__main__":
    with TemporaryDirectory() as directory:
        output = Path(directory)
        config = build_scene(output=output / "forward0")
        initial, loss = problem(config)
        x = torch.tensor(initial, dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.Adam([x], lr=1e-4)
        optimizer.zero_grad()
        before = loss(x)
        before.backward()
        optimizer.step()
        with torch.no_grad():
            after = loss(x)
        print(f"Loss after one update: {before.item():.10g} -> {after.item():.10g}")
