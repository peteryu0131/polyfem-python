"""Match the final surface to a target signed distance field and penalize mesh distortion.

x stores the handle rotations and translations, followed by the interior vertex coordinates.

Source: differentiability-data/optimizations/3d-bspline-shape-matching
Run: python -m differentiable_example.optimizations.shape_matching.bspline_shape_matching_3d
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import torch

import polyfem.diff as diff
import polyfem

from .._bbw import BOX_WEIGHTS, load_handle_data, rotate
from differentiable_example._scene import OPTIMIZATIONS, example_output, solver_config

ASSETS = OPTIMIZATIONS / "3d-bspline-shape-matching"


def build_scene(output):
    model = polyfem.model()
    model.mesh(
        mesh=str((ASSETS / "box.msh").resolve()),
        transformation=polyfem.transformation(scale=[2, 2, 2]),
        volume_selection=1,
        n_refs=1,
        advanced={"normalize_mesh": False},
        surface_selection=[
            {"id": 1, "axis": "-y", "position": 0, "relative": False},
            {"id": 2, "relative": False, "box": [[-1.1, 0.99, -1.1], [1.1, 1.01, 1.1]]},
        ],
    )
    config = model.config(
        materials=polyfem.neo_hookean(E=1000000000.0, nu=0.47, rho=1000, psi=0, phi=0),
        boundary_conditions=polyfem.boundary_conditions(
            dirichlet_boundary=[{"id": 1, "value": [0, 0, 0]}]
        ),
        space=polyfem.space(
            discr_order=1,
            advanced=polyfem.space_advanced(n_boundary_samples=5, quadrature_order=5),
        ),
        time=polyfem.time(t0=0, tend=0.3, dt=0.1),
        contact=polyfem.contact(
            enabled=True, use_convergent_formulation=True, friction_coefficient=0.0
        ),
        solver=polyfem.solver(
            linear=polyfem.linear(solver="Eigen::SimplicialLDLT"),
            nonlinear=polyfem.nonlinear(
                max_iterations=500,
                norm_type="Euclidean",
                rel_grad_norm_tol=0,
                first_grad_norm_tol=1.0000000000000001e-11,
                grad_norm_tol=1e-09,
                line_search=polyfem.line_search(use_grad_norm_tol=1e-07),
            ),
            contact=polyfem.solver_contact(barrier_stiffness=100000000.0),
            advanced=polyfem.solver_advanced(
                characteristic_force_density=1, characteristic_length=1
            ),
            max_threads=1,
        ),
        output=example_output(output),
    )
    return solver_config(config)


def problem(config):
    vertices, skin, inside, weights, pivots = load_handle_data(config, BOX_WEIGHTS)
    n_controls = len(pivots) * 6
    initial = torch.cat((vertices.new_zeros(n_controls), vertices[inside].flatten())).tolist()

    def geometry(x):
        # Each handle has three rotation angles followed by three translations.
        handles = x[:n_controls].reshape(-1, 6)
        offsets = vertices[skin, None, :] - pivots
        rotation = rotate(offsets, handles[:, :3]) - offsets
        shift = weights @ handles[:, 3:] + (weights[:, :, None] * rotation).sum(1)
        mesh = vertices.index_copy(0, skin, vertices[skin] + shift)
        return mesh.index_copy(0, inside, x[n_controls:].reshape(-1, 3))

    surface_distance = diff.shape(
        config,
        objective=diff.objectives.create(
            "transient_integral",
            integral_type="final",
            weight=1,
            static_objective={
                "type": "sdf-target",
                "state": 0,
                "surface_selection": [2],
                "control_points_grid": [
                    [-1.0, 1.0, -1.0],
                    [-0.33333333333333337, 1.0, -1.0],
                    [0.33333333333333326, 1.0, -1.0],
                    [1.0, 1.0, -1.0],
                    [-1.0, 1.0, -0.33333333333333337],
                    [-0.33333333333333337, 1.3, -0.33333333333333337],
                    [0.33333333333333326, 1.3, -0.33333333333333337],
                    [1.0, 1.0, -0.33333333333333337],
                    [-1.0, 1.0, 0.33333333333333326],
                    [-0.33333333333333337, 1.3, 0.33333333333333326],
                    [0.33333333333333326, 1.3, 0.33333333333333326],
                    [1.0, 1.0, 0.33333333333333326],
                    [-1.0, 1.0, 1.0],
                    [-0.33333333333333337, 1.0, 1.0],
                    [0.33333333333333326, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                ],
                "knots_u": [0, 0, 0, 0, 1, 1, 1, 1],
                "knots_v": [0, 0, 0, 0, 1, 1, 1, 1],
                "delta": 0.1,
            },
        ),
    )
    distortion = diff.shape(
        config,
        objective=diff.objectives.create("AMIPS", weight=1e-08, volume_selection=[1]),
    )

    def loss(x):
        vertices = geometry(x)
        return surface_distance(V=vertices) + distortion(V=vertices)

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
