"""Match surface 5 on body 2 to its reference trajectory.

x contains four horizontal offsets defining a cubic profile on boundary 4 of body 1. We chose these controls for the supplied scene.

Source: differentiability-data/optimizations/shape-trajectory-surface-opt-bspline
Run: python -m differentiable_example.optimizations.shape_matching.shape_trajectory_spline
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import torch

import polyfem.diff as diff
import polyfem

from differentiable_example._scene import OPTIMIZATIONS, example_output, solver_config
from .._trajectory import bernstein, mesh_and_boundary_nodes, trajectory_solver_settings

ASSETS = OPTIMIZATIONS / "shape-trajectory-surface-opt-bspline"


def build_scene(output, *, moving_mesh, selection_box, driven_boundary):
    model = polyfem.model()
    model.mesh(
        mesh=str((ASSETS / moving_mesh).resolve()),
        transformation=polyfem.transformation(translation=[0.5, 0]),
        volume_selection=1,
        n_refs=0,
        advanced={"normalize_mesh": False},
        surface_selection=[
            {"id": 1, "axis": "-x", "position": 0, "relative": False},
            {"id": 4, "relative": False, "box": selection_box},
        ],
    )
    model.mesh(
        mesh=str((ASSETS / "rectangle.obj").resolve()),
        volume_selection=2,
        n_refs=0,
        advanced={"normalize_mesh": False},
        surface_selection=[
            {"id": 2, "axis": "y", "position": 1.7, "relative": False},
            {"id": 3, "axis": "-y", "position": -1.7, "relative": False},
            {"id": 5, "relative": False, "box": [[1.1, -1.7], [1.3, 1.7]]},
        ],
    )
    config = model.config(
        materials=polyfem.neo_hookean(E=1000000000.0, nu=0.47, rho=1000, psi=0, phi=0),
        boundary_conditions=polyfem.boundary_conditions(
            dirichlet_boundary=[
                driven_boundary,
                {"id": 2, "value": [0, 0]},
                {"id": 3, "value": [0, 0]},
            ]
        ),
        space=polyfem.space(
            discr_order=1,
            advanced=polyfem.space_advanced(n_boundary_samples=5, quadrature_order=5),
        ),
        time=polyfem.time(t0=0, tend=0.4, dt=0.1),
        contact=polyfem.contact(
            enabled=True, use_convergent_formulation=True, friction_coefficient=0.0
        ),
        solver=polyfem.solver(
            linear=polyfem.linear(solver="Eigen::SimplicialLDLT"),
            nonlinear=polyfem.nonlinear(
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


def problem(config, target):
    trajectory_solver_settings([config, target])
    spatial = diff.objectives.target(
        target_state=1,
        surface_selection=[5],
        reference_cached_body_ids=[2],
    )
    obj = diff.shape(
        config,
        target=target,
        objective=diff.objectives.transient_integral(
            static_objective=spatial, integral_type="uniform"
        ),
    )
    vertices, nodes = mesh_and_boundary_nodes(config, boundary_id=4)
    profile = vertices[nodes]
    # Four controls move boundary 4 in x; weights vary along its height.
    weights = bernstein(profile[:, 1])
    initial = vertices.new_zeros(weights.shape[1]).tolist()

    def geometry(x):
        shift = (weights @ x)[:, None] * vertices.new_tensor([1., 0.])
        return vertices.index_copy(0, nodes, profile + shift)

    def loss(x):
        vertices = geometry(x)
        return obj(V=vertices)

    return initial, loss


if __name__ == "__main__":
    with TemporaryDirectory() as directory:
        output = Path(directory)
        config = build_scene(
            output=output / "forward0",
            moving_mesh="spline_9.obj",
            selection_box=[[0.49, -1.01], [1.01, 1.01]],
            driven_boundary={"id": 1, "value": ["1.0*t", 0]},
        )
        target = build_scene(
            output=output / "forward1",
            moving_mesh="spline_3.obj",
            selection_box=[[0.49, -0.51], [1.01, 1.51]],
            driven_boundary={"id": 1, "value": ["1.0*t", 0], "dimension": [True, True]},
        )
        initial, loss = problem(config, target)
        x = torch.tensor(initial, dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.Adam([x], lr=1e-4)
        optimizer.zero_grad()
        before = loss(x)
        before.backward()
        optimizer.step()
        with torch.no_grad():
            after = loss(x)
        print(f"Loss after one update: {before.item():.10g} -> {after.item():.10g}")
