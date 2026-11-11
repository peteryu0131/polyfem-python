"""Match the final surface to the reference simulation with a collision penalty.

x holds 16 offsets on a 4-by-4 control grid that moves surface 2 along the y axis.

Source: differentiability-data/optimizations/3d-bspline-shape-trajectory-opt
Run: python -m differentiable_example.optimizations.shape_matching.shape_trajectory_3d
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import torch

import polyfem.diff as diff
import polyfem

from differentiable_example._scene import OPTIMIZATIONS, example_output, solver_config
from .._trajectory import bernstein, mesh_and_boundary_nodes, trajectory_solver_settings

ASSETS = OPTIMIZATIONS / "3d-bspline-shape-trajectory-opt"


def build_scene(output, *, moving_mesh, displacement):
    model = polyfem.model()
    model.mesh(
        mesh=str((ASSETS / moving_mesh).resolve()),
        transformation=polyfem.transformation(scale=[2, 2, 2]),
        volume_selection=1,
        n_refs=0,
        advanced={"normalize_mesh": False},
        surface_selection=[
            {"id": 1, "axis": "-y", "position": 0, "relative": False},
            {"id": 2, "relative": False, "box": [[-1.1, 0.99, -1.1], [1.1, 1.01, 1.1]]},
        ],
    )
    model.mesh(
        mesh=str((ASSETS / "sheet.msh").resolve()),
        transformation=polyfem.transformation(translation=[0, 1.2, 0]),
        volume_selection=2,
        n_refs=0,
        advanced={"normalize_mesh": False},
        surface_selection=[
            {"id": 3, "axis": "-z", "position": -1.75, "relative": False},
            {"id": 4, "axis": "z", "position": 1.75, "relative": False},
            {"id": 5, "relative": False, "box": [[-1.1, 1.24, -1.1], [1.1, 1.26, 1.1]]},
        ],
    )
    config = model.config(
        materials=polyfem.neo_hookean(E=1000000000.0, nu=0.47, rho=1000, psi=0, phi=0),
        boundary_conditions=polyfem.boundary_conditions(
            dirichlet_boundary=[
                {"id": 1, "value": displacement},
                {"id": 3, "value": [0, 0, 0]},
                {"id": 4, "value": [0, 0, 0]},
            ]
        ),
        space=polyfem.space(
            discr_order=1,
            advanced=polyfem.space_advanced(n_boundary_samples=5, quadrature_order=5),
        ),
        time=polyfem.time(t0=0, tend=0.5, dt=0.1),
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
            static_objective=spatial, integral_type="final"
        ),
    )
    vertices, nodes = mesh_and_boundary_nodes(config, boundary_id=2)
    profile = vertices[nodes]
    # Sixteen controls move boundary 2 in y; weights vary along z and x.
    weights = (bernstein(profile[:, 2])[:, :, None] * bernstein(profile[:, 0])[:, None, :]).flatten(1)
    initial = vertices.new_zeros(weights.shape[1]).tolist()

    def geometry(x):
        shift = (weights @ x)[:, None] * vertices.new_tensor([0., 1., 0.])
        return vertices.index_copy(0, nodes, profile + shift)

    barrier = diff.shape(
        config,
        objective=diff.objectives.collision_barrier(dhat=0.001, weight=1),
    )

    def loss(x):
        vertices = geometry(x)
        return obj(V=vertices) + barrier(V=vertices)

    return initial, loss


if __name__ == "__main__":
    with TemporaryDirectory() as directory:
        output = Path(directory)
        config = build_scene(
            output=output / "forward0",
            moving_mesh="box.msh",
            displacement=[0, "0.5*t", 0],
        )
        target = build_scene(
            output=output / "forward1",
            moving_mesh="sphere.msh",
            displacement=[0, "0.8*t", 0],
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
