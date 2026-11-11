"""Fit the moving surface to a target curve with collision and motion-smoothing penalties.

x starts with four shape controls, followed by ten horizontal displacements, one per time step.

Source: differentiability-data/optimizations/multiparameter-sdf-trajectory-surface-opt
Run: python -m differentiable_example.optimizations.shape_matching.multiparameter_sdf_trajectory
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import torch

import polyfem.diff as diff
import polyfem

from differentiable_example._scene import OPTIMIZATIONS, example_output, solver_config
from .._trajectory import bernstein, mesh_and_boundary_nodes

ASSETS = OPTIMIZATIONS / "multiparameter-sdf-trajectory-surface-opt"


def build_scene(output):
    model = polyfem.model()
    model.mesh(
        mesh=str((ASSETS / "spline_9.obj").resolve()),
        transformation=polyfem.transformation(translation=[0, 0]),
        volume_selection=1,
        n_refs=0,
        advanced={"normalize_mesh": False},
        surface_selection=[
            {"id": 1, "axis": "-x", "position": -0.5, "relative": False},
            {"id": 4, "relative": False, "box": [[-0.01, -1.01], [0.51, 1.01]]},
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
                {
                    "id": 1,
                    "value": [
                        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    ],
                    "time_reference": [
                        0.1,
                        0.2,
                        0.3,
                        0.4,
                        0.5,
                        0.6,
                        0.7,
                        0.8,
                        0.9,
                        1.0,
                        1.1,
                    ],
                },
                {"id": 2, "value": [0, 0]},
                {"id": 3, "value": [0, 0]},
            ]
        ),
        space=polyfem.space(
            discr_order=1,
            advanced=polyfem.space_advanced(n_boundary_samples=5, quadrature_order=5),
        ),
        time=polyfem.time(t0=0, tend=1.0, dt=0.1),
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


def build_problem(
    config, target, collision_distance, collision_weight, smoothing_weight
):
    config["solver"]["nonlinear"].update(rel_grad_norm_tol=1e-7, grad_norm_tol=1e-5)
    config["solver"]["augmented_lagrangian"] = {
        "nonlinear": {"rel_grad_norm_tol": 1e-7, "grad_norm_tol": 1e-5}
    }
    vertices, nodes = mesh_and_boundary_nodes(config, boundary_id=4)
    profile = vertices[nodes]
    # Four controls move boundary 4 in x; weights vary along its height.
    weights = bernstein(profile[:, 1])
    shape = vertices.new_zeros(weights.shape[1]).tolist()

    def geometry(controls):
        shift = (weights @ controls)[:, None] * vertices.new_tensor([1., 0.])
        return vertices.index_copy(0, nodes, profile + shift)

    loss_at_step = diff.objectives.transient_integral(
        static_objective=target, integral_type="uniform"
    )
    motion = config["boundary_conditions"]["dirichlet_boundary"][0]
    dt = config["time"]["dt"]
    # Re-sample the original horizontal ramp at t=0,...,tend.
    times = [i * dt for i in range(round(config["time"]["tend"] / dt) + 1)]
    assert motion["time_reference"] == motion["value"][0] and not any(
        motion["value"][1]
    )
    padded = times + [times[-1] + dt]
    motion.update(time_reference=padded, value=[padded, [0.0] * len(padded)])
    obj = diff.dirichlet(config, boundary_ids=[motion["id"]], objective=loss_at_step)
    collision = diff.shape(
        config,
        objective=diff.objectives.collision_barrier(
            dhat=collision_distance, weight=collision_weight
        ),
    )
    assert obj.input_shape == (len(times) - 1, 1, 2)

    def loss(x):
        vertices = geometry(x[: len(shape)])
        shift = x[len(shape) :]
        values = torch.stack((shift, torch.zeros_like(shift)), dim=1)[:, None, :]
        increments = torch.diff(shift, prepend=shift.new_zeros(1))
        return (
            obj(values=values, V=vertices)
            + collision(V=vertices)
            + smoothing_weight * increments.square().sum()
        )

    return shape + times[1:], loss


if __name__ == "__main__":
    with TemporaryDirectory() as directory:
        output = Path(directory)
        config = build_scene(output=output / "forward0")
        curve = diff.objectives.sdf_target(
            surface_selection=[5],
            control_points=[
                [1.2, -1.7],
                [1.53333333, -1.36666667],
                [1.33333333, 1.36666667],
                [1.2, 1.7],
            ],
            knots=[0, 0, 0, 0, 1, 1, 1, 1],
            delta=0.01,
        )
        initial, loss = build_problem(
            config,
            curve,
            collision_distance=0.01,
            collision_weight=1,
            smoothing_weight=0.1,
        )
        x = torch.tensor(initial, dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.Adam([x], lr=1e-4)
        optimizer.zero_grad()
        before = loss(x)
        before.backward()
        optimizer.step()
        with torch.no_grad():
            after = loss(x)
        print(f"Loss after one update: {before.item():.10g} -> {after.item():.10g}")
