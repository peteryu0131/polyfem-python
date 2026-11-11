"""Increase the final contact force measure while penalizing stress, mesh distortion and collisions.

x holds the 2D coordinates of four groups of nodes. The default demo uses fixed-direction tractions and 100 time steps.

Source: differentiability-data/optimizations/2d-shape-traction-force
Run: python -m differentiable_example.optimizations.shape_design.shape_traction_force_2d
"""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

import polyfem.diff as diff
from polyfempy import polyfempy as backend
import polyfem

from differentiable_example._scene import OPTIMIZATIONS, example_output, solver_config

ASSETS = OPTIMIZATIONS / "2d-shape-traction-force"


def build_scene(output, *, full_time_grid=False, original_pressure=False):
    model = polyfem.model()
    model.mesh(
        mesh=str((ASSETS / "cube_dense.msh").resolve()),
        transformation=polyfem.transformation(translation=[-0.77, 0]),
        surface_selection=[
            {"id": 2, "axis": "-x", "position": -1.0, "relative": False},
            {"id": 3, "axis": "x", "position": -0.3, "relative": False},
        ],
        volume_selection=2,
        n_refs=0,
    )
    model.mesh(
        mesh=str((ASSETS / "cube_dense.msh").resolve()),
        transformation=polyfem.transformation(translation=[0.77, 0]),
        surface_selection=[
            {"id": 4, "axis": "x", "position": 1.0, "relative": False},
            {"id": 5, "axis": "-x", "position": 0.3, "relative": False},
        ],
        volume_selection=3,
        n_refs=0,
    )
    model.mesh(
        mesh=str((ASSETS / "cube_dense.msh").resolve()),
        transformation=polyfem.transformation(scale=[0.5, 0.5]),
        surface_selection=1,
        volume_selection=1,
    )
    if original_pressure:
        loads = {
            "pressure_boundary": [
                {"id": 3, "value": "30000 * t"},
                {"id": 5, "value": "30000 * t"},
            ]
        }
    else:
        # The two open faces use equal and opposite horizontal tractions.
        loads = {
            "neumann_boundary": [
                {"id": 3, "value": ["30000 * t", 0]},
                {"id": 5, "value": ["-(30000 * t)", 0]},
            ]
        }
    config = model.config(
        materials=[
            polyfem.mooney_rivlin(id=1, c1=13286, c2=53838, k=2640264, rho=1000),
            polyfem.mooney_rivlin(id=2, c1=13286, c2=53838, k=2640264, rho=1000),
            polyfem.mooney_rivlin(id=3, c1=13286, c2=53838, k=2640264, rho=1000),
        ],
        boundary_conditions=polyfem.boundary_conditions(
            rhs=[0, 0],
            dirichlet_boundary=[
                {"id": 2, "value": [0.0, 0.0]},
                {"id": 4, "value": [0.0, 0.0]},
            ],
            **loads,
        ),
        space=polyfem.space(
            discr_order=1,
            advanced=polyfem.space_advanced(quadrature_order=5, n_boundary_samples=5),
        ),
        time=polyfem.time(
            t0=0,
            tend=0.25,
            time_steps=1000 if full_time_grid else 100,
            integrator={"type": "BDF", "steps": 1},
        ),
        contact=polyfem.contact(enabled=True, use_convergent_formulation=True, dhat=0.001),
        solver=polyfem.solver(
            linear=polyfem.linear(solver="Eigen::SimplicialLDLT"),
            contact=polyfem.solver_contact(barrier_stiffness=500),
            nonlinear=polyfem.nonlinear(
                max_iterations=500,
                norm_type="Euclidean",
                rel_grad_norm_tol=1e-8,
                first_grad_norm_tol=2.5e-14,
                grad_norm_tol=1e-6,
                line_search=polyfem.line_search(use_grad_norm_tol=2.5e-10),
            ),
            advanced=polyfem.solver_advanced(
                characteristic_force_density=1, characteristic_length=1
            ),
            max_threads=1,
        ),
        output=example_output(output),
    )
    return solver_config(config)


def problem(config):
    session = backend.DifferentiableSession()
    session.set_settings(config)
    vertices = torch.tensor(session.mesh_vertices(), dtype=torch.float64)
    nodes = [
        *session.boundary_vertex_ids([3], False),
        *session.interior_vertex_ids([2]),
        *session.boundary_vertex_ids([5], False),
        *session.interior_vertex_ids([3]),
    ]
    ids = torch.tensor(nodes, dtype=torch.long)
    force = diff.objectives.create(
        "quadratic_contact_force_norm",
        surface_selection=[1],
        dhat=config["contact"]["dhat"],
    )
    stress = diff.objectives.stress_norm(selection=1, power=8)
    objective = diff.shape(
        config,
        objective=diff.objectives.sum(
            diff.objectives.transient_integral(
                static_objective=force, integral_type="final", weight=-0.0001
            ),
            diff.objectives.transient_integral(
                static_objective=stress, integral_type="final", weight=1e-28
            ),
            diff.objectives.create("AMIPS", weight=1e-8, volume_selection=[2]),
            diff.objectives.create("AMIPS", weight=1e-8, volume_selection=[3]),
            diff.objectives.collision_barrier(dhat=5e-5, weight=1),
        ),
    )

    def loss(x):
        mesh = vertices.index_copy(0, ids, x.reshape(-1, 2))
        return objective(V=mesh)

    return vertices[ids].tolist(), loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-time-grid",
        action="store_true",
        help="Use all 1,000 original time steps",
    )
    parser.add_argument(
        "--original-pressure",
        action="store_true",
        help="Retain the upstream open-face pressure condition",
    )
    args = parser.parse_args()
    with TemporaryDirectory() as directory:
        output = Path(directory)
        config = build_scene(
            output / "forward0",
            full_time_grid=args.full_time_grid,
            original_pressure=args.original_pressure,
        )
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
