"""Reduce bridge stress with penalties for excess volume, boundary roughness and mesh distortion.

x holds the 2D coordinates of the selected bridge nodes.

Source: differentiability-data/optimizations/shape-stress-opt
Run: python -m differentiable_example.optimizations.shape_design.shape_stress_opt
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import torch

import polyfem.diff as diff
from polyfempy import polyfempy as backend
import polyfem

from differentiable_example._scene import OPTIMIZATIONS, example_output, solver_config

ASSETS = OPTIMIZATIONS / "shape-stress-opt"


def build_scene(output):
    model = polyfem.model()
    model.mesh(
        mesh=str((ASSETS / "bridge-uniform.msh").resolve()),
        volume_selection=1,
        n_refs=0,
        surface_selection=[
            {"id": 10, "box": [[-1.00001, -0.02], [-0.9999, 0.027]]},
            {"id": 10, "box": [[0.9999, -0.02], [1.00001, 0.027]]},
            {"id": 11, "box": [[-0.6766, 0.0], [-0.6566, 0.1]]},
            {"id": 11, "box": [[-0.01, 0.0], [0.01, 0.1]]},
            {"id": 11, "box": [[0.6566, 0.0], [0.6766, 0.1]]},
            {"id": 1, "box": [[0, 0], [1, 1]], "relative": True},
        ],
    )
    config = model.config(
        materials=polyfem.linear_elasticity(E=200.0, nu=0.35),
        boundary_conditions=polyfem.boundary_conditions(
            dirichlet_boundary=[{"id": 10, "value": [0, 0]}],
            neumann_boundary=[{"id": 11, "value": [0, "-0.1/0.0206065848"]}],
        ),
        space=polyfem.space(discr_order=2, advanced=polyfem.space_advanced(quadrature_order=5)),
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
    session = backend.DifferentiableSession()
    session.set_settings(config)
    vertices = torch.tensor(session.mesh_vertices(), dtype=torch.float64)
    nodes = torch.tensor(
        [int(i) for i in (ASSETS / "opt-nodes.txt").read_text().split()]
    )
    terms = [
        diff.shape(
            config,
            objective=diff.objectives.create(
                "power",
                power=0.125,
                weight=0.001,
                objective={
                    "type": "stress_norm",
                    "power": 8,
                    "state": 0,
                    "volume_selection": [1],
                },
            ),
        ),
        diff.shape(
            config,
            objective=diff.objectives.create(
                "soft_constraint",
                objective={"type": "volume", "state": 0, "volume_selection": [1]},
                weight=10,
                soft_bound=[0, 0.236226],
            ),
        ),
        diff.shape(
            config,
            objective=diff.objectives.create(
                "boundary_smoothing",
                weight=0.001,
                scale_invariant=True,
                power=4,
                surface_selection=[],
            ),
        ),
        diff.shape(
            config,
            objective=diff.objectives.create(
                "AMIPS", weight=1e-06, volume_selection=[1]
            ),
        ),
    ]

    def loss(x):
        mesh = vertices.index_copy(0, nodes, x.reshape(-1, 2))
        return torch.stack([term(V=mesh) for term in terms]).sum()

    return vertices[nodes].tolist(), loss


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
