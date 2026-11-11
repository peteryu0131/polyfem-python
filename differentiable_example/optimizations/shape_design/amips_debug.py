"""Minimize the triangle distortion energy (AMIPS).

x is the 2D position of the only free vertex. Its initial gradient is zero.

Source: differentiability-data/optimizations/AMIPS-debug
Run: python -m differentiable_example.optimizations.shape_design.amips_debug
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import torch

import polyfem.diff as diff
import polyfem

from differentiable_example._scene import OPTIMIZATIONS, example_output, solver_config

ASSETS = OPTIMIZATIONS / "AMIPS-debug"


def build_scene(output):
    model = polyfem.model()
    model.mesh(
        mesh=str((ASSETS / "triangle.obj").resolve()),
        volume_selection=1,
        n_refs=0,
        surface_selection=[{"id": 1, "box": [[-0.1, -0.1], [1.1, 0.1]]}],
    )
    config = model.config(
        materials=polyfem.linear_elasticity(E=200.0, nu=0.35),
        boundary_conditions=polyfem.boundary_conditions(
            rhs=[0, 9.81], dirichlet_boundary=[{"id": 1, "value": [0, 0]}]
        ),
        space=polyfem.space(
            discr_order=1,
            advanced=polyfem.space_advanced(quadrature_order=4, n_boundary_samples=4),
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
    return solver_config(config)


def problem(config):
    model = diff.model(config)
    free = model.boundary_vertex_ids([1], exclude=True)
    assert free == (2,)
    vertices = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    obj = diff.shape(model, objective=diff.objectives.create("AMIPS", weight=1.0))

    def loss(x):
        return obj(V=vertices.index_copy(0, torch.tensor(free), x.reshape(1, 2)))

    return [0.0, 1.0], loss


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
