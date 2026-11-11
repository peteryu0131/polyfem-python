"""Minimize the sum of beam compliances under two load cases.

x contains one density per element. Both loads use the same filtered density field.

Source: differentiability-data/optimizations/topology-opt
Run: python -m differentiable_example.optimizations.material_and_contact.topology_opt
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import torch

import polyfem.diff as diff
import polyfem

from differentiable_example._scene import OPTIMIZATIONS, example_output, solver_config

ASSETS = OPTIMIZATIONS / "topology-opt"


def build_scene(output, *, loaded_boundary):
    model = polyfem.model()
    model.mesh(
        mesh=str((ASSETS / "cantilever.msh").resolve()),
        surface_selection=[{"threshold": 0.0001}],
        volume_selection=1,
        n_refs=0,
    )
    config = model.config(
        materials=polyfem.linear_elasticity(E=100.0, nu=0.35),
        boundary_conditions=polyfem.boundary_conditions(
            dirichlet_boundary=[{"id": 1, "value": [0, 0]}],
            neumann_boundary=[{"id": loaded_boundary, "value": [0, -1]}],
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


def triangle_centers(geometry):
    # These two states use the same untransformed ASCII Gmsh 2.2 triangle mesh.
    assert (
        len(geometry) == 1
        and "transformation" not in geometry[0]
        and geometry[0]["n_refs"] == 0
    )
    lines = Path(geometry[0]["mesh"]).read_text().splitlines()
    assert lines[1].split()[:2] == ["2.2", "0"]
    nodes = {}
    for line in lines[lines.index("$Nodes") + 2 : lines.index("$EndNodes")]:
        node, px, py, _ = line.split()
        nodes[int(node)] = [float(px), float(py)]
    faces = []
    for line in lines[lines.index("$Elements") + 2 : lines.index("$EndElements")]:
        _, kind, tags, *rest = map(int, line.split())
        if kind == 2:
            faces.append([nodes[n] for n in rest[tags:]])
    return torch.tensor(faces, dtype=torch.float64).mean(1)


def density_filter(geometry, radius):
    centers = triangle_centers(geometry)
    weights = (
        radius
        - torch.cdist(centers, centers, compute_mode="donot_use_mm_for_euclid_dist")
    ).clamp_min(0)
    weights = (weights / weights.sum(1, keepdim=True)).to_sparse()
    return weights


if __name__ == "__main__":
    with TemporaryDirectory() as directory:
        output = Path(directory)
        load_one = build_scene(output=output / "forward0", loaded_boundary=2)
        load_two = build_scene(output=output / "forward1", loaded_boundary=3)
        weights = density_filter(load_one["geometry"], radius=0.1)
        first = diff.material(load_one, body=1, objective=diff.Objective.COMPLIANCE)
        second = diff.material(load_two, body=1, objective=diff.Objective.COMPLIANCE)
        x = torch.full(
            (first.n_elements,), 0.3, dtype=torch.float64, requires_grad=True
        )

        def loss(x):
            density = weights @ x
            modulus = 200 * density.pow(5)
            return first(E=modulus, nu=0.35) + second(E=modulus, nu=0.35)

        optimizer = torch.optim.Adam([x], lr=1e-4)
        optimizer.zero_grad()
        before = loss(x)
        before.backward()
        optimizer.step()
        with torch.no_grad():
            x.clamp_(0.01, 1.0)
            after = loss(x)
        print(f"Loss after one update: {before.item():.10g} -> {after.item():.10g}")
