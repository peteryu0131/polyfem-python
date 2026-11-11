"""Minimize the norm of (xx, xy, yx) stress divided by the norm of all four stress components.

x is the xx component of the imposed displacement gradient. The demo uses a coarse periodic mesh and a central-difference gradient.

Source: differentiability-data/optimizations/macro_strain-opt
Run: python -m differentiable_example.optimizations.initial_and_boundary_conditions.macro_strain_opt
"""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

import polyfem.diff as diff
from polyfempy import polyfempy as backend
import polyfem

from differentiable_example._scene import OPTIMIZATIONS, example_output, solver_config

ASSETS = OPTIMIZATIONS / "macro_strain-opt"


def build_scene(output, *, original_mesh=False):
    mesh = (
        ASSETS / "CMM.msh"
        if original_mesh
        else Path(__file__).resolve().parents[1] / "geometry" / "cmm_coarse.msh"
    )
    model = polyfem.model()
    model.mesh(
        mesh=str(mesh),
        surface_selection=[{"threshold": 1e-05}],
        transformation=polyfem.transformation(scale=1),
        volume_selection=1,
        n_refs=0,
    )
    config = model.config(
        materials=polyfem.neo_hookean(E=10000.0, nu=0.5, rho=1),
        boundary_conditions=polyfem.boundary_conditions(
            periodic=[{"boundary_ids": [1, 3]}, {"boundary_ids": [2, 4]}]
        ),
        space=polyfem.space(discr_order=1, advanced=polyfem.space_advanced(quadrature_order=5)),
        contact=polyfem.contact(enabled=False),
        solver=polyfem.solver(
            contact=polyfem.solver_contact(barrier_stiffness=10000.0),
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
    scene = solver_config(config)
    # SPQR is available in the solver but missing from the generated enum.
    scene["solver"]["linear"] = {"solver": "Eigen::SPQR"}
    scene["solver"]["adjoint_linear"] = {"solver": "Eigen::SPQR"}
    return scene


def problem(config):
    xx, xy, yx, yy = [-0.05, 0.02, 0.02, -0.1]

    def evaluate(xx):
        config["constraints"] = {
            "macro_displacement_gradient": {
                "value": [[xx, xy], [yx, yy]],
                "fixed_components": [0, 1, 2, 3],
            }
        }
        session = backend.DifferentiableSession()
        session.set_settings(config)
        vertices = torch.tensor(session.mesh_vertices(), dtype=torch.float64)
        stress = []
        for i, j in ((0, 0), (0, 1), (1, 0), (1, 1)):
            obj = diff.shape(
                config, objective=diff.objectives.stress(dimensions=[i, j])
            )
            stress.append(obj(V=vertices))
        sigma = torch.stack(stress)
        # The common cell-volume normalization cancels in this stress ratio.
        return torch.linalg.vector_norm(sigma[:3]) / torch.linalg.vector_norm(sigma)

    class MacroStress(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.xx = x.item()
            return evaluate(ctx.xx)

        @staticmethod
        def backward(ctx, *grad_outputs):
            (upstream,) = grad_outputs
            # The macro-strain control has no native adjoint binding. Differentiate
            # this single input with two independent periodic forward evaluations.
            h = 1e-5
            slope = (evaluate(ctx.xx + h) - evaluate(ctx.xx - h)) / (2 * h)
            return (upstream * slope).reshape(1)

    def loss(x):
        return MacroStress.apply(x)

    return [xx], loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--original-mesh",
        action="store_true",
        help="Use the 55,601-node CMM mesh (high memory requirement)",
    )
    args = parser.parse_args()
    with TemporaryDirectory() as directory:
        output = Path(directory)
        config = build_scene(
            output=output / "forward0", original_mesh=args.original_mesh
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
