#!/usr/bin/env python3
"""Forward solve using the generated config class directly.

This mirrors ``examples/01_forward_solve.py`` but avoids the guided
``SimulationConfig`` authoring path. The user-facing config is a generated
``GeneratedConfig`` object, and ``solve(cfg=generated_config)`` enters the
generated-only branch in the solve contract.

Expected command:
    python polyfempy/new_api/example/01_forward_solve_generated.py

Expected output:
    A run directory under ``examples/runs/``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from polyfempy.api import solve
from polyfempy.api.runtime import make_timestamped_workspace


REPO_ROOT = Path(__file__).resolve().parents[3]
GENERATED_DIR = REPO_ROOT / "python-from-jse" / "generated"


def _load_generated_config_class() -> Any:
    generated_file = GENERATED_DIR / "generated_class.py"
    spec = importlib.util.spec_from_file_location("polyfem_generated_class", generated_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load generated class file: {generated_file}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.Root


GeneratedConfig = _load_generated_config_class()


EXAMPLE_ROOT = REPO_ROOT / "examples"
RUNS_DIR = EXAMPLE_ROOT / "runs"
MESH_DIR = EXAMPLE_ROOT / "assets" / "impact"
LATTICE_MESH = MESH_DIR / "triangular_lattice.msh"
BLOCK_MESH = MESH_DIR / "falling_weight_block.msh"


def _value_with_unit(value: float, unit: str) -> dict[str, float | str]:
    return {"value": value, "unit": unit}


def _build_generated_config(workspace: Path) -> Any:
    fixed_surface = GeneratedConfig.Geometry.Mesh.Surface_selection.Axis(
        id=1,
        axis=-2,
        position=0.0001,
    )

    geometry = GeneratedConfig.Geometry(
        items=[
            GeneratedConfig.Geometry.Mesh(
                mesh=str(LATTICE_MESH),
                volume_selection=1,
                surface_selection=GeneratedConfig.Geometry.Mesh.Surface_selection(
                    fixed_surface
                ),
            ),
            GeneratedConfig.Geometry.Mesh(
                mesh=str(BLOCK_MESH),
                volume_selection=2,
            ),
        ]
    )

    materials = GeneratedConfig.Materials(
        items=[
            GeneratedConfig.Materials.NeoHookean(
                id=1,
                E=_value_with_unit(20.0, "MPa"),
                nu=0.45,
                rho=_value_with_unit(1100.0, "kg/m^3"),
            ),
            GeneratedConfig.Materials.NeoHookean(
                id=2,
                E=_value_with_unit(200.0, "GPa"),
                nu=0.45,
                rho=_value_with_unit(7850.0, "kg/m^3"),
            ),
        ]
    )

    boundary_conditions = GeneratedConfig.Boundary_conditions(
        rhs=[0.0, 980.0],
        dirichlet_boundary=GeneratedConfig.Boundary_conditions.Dirichlet_boundary(
            items=[
                GeneratedConfig.Boundary_conditions.Dirichlet_boundary.Item(
                    id=1,
                    value=GeneratedConfig.Boundary_conditions.Dirichlet_boundary.Item.Value(
                        items=[0.0, 0.0]
                    ),
                )
            ]
        ),
    )

    solver = GeneratedConfig.Solver(
        linear=GeneratedConfig.Solver.Linear(solver="Eigen::PardisoLDLT"),
        nonlinear=GeneratedConfig.Solver.Nonlinear(
            solver="Newton",
            grad_norm_tol=0.002,
            max_iterations=800,
            Newton=GeneratedConfig.Solver.Nonlinear.Newton(residual_tolerance=100.0),
        ),
        contact=GeneratedConfig.Solver.Contact(
            CCD=GeneratedConfig.Solver.Contact.CCD(
                broad_phase="hash_grid",
                tolerance=1e-6,
                max_iterations=1_000_000,
            ),
            barrier_stiffness="adaptive",
        ),
    )

    output = GeneratedConfig.Output(
        directory=str(workspace),
        json="results.json",
        log=GeneratedConfig.Output.Log(
            level="info",
            file_level="info",
            path=str(workspace / "polyfem.log"),
            quiet=True,
        ),
        paraview=GeneratedConfig.Output.Paraview(
            file_name="results.pvd",
            volume=True,
            surface=False,
            wireframe=False,
            points=False,
            high_order_mesh=True,
            vismesh_rel_area=10000000.0,
            skip_frame=1,
            options=GeneratedConfig.Output.Paraview.Options(
                material=True,
                body_ids=True,
                velocity=True,
            ),
        ),
    )

    return GeneratedConfig(
        geometry=geometry,
        materials=materials,
        units=GeneratedConfig.Units(length="cm", mass="g", time="s"),
        space=GeneratedConfig.Space(discr_order=1),
        time=GeneratedConfig.Time(
            GeneratedConfig.Time.TendDt(
                t0=0.0,
                tend=0.02,
                dt=0.01,
                integrator="ImplicitEuler",
            )
        ),
        contact=GeneratedConfig.Contact(enabled=True, dhat=0.012),
        boundary_conditions=boundary_conditions,
        solver=solver,
        output=output,
    )


def main() -> int:
    workspace = make_timestamped_workspace(RUNS_DIR, "01_forward_solve_generated")
    generated_config = _build_generated_config(workspace)
    solve(cfg=generated_config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
