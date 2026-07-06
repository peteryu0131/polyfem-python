#!/usr/bin/env python3
"""Forward solve using generated convenience shortcuts.

This mirrors ``examples/configs/contact_impact.json`` like
``02_forward_solve_generated_api.py``, but uses generated shortcut factories
such as ``G.dirichlet`` and ``G.surface_axis`` for common nested config shapes.
These shortcuts are generated from the same class tree/spec; they only reduce
verbose nested construction and do not introduce a second config model.
The final object is still a generated ``Root`` object.

Expected command:
    python polyfempy/new_api/example/03_forward_solve_generated_api_shortcuts.py

Expected output:
    A run directory under ``examples/runs/``.
"""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from polyfempy.api import solve
from polyfempy.api.runtime import make_timestamped_workspace


REPO_ROOT = Path(__file__).resolve().parents[3]
GENERATED_DIR = REPO_ROOT / "python-from-jse" / "generated"


def _load_module(module_name: str, module_file: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, module_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module file: {module_file}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_generated_api() -> ModuleType:
    _load_module("generated_class", GENERATED_DIR / "generated_class.py")
    return _load_module("polyfem_generated_api_shortcuts", GENERATED_DIR / "generated_api.py")


G = _load_generated_api()


EXAMPLE_ROOT = REPO_ROOT / "examples"
CONFIG_PATH = EXAMPLE_ROOT / "configs" / "contact_impact.json"
RUNS_DIR = EXAMPLE_ROOT / "runs"














materials = [
    G.neo_hookean(E=G.unit(20.0, "MPa"), nu=0.45, id=1,rho=G.unit(1100.0, "kg/m^3")),
    G.neo_hookean(E=G.unit(200.0, "GPa"),nu=0.45,id=2, rho=G.unit(7850.0, "kg/m^3")),
]

boundary_conditions = G.boundary_conditions(
    dirichlet=[G.dirichlet(id=1, value=[0.0, 0.0])],
    rhs=[0.0, 980.0],
)

geometry = [
    G.mesh(
        mesh="../assets/impact/triangular_lattice.msh",
        volume_selection=1,
        surface_selection=G.surface_axis(id=1, axis=-2, position=0.0001),
    ),
    G.mesh(mesh="../assets/impact/falling_weight_block.msh", volume_selection=2),
]

units = G.units(length="cm", mass="g", time="s")

solver = G.solver(
    linear=G.linear(solver="Eigen::PardisoLDLT"),
    nonlinear=G.nonlinear(solver="Newton", max_iterations=800, grad_norm_tol=0.002, Newton=G.newton(residual_tolerance=100.0)),
    contact=G.solver_contact(CCD=G.ccd(), barrier_stiffness="adaptive"),
)

time_cfg = G.tend_dt(tend=0.02, dt=0.01, integrator="ImplicitEuler")

output = G.output(
    directory="",
    json="impact_stats.json",
    paraview=G.output_paraview(
        options=G.options(material=True, body_ids=True, velocity=True),
        vismesh_rel_area=10000000,
    ),
    log=G.output_log(level="debug", file_level="debug", path="polyfem.log", quiet=True),
    advanced=G.output_advanced(timestep_prefix="impact_step_"),
)

contact = G.contact(enabled=True, dhat=0.012)

space = G.space(discr_order=1)

polyfem_config = G.config(
    root_path=str(CONFIG_PATH),
    materials=materials,
    boundary_conditions=boundary_conditions,
    geometry=geometry,
    units=units,
    solver=solver,
    time=time_cfg,
    output=output,
    contact=contact,
    space=space,
)







def config_for_workspace(workspace: Path) -> Any:
    cfg = copy.deepcopy(polyfem_config)
    cfg.output.directory = str(workspace)
    cfg.output.log.path = str(workspace / "polyfem.log")
    return cfg


def main() -> int:
    workspace = make_timestamped_workspace(RUNS_DIR, "03_forward_solve_generated_api_shortcuts")
    generated_config = config_for_workspace(workspace)
    solve(cfg=generated_config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
