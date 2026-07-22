"""Public solve entry point and internal orchestration."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from . import _solve_backend as _backend
from . import _solve_contract as _contract
from .result import Result

__all__ = ["solve"]


def _result_from_backend_return(ret) -> Result:
    """Convert the standard C++ backend result bundle into a Result."""
    if not isinstance(ret, dict) or not ret.get("_result_bundle"):
        raise RuntimeError(
            "PolyFEM backend did not return the standard result bundle. "
            "Expected a dict with _result_bundle, vertices, cells, and u."
        )

    required = ("vertices", "cells", "u")
    missing = [key for key in required if key not in ret]
    if missing:
        raise RuntimeError(
            "PolyFEM backend result bundle is missing required keys: "
            + ", ".join(missing)
        )

    pressure = None
    if ret.get("p") is not None:
        pressure_candidate = np.asarray(ret["p"])
        if pressure_candidate.size > 0:
            pressure = pressure_candidate

    meta = {}
    if isinstance(ret.get("meta"), dict):
        meta.update(ret["meta"])

    return Result(
        np.asarray(ret["vertices"]),
        np.asarray(ret["cells"], dtype=np.int32),
        np.asarray(ret["u"]),
        p=pressure,
        meta=meta,
    )


def run_pipeline(
    vertices=None,
    cells=None,
    cfg=None,
    sidesets_func: Optional[Callable] = None,
    dtype=None,
) -> Result:
    """Drive the full solve pipeline."""
    canonical = _contract.prepare_canonical_solve_input(
        vertices=vertices,
        cells=cells,
        cfg=cfg,
        dtype=dtype,
    )
    solver = _backend.build_solver()
    ctx = _backend.configure_solver(
        solver,
        canonical.config,
        canonical.full_json,
        canonical.mesh_source,
        backend_settings=canonical.backend_settings,
    )
    _backend.apply_sidesets(solver, sidesets_func, ctx)

    ret = _backend.run_solver_stage(solver, canonical.full_json)
    return _result_from_backend_return(ret)


def solve(
    vertices=None,
    cells=None,
    cfg=None,
    sidesets_func: Optional[Callable] = None,
    dtype=None,
) -> Result:
    """Run a PolyFEM solve and return a ``Result``."""
    return run_pipeline(
        vertices=vertices,
        cells=cells,
        cfg=cfg,
        sidesets_func=sidesets_func,
        dtype=dtype,
    )
