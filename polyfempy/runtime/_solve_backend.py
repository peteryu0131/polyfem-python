"""Backend adapter helpers for the forward solve pipeline.

This module owns the mechanics of talking to the compiled PolyFEM backend:
constructing a solver, applying settings, attaching meshes, applying sidesets,
and running assemble/solve. User-facing config and mesh-source semantics live
in ``_solve_contract.py``; the final backend bundle is converted in ``solve.py``.
"""

from __future__ import annotations

import copy
import importlib
import json
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ._solve_contract import MeshSource, build_canonical_solver_settings


@dataclass
class SolverConfigContext:
    """State produced while configuring the solver, needed later for BC retouch."""

    settings_dict: Optional[Dict[str, Any]] = None
    bc: Dict[str, Any] = field(default_factory=dict)


def _first_attr(obj, *names):
    for name in names:
        if hasattr(obj, name):
            return name
    return None


def build_solver():
    """Instantiate the C++ Solver/State binding, failing with a clear message otherwise."""
    try:
        import polyfempy as pf
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(
            "polyfempy bindings not found. Please install/compile them first."
        ) from exc

    solver = None
    if getattr(pf, "cpp_backend_available", lambda: False)():
        _core = importlib.import_module("polyfempy.polyfempy")
        for ctor in ("Solver", "State"):
            if hasattr(_core, ctor):
                try:
                    solver = getattr(_core, ctor)()
                    break
                except Exception:
                    pass

    if solver is None:
        for ctor in ("Solver", "State"):
            if hasattr(pf, ctor):
                try:
                    candidate = getattr(pf, ctor)()
                    if hasattr(candidate, "set_settings") or hasattr(candidate, "settings"):
                        solver = candidate
                        break
                except Exception:
                    pass

    if solver is None:
        if not getattr(pf, "cpp_backend_available", lambda: False)():
            err = getattr(pf, "cpp_backend_error", lambda: None)()
            raise RuntimeError(
                "C++ backend not loaded. JSON/array mode requires the compiled extension. "
                f"Error: {err}. Build with: pip install -e . --no-build-isolation"
            )
        raise RuntimeError("No usable Solver/State constructor found in polyfempy.")
    return solver


def _apply_settings_json(solver, settings_json: str) -> None:
    if hasattr(solver, "set_settings"):
        solver.set_settings(settings_json, strict_validation=False)
        return
    if hasattr(solver, "settings"):
        try:
            solver.settings(settings_json, strict_validation=False)
            return
        except TypeError as exc:
            raise RuntimeError(
                "Found solver.settings() but it does not accept arguments; "
                "expected a settings setter or set_settings()."
            ) from exc
    raise RuntimeError("Missing set_settings(...) on solver.")


def _configure_json_mode(
    solver,
    full_json: dict,
    cfg,
    *,
    backend_settings: Optional[Dict[str, Any]] = None,
) -> SolverConfigContext:
    settings_dict = (
        copy.deepcopy(backend_settings)
        if backend_settings is not None
        else build_canonical_solver_settings(
            cfg,
            full_json=full_json,
            mesh_source=MeshSource(mode="json"),
        )
    )
    _apply_settings_json(solver, json.dumps(settings_dict))

    if hasattr(solver, "load_mesh_from_settings"):
        solver.load_mesh_from_settings()
    else:
        raise RuntimeError("JSON mode requires load_mesh_from_settings() method")
    return SolverConfigContext(settings_dict=None, bc={})


def _configure_array_mode(
    solver,
    cfg,
    mesh_source: MeshSource,
    *,
    backend_settings: Optional[Dict[str, Any]] = None,
) -> SolverConfigContext:
    settings_dict = (
        copy.deepcopy(backend_settings)
        if backend_settings is not None
        else build_canonical_solver_settings(
            cfg,
            full_json=None,
            mesh_source=mesh_source,
        )
    )

    if isinstance(cfg, dict):
        bc_raw = cfg.get("boundary_conditions", {}) or {}
    else:
        bc_raw = getattr(cfg, "boundary_conditions", {}) or {}
    bc = bc_raw.to_dict() if hasattr(bc_raw, "to_dict") else (bc_raw if isinstance(bc_raw, dict) else {})

    if bc:
        settings_dict.setdefault("boundary_conditions", {})
        settings_dict["boundary_conditions"].update(bc)

    _apply_settings_json(solver, json.dumps(settings_dict))
    return SolverConfigContext(settings_dict=settings_dict, bc=bc)


def _apply_mesh_array_mode(solver, mesh_source: MeshSource) -> None:
    V_np, C_np = mesh_source.vertices, mesh_source.cells
    for name in ("set_mesh", "set_mesh_data", "load_mesh_from_points"):
        if hasattr(solver, name):
            fn = getattr(solver, name)
            try:
                fn(V_np, C_np)
                break
            except TypeError:
                try:
                    fn(points=V_np, cells=C_np)
                    break
                except Exception:
                    pass
    else:
        raise RuntimeError("No mesh setter found (set_mesh / set_mesh_data / load_mesh_from_points).")

    mesh = solver.mesh() if hasattr(solver, "mesh") else None
    if mesh is None:
        return
    if mesh_source.body_ids is not None and hasattr(mesh, "set_body_ids"):
        mesh.set_body_ids(mesh_source.body_ids)
    if mesh_source.boundary_ids is not None and hasattr(mesh, "set_boundary_ids"):
        mesh.set_boundary_ids(mesh_source.boundary_ids)


def _retouch_bc_after_mesh(solver, ctx: SolverConfigContext) -> None:
    """Mirror original behavior: after set_mesh, re-apply BC on top of current/original settings."""
    bc = ctx.bc
    if not bc:
        return
    try:
        current_settings = solver.settings()
        if isinstance(current_settings, dict):
            current_settings.setdefault("boundary_conditions", {})
            if not isinstance(current_settings["boundary_conditions"], dict):
                current_settings["boundary_conditions"] = {}
            current_settings["boundary_conditions"].update(bc)
            _apply_settings_json(solver, json.dumps(current_settings))
            return
    except Exception:
        pass

    if ctx.settings_dict is None:
        return
    try:
        ctx.settings_dict.setdefault("boundary_conditions", {})
        ctx.settings_dict["boundary_conditions"].update(bc)
        _apply_settings_json(solver, json.dumps(ctx.settings_dict))
    except Exception:
        pass


def configure_solver(
    solver,
    cfg,
    full_json: Optional[dict],
    mesh_source: MeshSource,
    *,
    backend_settings: Optional[Dict[str, Any]] = None,
) -> SolverConfigContext:
    """Set settings, load/attach mesh. Returns context used by later retouch stages."""
    if mesh_source.mode == "json":
        return _configure_json_mode(
            solver,
            full_json,
            cfg,
            backend_settings=backend_settings,
        )

    ctx = _configure_array_mode(
        solver,
        cfg,
        mesh_source,
        backend_settings=backend_settings,
    )
    _apply_mesh_array_mode(solver, mesh_source)
    _retouch_bc_after_mesh(solver, ctx)
    return ctx


def apply_sidesets(
    solver,
    sidesets_func: Optional[Callable],
    ctx: SolverConfigContext,
) -> None:
    """Attach user-supplied boundary IDs in array mode and retouch BC after."""
    if sidesets_func is None:
        return

    try:
        mesh = solver.mesh()
        if hasattr(mesh, "set_boundary_ids") and hasattr(mesh, "n_boundary_elements"):
            n_boundary = mesh.n_boundary_elements()
            boundary_ids: List[int] = []
            for i in range(n_boundary):
                try:
                    v0 = mesh.boundary_element_vertex(i, 0)
                    v1 = mesh.boundary_element_vertex(i, 1)
                    p0 = mesh.point(v0)
                    p1 = mesh.point(v1)
                    center = (p0 + p1) / 2.0
                    bid = sidesets_func(center, True)
                    boundary_ids.append(bid)
                except Exception:
                    boundary_ids.append(-1)
            if boundary_ids:
                mesh.set_boundary_ids(np.array(boundary_ids, dtype=np.int32))
    except Exception as exc:
        warnings.warn(f"Failed to set boundary IDs: {exc}", RuntimeWarning)

    if ctx.bc and ctx.settings_dict is not None:
        try:
            ctx.settings_dict.setdefault("boundary_conditions", {})
            ctx.settings_dict["boundary_conditions"].update(ctx.bc)
            _apply_settings_json(solver, json.dumps(ctx.settings_dict))
        except Exception:
            pass


def _resolve_log_level(full_json: Optional[dict]) -> int:
    if not (full_json and "output" in full_json and "log" in full_json["output"]):
        return 2
    log_cfg = full_json["output"]["log"]
    # Keep terminal logging aligned with the generated OutputLog default.
    # ``OutputLog.to_dict()`` omits ``level`` when it is the default "debug",
    # so falling back to "info" here would silently make console output less
    # verbose than the file log.
    log_level_str = log_cfg.get("level", "debug")
    log_level_map = {
        "trace": 0, "debug": 1, "info": 2, "warn": 3, "warning": 3,
        "error": 4, "critical": 5, "off": 6,
    }
    return log_level_map.get(log_level_str, 2)


def run_solver_stage(solver, full_json: Optional[dict]):
    """Run the backend solve entry point and return its raw result."""
    name = _first_attr(solver, "solve", "run")
    if not name:
        raise RuntimeError("No solver entry point found (solve / run).")

    log_level = _resolve_log_level(full_json)
    try:
        return getattr(solver, name)(log_level=log_level)
    except TypeError:
        return getattr(solver, name)()


__all__ = [
    "SolverConfigContext",
    "apply_sidesets",
    "build_solver",
    "configure_solver",
    "run_solver_stage",
]
