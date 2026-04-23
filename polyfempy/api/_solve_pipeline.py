"""Solve pipeline stages.

This module breaks ``solve()`` into small, independently testable stages plus a
few typed intermediate structures. The public entry point in ``solve.py`` keeps
its signature and external behavior; it only delegates to the functions here.

Pipeline (linear):

    normalize_cfg          -> SimulationConfig
    build_full_json        -> Optional[dict]
    resolve_runtime        -> RuntimeOptions
    normalize_inputs       -> NormalizedInputs
    build_solver           -> solver handle (C++ backend)
    configure_solver       -> SolverConfigContext
    apply_sidesets         -> (in place on solver, may retouch settings)
    run_solver_stage       -> raw solver return value
    extract_native_outputs -> NativeOutputs
    apply_sampled_fallback -> Result (in place)
    finalize_result        -> Result

Each stage takes typed inputs and returns a typed (or documented) output, so
the branch matrix can be unit tested without needing the compiled backend.
"""

from __future__ import annotations

import copy
import importlib
import json
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import tensor as T
from .result import Result


# ---------------------------------------------------------------------------
# Intermediate data structures
# ---------------------------------------------------------------------------


@dataclass
class RuntimeOptions:
    """Runtime output/fallback knobs extracted from cfg.output / full_json.output."""

    requested_fields: Optional[List[str]] = None
    strict: bool = False
    fallback_mode: str = "never"
    temp_storage: str = "ram"
    keep_temp_files: bool = False


@dataclass
class NormalizedInputs:
    """Mesh inputs normalized to NumPy + the resolved execution mode."""

    V_np: Optional[np.ndarray]
    C_np: Optional[np.ndarray]
    v_backend: str
    use_json_mode: bool


@dataclass
class SolverConfigContext:
    """State produced while configuring the solver, needed later for BC retouch."""

    settings_dict: Optional[Dict[str, Any]] = None
    bc: Dict[str, Any] = field(default_factory=dict)
    use_json_mode: bool = False


@dataclass
class NativeOutputs:
    """Native (i.e. direct from solver) fields + mesh that become a Result."""

    vertices: np.ndarray
    cells: np.ndarray
    fields: Dict[str, np.ndarray]
    meta: Dict[str, Any]


# ---------------------------------------------------------------------------
# Small helpers (no solver state)
# ---------------------------------------------------------------------------


def _first_attr(obj, *names):
    for n in names:
        if hasattr(obj, n):
            return n
    return None


def _ensure_i32(cells):
    return cells.astype(np.int32, copy=False) if cells.dtype != np.int32 else cells


def _field_available(result: Result, name: str) -> bool:
    if name == "von_mises":
        return result.von_mises is not None
    if name == "von_mises_avg":
        arr = result.field("von_mises_avg")
        return arr is not None and np.asarray(arr).size > 0
    arr = result.field(name)
    return arr is not None and np.asarray(arr).size > 0

def _promote_materials_to_list(payload: Dict[str, Any], *, infer_type_from_pde: bool = False) -> None:
    """In-place: promote ``payload['materials']`` from a dict to a singleton list.

    The C++ JSON schema always expects ``materials`` to be a list, but Python-side
    callers often build a single material as a flat dict. Both entry points that
    feed the solver (``build_full_json`` and ``_configure_array_mode``) need this
    normalization, so it lives here once.

    When ``infer_type_from_pde`` is True and the material dict has no ``type``
    key, one is filled in based on ``payload.get('pde')`` — ``"Laplacian"`` for
    ``"Poisson"``, ``"LinearElasticity"`` otherwise. This matches the legacy
    behavior used by the array-mode configure path.
    """
    materials = payload.get("materials")
    if not isinstance(materials, dict):
        return
    if infer_type_from_pde and "type" not in materials:
        pde = payload.get("pde", "LinearElasticity")
        materials["type"] = "Laplacian" if pde == "Poisson" else "LinearElasticity"
    payload["materials"] = [materials]


# ---------------------------------------------------------------------------
# JSON-config normalization helpers (kept as module-level, pure functions)
# ---------------------------------------------------------------------------


def process_json_config(full_json: dict, cfg) -> dict:
    """Normalize a full JSON config for the C++ solver.

    - Strips the optional ``common`` key (no separate common.json is used).
    - Removes Python-side runtime output controls (``output.result`` /
      ``output.fallback`` / ``output.save_paraview`` / ``output.save_vtu``),
      which are not part of the C++ JSON schema.
    - Resolves relative mesh paths using ``root_path``.
    """
    processed = copy.deepcopy(full_json)
    processed.pop("common", None)

    out = processed.get("output")
    if isinstance(out, dict):
        out.pop("result", None)
        out.pop("fallback", None)
        out.pop("save_paraview", None)
        out.pop("save_vtu", None)

    root_path = processed.get("root_path")
    geometry = processed.get("geometry")
    if root_path and isinstance(geometry, list):
        root_dir = Path(root_path).resolve().parent
        for entry in geometry:
            if not isinstance(entry, dict):
                continue
            mesh_path = entry.get("mesh")
            if not isinstance(mesh_path, str) or not mesh_path.strip():
                continue
            mesh_file = Path(mesh_path)
            if mesh_file.is_absolute():
                continue
            direct = root_dir / mesh_file
            if direct.exists():
                entry["mesh"] = str(direct)
                continue
            sibling = root_dir.parent / "meshes" / mesh_file.name
            if sibling.exists():
                entry["mesh"] = str(sibling)
    return processed


def clean_json_for_cpp(obj, path: str = ""):
    """Recursively drop ``None`` values in a JSON-like object, keeping solver blocks intact."""
    if isinstance(obj, dict):
        cleaned = {}
        for key, value in obj.items():
            current_path = f"{path}.{key}" if path else key
            cleaned_value = clean_json_for_cpp(value, current_path)
            if cleaned_value is not None:
                cleaned[key] = cleaned_value
        return cleaned
    if isinstance(obj, list):
        return [
            clean_json_for_cpp(item, f"{path}[{i}]")
            for i, item in enumerate(obj)
            if clean_json_for_cpp(item, f"{path}[{i}]") is not None
        ]
    return obj


def merge_user_cfg_over_full_json(cfg, full_json) -> dict:
    """Overlay ``SimulationConfig`` edits on top of the original full JSON."""
    try:
        cfg_dict = cfg.to_dict()
    except Exception:
        return full_json

    if not isinstance(cfg_dict, dict):
        return full_json

    merged = copy.deepcopy(full_json) if isinstance(full_json, dict) else {}
    for key, value in cfg_dict.items():
        merged[key] = value

    if hasattr(cfg, "extras") and cfg.extras and "_root_path" in cfg.extras:
        merged["root_path"] = cfg.extras["_root_path"]
    elif isinstance(full_json, dict) and "root_path" in full_json:
        merged["root_path"] = full_json["root_path"]

    return merged


# ---------------------------------------------------------------------------
# Stage 1: normalize cfg into a SimulationConfig
# ---------------------------------------------------------------------------


def normalize_cfg(cfg):
    """Accept dict / path / SimulationConfig, always return a SimulationConfig.

    Fails fast with a clear error for everything else. Imported lazily so that
    this module remains importable without the full config surface loaded.
    """
    from .config import SimulationConfig

    if cfg is None:
        raise ValueError("cfg (configuration) is required")

    if isinstance(cfg, dict):
        return SimulationConfig.from_json_dict(cfg)
    if isinstance(cfg, str):
        return SimulationConfig.from_json_file(cfg)
    if isinstance(cfg, SimulationConfig):
        return cfg
    raise TypeError(
        f"cfg must be SimulationConfig, dict, or str (file path), got {type(cfg).__name__}"
    )


# ---------------------------------------------------------------------------
# Stage 2: derive a "full_json" representation (for JSON-mode solves)
# ---------------------------------------------------------------------------


def build_full_json(cfg) -> Optional[dict]:
    """Return a merged full JSON config for the solver, or None when not available.

    Sources, in order:
    1. ``cfg.extras['_full_json_config']`` – present when the user loaded a JSON file;
       it is merged with any later Python overrides.
    2. ``cfg.to_dict()`` – only used when a geometry block is present (i.e. JSON mode
       is truly feasible). Materials dicts are promoted to the array form expected
       by the C++ JSON schema.
    """
    if hasattr(cfg, "validate"):
        cfg.validate()

    if hasattr(cfg, "extras") and cfg.extras and "_full_json_config" in cfg.extras:
        return merge_user_cfg_over_full_json(cfg, cfg.extras["_full_json_config"])

    try:
        cfg_dict = cfg.to_dict()
    except Exception:
        return None

    if not (isinstance(cfg_dict, dict) and "geometry" in cfg_dict):
        return None

    full_json = cfg_dict
    _promote_materials_to_list(full_json, infer_type_from_pde=False)
    if (
        hasattr(cfg, "extras")
        and cfg.extras
        and "root_path" not in full_json
        and "_root_path" in cfg.extras
    ):
        full_json["root_path"] = cfg.extras["_root_path"]
    return full_json


# ---------------------------------------------------------------------------
# Stage 3: runtime options (requested fields / strict / fallback)
# ---------------------------------------------------------------------------


def resolve_runtime_options(
    cfg,
    full_json: Optional[dict],
    sampled_vtu_fallback: Optional[bool],
) -> RuntimeOptions:
    """Collect runtime flags from ``cfg.output`` (preferred) or ``full_json.output``.

    ``sampled_vtu_fallback`` (the ``solve()`` parameter) forces the mode when set:
    ``True`` → ``always``, ``False`` → ``never``.
    """
    runtime: Dict[str, Any] = {}

    output_obj = getattr(cfg, "output", None)
    if output_obj is not None and hasattr(output_obj, "runtime_options"):
        try:
            runtime = dict(output_obj.runtime_options())
        except Exception:
            runtime = {}

    if not runtime and isinstance(full_json, dict):
        out = full_json.get("output")
        if isinstance(out, dict):
            if isinstance(out.get("result"), dict):
                runtime["result"] = dict(out["result"])
            if isinstance(out.get("fallback"), dict):
                runtime["fallback"] = dict(out["fallback"])

    result_cfg = runtime.get("result") if isinstance(runtime.get("result"), dict) else {}
    fallback_cfg = runtime.get("fallback") if isinstance(runtime.get("fallback"), dict) else {}

    requested_fields = result_cfg.get("fields")
    if requested_fields is not None:
        requested_fields = [str(x) for x in requested_fields]

    opts = RuntimeOptions(
        requested_fields=requested_fields,
        strict=bool(result_cfg.get("strict", False)),
        fallback_mode=str(fallback_cfg.get("sampled_vtu", "never")).strip().lower(),
        temp_storage=str(fallback_cfg.get("temp_storage", "ram")).strip().lower(),
        keep_temp_files=bool(fallback_cfg.get("keep_temp_files", False)),
    )

    if sampled_vtu_fallback is True:
        opts.fallback_mode = "always"
    elif sampled_vtu_fallback is False:
        opts.fallback_mode = "never"

    return opts


# ---------------------------------------------------------------------------
# Stage 4: mesh input normalization
# ---------------------------------------------------------------------------


def normalize_mesh_inputs(
    vertices,
    cells,
    full_json: Optional[dict],
    dtype,
) -> NormalizedInputs:
    """Resolve execution mode (JSON vs array) and normalize vertices/cells to NumPy."""
    use_json_mode = (
        full_json is not None
        and "geometry" in full_json
        and (vertices is None or cells is None)
    )

    if vertices is not None and cells is not None:
        V_np, v_backend = T.as_numpy(vertices, dtype=dtype)
        C_np, _ = T.as_numpy(cells, dtype=np.int32)
        C_np = _ensure_i32(C_np)
        return NormalizedInputs(V_np=V_np, C_np=C_np, v_backend=v_backend, use_json_mode=use_json_mode)

    if not use_json_mode:
        raise ValueError(
            "Either provide vertices/cells arrays, or use JSON config with geometry (mesh files)"
        )
    return NormalizedInputs(V_np=None, C_np=None, v_backend="numpy", use_json_mode=True)


# ---------------------------------------------------------------------------
# Stage 5: build C++ solver
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Stage 6: configure solver (settings + mesh load)
# ---------------------------------------------------------------------------


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


def _configure_json_mode(solver, full_json: dict, cfg) -> SolverConfigContext:
    processed = process_json_config(full_json, cfg)
    processed.pop("common", None)
    processed = clean_json_for_cpp(processed)
    _apply_settings_json(solver, json.dumps(processed))

    if hasattr(solver, "load_mesh_from_settings"):
        solver.load_mesh_from_settings()
    else:
        raise RuntimeError("JSON mode requires load_mesh_from_settings() method")
    return SolverConfigContext(settings_dict=None, bc={}, use_json_mode=True)


def _configure_array_mode(solver, cfg) -> SolverConfigContext:
    if hasattr(cfg, "to_dict"):
        settings_dict = cfg.to_dict()
    elif hasattr(cfg, "to_json_dict"):
        settings_dict = cfg.to_json_dict()
    else:
        raise TypeError("cfg must provide to_dict() or to_json_dict() for non-JSON mode.")

    bc_raw = getattr(cfg, "boundary_conditions", {}) or {}
    bc = bc_raw.to_dict() if hasattr(bc_raw, "to_dict") else (bc_raw if isinstance(bc_raw, dict) else {})

    if "geometry" not in settings_dict:
        settings_dict["geometry"] = [
            {"type": "ground", "height": 0.0, "enabled": True, "is_obstacle": False}
        ]

    _promote_materials_to_list(settings_dict, infer_type_from_pde=True)

    if bc:
        settings_dict.setdefault("boundary_conditions", {})
        settings_dict["boundary_conditions"].update(bc)

    _apply_settings_json(solver, json.dumps(settings_dict))
    return SolverConfigContext(settings_dict=settings_dict, bc=bc, use_json_mode=False)


def _apply_mesh_array_mode(solver, inputs: NormalizedInputs) -> None:
    V_np, C_np = inputs.V_np, inputs.C_np
    for name in ("set_mesh", "set_mesh_data", "load_mesh_from_points"):
        if hasattr(solver, name):
            fn = getattr(solver, name)
            try:
                fn(V_np, C_np)
                return
            except TypeError:
                try:
                    fn(points=V_np, cells=C_np)
                    return
                except Exception:
                    pass
    raise RuntimeError("No mesh setter found (set_mesh / set_mesh_data / load_mesh_from_points).")


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


def configure_solver(solver, cfg, full_json: Optional[dict], inputs: NormalizedInputs) -> SolverConfigContext:
    """Set settings, load/attach mesh. Returns context used by later retouch stages."""
    if inputs.use_json_mode:
        return _configure_json_mode(solver, full_json, cfg)

    ctx = _configure_array_mode(solver, cfg)
    _apply_mesh_array_mode(solver, inputs)
    _retouch_bc_after_mesh(solver, ctx)
    return ctx


# ---------------------------------------------------------------------------
# Stage 7: sidesets + BC retouch
# ---------------------------------------------------------------------------


def apply_sidesets(
    solver,
    sidesets_func: Optional[Callable],
    ctx: SolverConfigContext,
) -> None:
    """Attach user-supplied boundary IDs (array mode) and retouch BC after."""
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

    if not ctx.use_json_mode and ctx.bc and ctx.settings_dict is not None:
        try:
            ctx.settings_dict.setdefault("boundary_conditions", {})
            ctx.settings_dict["boundary_conditions"].update(ctx.bc)
            _apply_settings_json(solver, json.dumps(ctx.settings_dict))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Stage 8: assemble + run
# ---------------------------------------------------------------------------


def _resolve_log_level(full_json: Optional[dict]) -> int:
    if not (full_json and "output" in full_json and "log" in full_json["output"]):
        return 2
    log_cfg = full_json["output"]["log"]
    # Keep terminal logging aligned with the Python-side OutputLog default.
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
    """Run assemble + solve. Returns raw solver return value (may be dict/tuple/None)."""
    if hasattr(solver, "build_basis"):
        solver.build_basis()
    if hasattr(solver, "assemble"):
        solver.assemble()

    name = _first_attr(solver, "solve", "run")
    if not name:
        raise RuntimeError("No solver entry point found (solve / run).")

    log_level = _resolve_log_level(full_json)
    try:
        return getattr(solver, name)(log_level=log_level)
    except TypeError:
        return getattr(solver, name)()


# ---------------------------------------------------------------------------
# Stage 9: extract native outputs from solver return
# ---------------------------------------------------------------------------


def _query_first(solver, names: Sequence[str], *, as_int32: bool = False):
    for n in names:
        if hasattr(solver, n):
            try:
                val = np.asarray(getattr(solver, n)())
                if as_int32:
                    val = val.astype(np.int32, copy=False)
                return val
            except Exception:
                continue
    return None


def _write_array_field(fields: dict, meta: dict, value, *, key: str) -> None:
    fields[key] = np.asarray(value)


def _write_scalar_or_array(fields: dict, meta: dict, value, *, key: str) -> None:
    if isinstance(value, (int, float)):
        meta[key] = float(value)
    else:
        fields[key] = np.asarray(value)


def _merge_dict_into_meta(fields: dict, meta: dict, value, *, key: Optional[str] = None) -> None:
    if isinstance(value, dict):
        meta.update(value)


# Each entry probes a group of solver method names in order; the first method
# that exists and returns a non-None value "wins" and its result is handed to
# the handler. Exceptions inside a probe are swallowed to preserve the legacy
# best-effort behavior (solvers may expose broken getters on some code paths).
_FIELD_PROBES: Tuple[Tuple[Tuple[str, ...], Callable[..., None], Dict[str, Any]], ...] = (
    (("get_stress", "get_cauchy_stress", "stress"), _write_array_field,    {"key": "stress"}),
    (("get_strain", "strain"),                      _write_array_field,    {"key": "strain"}),
    (("get_energy", "energy", "total_energy"),      _write_scalar_or_array, {"key": "energy"}),
    (("get_pressure", "pressure"),                  _write_array_field,    {"key": "p"}),
    (("get_velocity", "velocity"),                  _write_array_field,    {"key": "v"}),
    (("get_stats", "stats", "get_log"),             _merge_dict_into_meta,  {}),
)


def _extract_additional_fields(solver, fields: dict, meta: dict) -> None:
    """Pull stress/strain/energy/pressure/velocity/stats off solver when available.

    Driven by the ``_FIELD_PROBES`` table so adding a new field is a one-line
    change. Preserves the historical semantics:

    - any probe that raises is silently skipped (try the next name in the group)
    - the first non-None value in a group wins, even if the handler decides it's
      unusable (e.g. get_stats returning something that isn't a dict)
    """
    for probe_names, handler, handler_kwargs in _FIELD_PROBES:
        for name in probe_names:
            if not hasattr(solver, name):
                continue
            try:
                value = getattr(solver, name)()
                if value is None:
                    continue
                handler(fields, meta, value, **handler_kwargs)
                break
            except Exception:
                continue


def _outputs_from_bundle(ret: dict, solver) -> NativeOutputs:
    fields: Dict[str, Any] = {"u": np.asarray(ret["u"])}
    if ret.get("p") is not None:
        p = np.asarray(ret["p"])
        if p.size > 0:
            fields["p"] = p
    for key in ("stress", "strain", "v"):
        if ret.get(key) is not None:
            val = np.asarray(ret[key])
            if val.size > 0:
                fields[key] = val

    meta: Dict[str, Any] = {"solver_type": type(solver).__name__}
    if ret.get("energy") is not None:
        e = ret["energy"]
        if isinstance(e, (int, float)):
            meta["energy"] = float(e)
        else:
            val = np.asarray(e)
            if val.size > 0:
                fields["energy"] = val
    if ret.get("meta") is not None and isinstance(ret["meta"], dict):
        meta.update(ret["meta"])

    return NativeOutputs(
        vertices=np.asarray(ret["vertices"]),
        cells=np.asarray(ret["cells"], dtype=np.int32),
        fields=fields,
        meta=meta,
    )


def _resolve_vertices(solver, inputs: NormalizedInputs) -> np.ndarray:
    pts = _query_first(solver, ("get_vertices", "get_points"))
    if pts is not None:
        return pts
    return inputs.V_np if inputs.V_np is not None else np.array([])


def _resolve_cells(solver, inputs: NormalizedInputs) -> np.ndarray:
    cells = _query_first(solver, ("get_elements", "get_cells"))
    if cells is not None:
        return cells
    if inputs.C_np is not None:
        return inputs.C_np
    return np.array([], dtype=np.int32)


def _outputs_from_tuple(ret, solver, inputs: NormalizedInputs) -> NativeOutputs:
    meta: Dict[str, Any] = {"solver_type": type(solver).__name__}
    fields: Dict[str, Any] = {"u": np.asarray(ret[0])}
    if len(ret) >= 2 and ret[1] is not None:
        try:
            fields["p"] = np.asarray(ret[1])
        except Exception:
            pass
    _extract_additional_fields(solver, fields, meta)
    return NativeOutputs(
        vertices=_resolve_vertices(solver, inputs),
        cells=_resolve_cells(solver, inputs),
        fields=fields,
        meta=meta,
    )


def _outputs_from_sampled_getter(solver, inputs: NormalizedInputs) -> Optional[NativeOutputs]:
    if not hasattr(solver, "get_sampled_solution"):
        return None
    out = solver.get_sampled_solution()
    if not (isinstance(out, (list, tuple)) and len(out) >= 5):
        return None

    meta: Dict[str, Any] = {"solver_type": type(solver).__name__}
    fields: Dict[str, Any] = {"u": np.asarray(out[4])}
    _extract_additional_fields(solver, fields, meta)
    return NativeOutputs(
        vertices=np.asarray(out[0]),
        cells=_resolve_cells(solver, inputs),
        fields=fields,
        meta=meta,
    )


def _outputs_from_direct_getters(solver, inputs: NormalizedInputs) -> NativeOutputs:
    u = None
    for name in ("get_solution", "get_displacement", "get_u"):
        if hasattr(solver, name):
            u = np.asarray(getattr(solver, name)())
            break
    if u is None:
        raise RuntimeError(
            "Failed to retrieve solution: no known getters (sampled or direct)."
        )

    meta: Dict[str, Any] = {"solver_type": type(solver).__name__}
    fields: Dict[str, Any] = {"u": u}
    _extract_additional_fields(solver, fields, meta)

    return NativeOutputs(
        vertices=_resolve_vertices(solver, inputs),
        cells=_resolve_cells(solver, inputs),
        fields=fields,
        meta=meta,
    )


def extract_native_outputs(ret, solver, inputs: NormalizedInputs) -> NativeOutputs:
    """Pick the right extraction strategy based on the solver's return shape.

    Strategies, in order:
    1. ``ret`` is a ``_result_bundle`` dict       → use embedded mesh + fields as-is.
    2. ``ret`` is a (sol, pressure) tuple/list    → combine with solver getters.
    3. ``solver.get_sampled_solution()`` works    → sampled mesh + solution.
    4. ``solver.get_solution()`` / equivalents    → direct on original mesh.
    """
    if isinstance(ret, dict) and ret.get("_result_bundle") and "vertices" in ret and "u" in ret:
        return _outputs_from_bundle(ret, solver)

    if isinstance(ret, (tuple, list)) and len(ret) >= 1:
        return _outputs_from_tuple(ret, solver, inputs)

    sampled = _outputs_from_sampled_getter(solver, inputs)
    if sampled is not None:
        return sampled

    return _outputs_from_direct_getters(solver, inputs)


# ---------------------------------------------------------------------------
# Stage 10: sampled VTU fallback
# ---------------------------------------------------------------------------


def _extract_meshio_array(mesh, name: str) -> Tuple[Optional[np.ndarray], Optional[str]]:
    point_data = getattr(mesh, "point_data", {}) or {}
    if name in point_data:
        return np.asarray(point_data[name]), "point"
    cell_data = getattr(mesh, "cell_data", {}) or {}
    if name in cell_data:
        raw = cell_data[name]
        if isinstance(raw, list) and raw:
            return np.asarray(raw[0]), "cell"
        return np.asarray(raw), "cell"
    return None, None


def _reconstruct_sampled_cauchy_stress(mesh) -> Tuple[Optional[np.ndarray], Optional[str]]:
    rows: List[np.ndarray] = []
    location: Optional[str] = None
    for idx in range(1, 4):
        arr, loc = _extract_meshio_array(mesh, f"cauchy_stess_{idx}")
        if arr is None:
            break
        arr = np.asarray(arr)
        if arr.ndim != 2:
            return None, None
        rows.append(arr)
        location = loc

    if len(rows) == 2:
        r1, r2 = rows
        if r1.shape[1] < 2 or r2.shape[1] < 2:
            return None, None
        sxx = r1[:, 0]
        syy = r2[:, 1]
        sxy = 0.5 * (r1[:, 1] + r2[:, 0])
        return np.column_stack([sxx, syy, sxy]), location

    if len(rows) == 3:
        r1, r2, r3 = rows
        if min(r.shape[1] for r in rows) < 3:
            return None, None
        sxx = r1[:, 0]
        syy = r2[:, 1]
        szz = r3[:, 2]
        sxy = 0.5 * (r1[:, 1] + r2[:, 0])
        syz = 0.5 * (r2[:, 2] + r3[:, 1])
        szx = 0.5 * (r1[:, 2] + r3[:, 0])
        return np.column_stack([sxx, syy, szz, sxy, syz, szx]), location

    return None, None


def _meshio_primary_cells(mesh) -> np.ndarray:
    cells = getattr(mesh, "cells", None) or []
    for block in cells:
        data = getattr(block, "data", None)
        if data is None:
            continue
        arr = np.asarray(data)
        if arr.size > 0:
            return _ensure_i32(arr)
    return np.empty((0, 0), dtype=np.int32)


def _flatten_optional_scalar(arr) -> np.ndarray:
    out = np.asarray(arr)
    if out.ndim == 2 and out.shape[1] == 1:
        out = out[:, 0]
    return out


def _history_field_source_label(history) -> str:
    source = getattr(history, "source", "solver.solution_frames")
    if source == "exported_vtu_sequence":
        return "exported_vtu:last_frame"
    return "history:last_frame"


def _populate_result_from_history(result: Result) -> Result:
    history = result.history
    if not history.available:
        return result

    source_label = _history_field_source_label(history)

    stress = np.asarray(history.stress)
    if stress.size > 0 and stress.ndim >= 3:
        result.set_sampled_field("stress", stress[-1])
        result.meta["stress_source"] = source_label
        result.meta["stress_location"] = "point"

    vm = np.asarray(history.vm)
    if vm.size > 0 and vm.ndim >= 2:
        result.set_sampled_field("von_mises", vm[-1])
        result.meta["von_mises_source"] = source_label
        result.meta["von_mises_location"] = "point"

    vm_avg = np.asarray(history.vm_avg)
    if vm_avg.size > 0 and vm_avg.ndim >= 2:
        result.set_sampled_field("von_mises_avg", vm_avg[-1])
        result.meta["von_mises_avg_source"] = source_label
        result.meta["von_mises_avg_location"] = "point"

    body_ids = np.asarray(history.body_ids)
    if body_ids.size > 0:
        result.set_sampled_field("body_ids", body_ids.astype(np.int32, copy=False))
        result.meta["body_ids_source"] = (
            "exported_vtu" if source_label.startswith("exported_vtu") else "history"
        )
        result.meta["body_ids_location"] = "point"

    return result


def _resolve_output_directory(full_json: Optional[dict]) -> Optional[Path]:
    if not isinstance(full_json, dict):
        return None
    output = full_json.get("output")
    if not isinstance(output, dict):
        return None
    directory = output.get("directory")
    if not isinstance(directory, str) or not directory.strip():
        return None
    return Path(directory).expanduser()


def _resolve_exported_vtu_paths(full_json: Optional[dict]) -> List[Path]:
    output_dir = _resolve_output_directory(full_json)
    if output_dir is None:
        return []

    output = full_json.get("output") if isinstance(full_json, dict) else None
    if not isinstance(output, dict):
        return []

    paraview = output.get("paraview")
    advanced = output.get("advanced")
    if not isinstance(paraview, dict):
        return []

    file_name = str(paraview.get("file_name", "") or "").strip()
    if not file_name:
        return []

    # ``OutputAdvanced.to_dict()`` omits ``save_time_sequence`` when it is the
    # default ``True``. Treat a missing key as enabled so exported
    # ``timestep_prefix*.vtu`` sequences can still be discovered and rebuilt
    # into ``result.history``.
    if isinstance(advanced, dict) and bool(advanced.get("save_time_sequence", True)):
        prefix = str(advanced.get("timestep_prefix", "") or "").strip()
        if not prefix:
            return []
        paths = list(output_dir.glob(f"{prefix}*.vtu"))
        if not paths:
            return []

        def _step_key(path: Path):
            step = _extract_history_step_index(path.name)
            return (step is None, step if step is not None else path.name)

        return sorted(paths, key=_step_key)

    paraview_path = Path(file_name).expanduser()
    if paraview_path.is_absolute():
        final_path = paraview_path.with_suffix(".vtu")
    else:
        final_path = output_dir / f"{paraview_path.stem}.vtu"
    return [final_path] if final_path.is_file() else []


def _read_meshio_file(path: Path):
    try:
        meshio_mod = importlib.import_module("meshio")
    except Exception as exc:
        warnings.warn(
            f"VTU readback requested, but meshio is unavailable: {exc}",
            RuntimeWarning,
        )
        return None
    try:
        return meshio_mod.read(str(path))
    except Exception as exc:
        warnings.warn(
            f"Failed to read exported VTU {path}: {exc}",
            RuntimeWarning,
        )
        return None


def _frame_from_exported_vtu(path: Path, mesh) -> Dict[str, Any]:
    solution, _ = _extract_meshio_array(mesh, "solution")
    if solution is None:
        solution = np.empty((0, 0))

    pressure, _ = _extract_meshio_array(mesh, "pressure")
    if pressure is None:
        pressure = np.empty((0, 0))

    scalar_value, _ = _extract_meshio_array(mesh, "von_mises")
    if scalar_value is None:
        scalar_value = np.empty((0, 0))

    scalar_value_avg, _ = _extract_meshio_array(mesh, "von_mises_avg")
    if scalar_value_avg is None:
        scalar_value_avg = np.empty((0, 0))

    body_ids, _ = _extract_meshio_array(mesh, "body_ids")
    if body_ids is None:
        body_ids = np.empty((0,), dtype=np.int32)
    else:
        body_ids = _flatten_optional_scalar(body_ids).astype(np.int32, copy=False)

    tensor_value, _ = _reconstruct_sampled_cauchy_stress(mesh)
    if tensor_value is None:
        tensor_value = np.empty((0, 0))

    return {
        "name": str(path),
        "points": np.asarray(getattr(mesh, "points", np.empty((0, 0)))),
        "connectivity": _meshio_primary_cells(mesh),
        "solution": np.asarray(solution),
        "pressure": np.asarray(pressure),
        "scalar_value": np.asarray(scalar_value),
        "scalar_value_avg": np.asarray(scalar_value_avg),
        "tensor_value": np.asarray(tensor_value),
        "body_ids": np.asarray(body_ids),
    }


def _collect_history_from_exported_vtus(full_json: Optional[dict]):
    from .result import HistoryView

    paths = _resolve_exported_vtu_paths(full_json)
    if not paths:
        return HistoryView()

    frames = []
    for path in paths:
        mesh = _read_meshio_file(path)
        if mesh is None:
            return HistoryView()
        frames.append(_frame_from_exported_vtu(path, mesh))

    times = _infer_history_times(frames, full_json)
    history = HistoryView(frames=frames, times=times)
    history.raw_frame_count = len(frames)
    history.deduped_frame_count = len(frames)
    history.dropped_duplicate_frames = 0
    history.source = "exported_vtu_sequence"
    return history


def apply_sampled_vtu_fallback(
    result: Result,
    *,
    solver,
    native: NativeOutputs,
    full_json: Optional[dict],
    runtime: RuntimeOptions,
) -> Result:
    """Fill sampled fields from history, or from user-exported VTUs as backup.

    The old temporary-``export_vtu()`` path is intentionally gone. We now only
    use two data sources:

    1. in-memory ``result.history`` populated by ``solver.solution_frames``
    2. user-exported ``impact_step_*.vtu`` files, when on-disk export is enabled
    """
    result = _populate_result_from_history(result)
    if result.history.available:
        return result

    exported_history = _collect_history_from_exported_vtus(full_json)
    if not exported_history.available:
        return result

    result.history = exported_history
    result.meta["sampled_vtu_fallback"] = True
    result.meta["sampled_vtu_fallback_mode"] = "exported_files"
    result.meta["sampled_vtu_point_count"] = int(
        getattr(exported_history, "points", np.empty((0, 0))).shape[0]
    )
    result = _populate_result_from_history(result)
    return result


# ---------------------------------------------------------------------------
# Stage 11: finalize (requested-fields / strict checks)
# ---------------------------------------------------------------------------


def finalize_result(result: Result, runtime: RuntimeOptions) -> Result:
    if runtime.requested_fields:
        result.meta["requested_fields"] = list(runtime.requested_fields)
    missing: List[str] = []
    if runtime.requested_fields:
        missing = [
            name for name in runtime.requested_fields if not _field_available(result, name)
        ]
        if missing:
            result.meta["missing_requested_fields"] = list(missing)
    if runtime.strict and missing:
        raise RuntimeError(
            "Requested result fields are unavailable after extraction/fallback: "
            + ", ".join(missing)
        )
    return result


# ---------------------------------------------------------------------------
# Top-level pipeline wrapper (keeps solve.py thin)
# ---------------------------------------------------------------------------


def _collect_solver_history(solver, full_json: Optional[dict]):
    """Pull PolyFEM's in-memory per-timestep frames off the solver.

    Returns a ``HistoryView`` populated from ``solver.solution_frames`` if the
    C++ binding exposes that attribute (added in the Round-1 nanobind change),
    otherwise an empty ``HistoryView``. This is strictly additive — callers
    that don't care about history simply ignore it.

    Time values for each frame are derived from ``full_json["time"]`` when
    possible. We prefer the saved frame names (``...step_17.vtu`` -> step 17)
    over raw frame order because PolyFEM may emit duplicate initial frames
    (e.g. nonlinear transient solves often save ``step_0`` twice). We also
    collapse consecutive duplicate step indices, keeping the last frame in each
    run so ``result.history`` reflects user-visible timesteps rather than the
    solver's internal bookkeeping saves. If we can't infer step numbers from
    names, we fall back to frame order.
    """
    from .result import HistoryView

    raw = getattr(solver, "solution_frames", None)
    if raw is None:
        return HistoryView()
    try:
        frames = list(raw)
    except TypeError:
        return HistoryView()
    if not frames:
        return HistoryView()

    raw_count = len(frames)
    frames = _dedupe_history_frames(frames)
    times = _infer_history_times(frames, full_json)
    history = HistoryView(frames=frames, times=times)
    history.raw_frame_count = raw_count
    history.deduped_frame_count = len(frames)
    history.dropped_duplicate_frames = raw_count - len(frames)
    history.source = "solver.solution_frames"
    return history


def _extract_history_step_index(name: str) -> Optional[int]:
    """Parse ``impact_step_12.vtu``-style names emitted by PolyFEM."""
    if not isinstance(name, str) or not name.strip():
        return None
    stem = Path(name).stem
    match = re.search(r"(?:^|_)step_(\d+)$", stem)
    if match is None:
        return None
    return int(match.group(1))


def _dedupe_history_frames(frames):
    """Collapse consecutive duplicate saved step indices.

    PolyFEM's nonlinear transient path currently saves the initial state twice:
    once in ``init_solve()`` and again when entering the nonlinear time loop.
    Those frames share the same ``...step_0.vtu`` name. For the public
    ``result.history`` API we keep the last frame in each consecutive run so
    callers see one frame per saved timestep.
    """
    if len(frames) < 2:
        return list(frames)

    step_indices = []
    for frame in frames:
        if not isinstance(frame, dict):
            return list(frames)
        step = _extract_history_step_index(frame.get("name"))
        if step is None:
            return list(frames)
        step_indices.append(step)

    deduped = []
    run_start = 0
    for i in range(1, len(frames) + 1):
        if i < len(frames) and step_indices[i] == step_indices[run_start]:
            continue
        deduped.append(frames[i - 1])  # keep the last frame in the run
        run_start = i
    return deduped


def _infer_history_times(frames, full_json: Optional[dict]):
    """Best-effort simulation times for history frames.

    Preference order:
    1. Parse the saved step index from each frame name and map ``step -> t0 + step*dt``.
    2. Fall back to raw frame order when ``dt`` is known but names are not parseable.
    3. Return ``None`` so ``HistoryView`` uses default indices.
    """
    if not frames or not isinstance(full_json, dict):
        return None

    tcfg = full_json.get("time") or {}
    if not isinstance(tcfg, dict):
        return None

    dt = float(tcfg.get("dt", 0.0) or 0.0)
    if dt <= 0.0:
        return None
    t0 = float(tcfg.get("t0", 0.0) or 0.0)

    step_indices = []
    for frame in frames:
        step = _extract_history_step_index(frame.get("name") if isinstance(frame, dict) else None)
        if step is None:
            step_indices = []
            break
        step_indices.append(step)

    if step_indices:
        return [t0 + step * dt for step in step_indices]

    return [t0 + i * dt for i in range(len(frames))]


def run_pipeline(
    vertices=None,
    cells=None,
    cfg=None,
    sidesets_func: Optional[Callable] = None,
    dtype=None,
    sampled_vtu_fallback: Optional[bool] = None,
) -> Result:
    """Drive the full solve pipeline. External contract matches ``api.solve.solve``."""
    cfg = normalize_cfg(cfg)
    full_json = build_full_json(cfg)
    runtime = resolve_runtime_options(cfg, full_json, sampled_vtu_fallback)
    inputs = normalize_mesh_inputs(vertices, cells, full_json, dtype)

    solver = build_solver()
    ctx = configure_solver(solver, cfg, full_json, inputs)
    apply_sidesets(solver, sidesets_func, ctx)

    ret = run_solver_stage(solver, full_json)
    native = extract_native_outputs(ret, solver, inputs)
    history = _collect_solver_history(solver, full_json)

    result = Result(
        inputs.v_backend,
        native.vertices,
        native.cells,
        native.fields,
        meta=native.meta,
        history=history,
    )
    result = apply_sampled_vtu_fallback(
        result,
        solver=solver,
        native=native,
        full_json=full_json,
        runtime=runtime,
    )
    final_history = result.history
    if final_history.available:
        result.meta["history_frames"] = len(final_history)
        raw_history_frames = getattr(final_history, "raw_frame_count", len(final_history))
        result.meta["history_raw_frames"] = int(raw_history_frames)
        if raw_history_frames != len(final_history):
            result.meta["history_dropped_duplicate_frames"] = int(
                raw_history_frames - len(final_history)
            )
        result.meta["history_source"] = getattr(
            final_history, "source", "solver.solution_frames"
        )
    return finalize_result(result, runtime)
