"""Solve pipeline stages.

This module breaks ``solve()`` into small, independently testable stages plus a
few typed intermediate structures. The public entry point in ``solve.py`` keeps
its signature and external behavior; it only delegates to the functions here.

This is an internal implementation module. User code should call
``polyfempy.api.solve(...)`` rather than importing pipeline stages directly.
Tests import the stages to protect behavior.

Pipeline (linear):

    normalize_cfg          -> backend-shaped dict
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

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ._solve_contract import (
    MeshSource,
    build_full_json as _contract_build_full_json,
    clean_json_for_cpp as _contract_clean_json_for_cpp,
    cfg_array_mesh_payload,
    choose_mesh_source,
    merge_user_cfg_over_full_json as _contract_merge_user_cfg_over_full_json,
    normalize_config,
    process_json_config as _contract_process_json_config,
    prepare_canonical_solve_input,
)
from ._solve_backend import (
    SolverConfigContext,
    apply_sidesets,
    build_solver,
    configure_solver,
    run_solver_stage,
)
from ._solve_outputs import (
    NativeOutputs,
    _collect_solver_history,
    _dedupe_history_frames,
    _extract_additional_fields,
    _extract_history_step_index,
    _extract_meshio_array,
    _field_available,
    _infer_history_times,
    _reconstruct_sampled_cauchy_stress,
    apply_sampled_vtu_fallback,
    extract_native_outputs,
    finalize_result,
)
from .result import Result


# ---------------------------------------------------------------------------
# Intermediate data structures
# ---------------------------------------------------------------------------


@dataclass
class RuntimeOptions:
    """Runtime output/fallback knobs extracted from payload output settings."""

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
    body_ids_np: Optional[np.ndarray]
    boundary_ids_np: Optional[np.ndarray]
    v_backend: str
    use_json_mode: bool
    mesh_source: str = "array"


# ---------------------------------------------------------------------------
# Small helpers (no solver state)
# ---------------------------------------------------------------------------


def _cfg_array_mesh_payload(cfg) -> Optional[Dict[str, Any]]:
    return cfg_array_mesh_payload(cfg)


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
    return _contract_process_json_config(full_json, cfg)


def clean_json_for_cpp(obj, path: str = ""):
    """Recursively drop ``None`` values in a JSON-like object, keeping solver blocks intact."""
    return _contract_clean_json_for_cpp(obj)


def merge_user_cfg_over_full_json(cfg, full_json) -> dict:
    """Overlay Python-side payload edits on top of the original full JSON."""
    return _contract_merge_user_cfg_over_full_json(cfg, full_json)


# ---------------------------------------------------------------------------
# Stage 1: normalize cfg into a backend-shaped dict
# ---------------------------------------------------------------------------


def normalize_cfg(cfg):
    """Accept dict / path / generated object and return a backend-shaped dict."""
    return normalize_config(cfg)


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
    return _contract_build_full_json(cfg)


# ---------------------------------------------------------------------------
# Stage 3: runtime options (requested fields / strict / fallback)
# ---------------------------------------------------------------------------


def resolve_runtime_options(
    cfg,
    full_json: Optional[dict],
    sampled_vtu_fallback: Optional[bool],
) -> RuntimeOptions:
    """Collect runtime flags from dict output blocks or ``full_json.output``.

    ``sampled_vtu_fallback`` (the ``solve()`` parameter) forces the mode when set:
    ``True`` → ``always``, ``False`` → ``never``.
    """
    runtime: Dict[str, Any] = {}

    output_obj = None if isinstance(cfg, dict) else getattr(cfg, "output", None)
    if output_obj is not None and hasattr(output_obj, "runtime_options"):
        try:
            runtime = dict(output_obj.runtime_options())
        except Exception:
            runtime = {}

    def _read_runtime_from_output(out: Any) -> Dict[str, Any]:
        values: Dict[str, Any] = {}
        if isinstance(out, dict):
            if isinstance(out.get("result"), dict):
                values["result"] = dict(out["result"])
            if isinstance(out.get("fallback"), dict):
                values["fallback"] = dict(out["fallback"])
        return values

    if not runtime:
        out = cfg.get("output") if isinstance(cfg, dict) else None
        runtime = _read_runtime_from_output(out)

    if not runtime and isinstance(full_json, dict):
        runtime = _read_runtime_from_output(full_json.get("output"))

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
    cfg=None,
) -> NormalizedInputs:
    """Resolve execution mode (JSON vs array) and normalize vertices/cells to NumPy."""
    mesh_source = choose_mesh_source(
        vertices,
        cells,
        full_json,
        dtype=dtype,
        cfg=cfg,
    )
    use_json_mode = mesh_source.mode == "json"
    if not use_json_mode:
        return NormalizedInputs(
            V_np=mesh_source.vertices,
            C_np=mesh_source.cells,
            body_ids_np=mesh_source.body_ids,
            boundary_ids_np=mesh_source.boundary_ids,
            v_backend=mesh_source.v_backend,
            use_json_mode=False,
            mesh_source=mesh_source.mode,
        )
    return NormalizedInputs(
        V_np=None,
        C_np=None,
        body_ids_np=None,
        boundary_ids_np=None,
        v_backend="numpy",
        use_json_mode=True,
        mesh_source="json",
    )


def _inputs_from_mesh_source(mesh_source: MeshSource) -> NormalizedInputs:
    use_json_mode = mesh_source.mode == "json"
    return NormalizedInputs(
        V_np=None if use_json_mode else mesh_source.vertices,
        C_np=None if use_json_mode else mesh_source.cells,
        body_ids_np=None if use_json_mode else mesh_source.body_ids,
        boundary_ids_np=None if use_json_mode else mesh_source.boundary_ids,
        v_backend=mesh_source.v_backend,
        use_json_mode=use_json_mode,
        mesh_source=mesh_source.mode,
    )


# ---------------------------------------------------------------------------
# Top-level pipeline wrapper (keeps solve.py thin)
# ---------------------------------------------------------------------------

def run_pipeline(
    vertices=None,
    cells=None,
    cfg=None,
    sidesets_func: Optional[Callable] = None,
    dtype=None,
    sampled_vtu_fallback: Optional[bool] = None,
) -> Result:
    """Drive the full solve pipeline. External contract matches ``api.solve.solve``."""
    canonical = prepare_canonical_solve_input(
        vertices=vertices,
        cells=cells,
        cfg=cfg,
        dtype=dtype,
    )
    cfg = canonical.config
    full_json = canonical.full_json
    runtime = resolve_runtime_options(cfg, full_json, sampled_vtu_fallback)
    inputs = _inputs_from_mesh_source(canonical.mesh_source)

    solver = build_solver()
    ctx = configure_solver(
        solver,
        cfg,
        full_json,
        inputs,
        backend_settings=canonical.backend_settings,
    )
    apply_sidesets(solver, sidesets_func, ctx)

    ret = run_solver_stage(solver, full_json)
    native = extract_native_outputs(ret, solver, inputs)
    history = _collect_solver_history(solver, full_json)
    native.meta.setdefault("mesh_source", inputs.mesh_source)

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
