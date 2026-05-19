"""Output extraction and result finalization for the forward solve pipeline.

This module owns raw solver return parsing, history collection, sampled VTU
readback, and requested-field validation. Config and mesh semantics live in
``_solve_contract.py``; backend construction and execution live in
``_solve_backend.py``.
"""

from __future__ import annotations

import importlib
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .result import Result


@dataclass
class NativeOutputs:
    """Native (i.e. direct from solver) fields + mesh that become a Result."""

    vertices: np.ndarray
    cells: np.ndarray
    fields: Dict[str, np.ndarray]
    meta: Dict[str, Any]



# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


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

    if runtime.fallback_mode == "never":
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
# History extraction helpers
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


__all__ = [
    "NativeOutputs",
    "apply_sampled_vtu_fallback",
    "extract_native_outputs",
    "finalize_result",
]
