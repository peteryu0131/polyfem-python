"""Reusable reporting helpers for ``Result`` and history data.

This module formats and summarizes result objects. It does not run solvers and
does not mutate simulation configuration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np


__all__ = [
    "summarize_result",
    "format_result_summary",
    "summarize_history_bundle",
    "format_history_bundle_txt",
    "write_history_bundle_txt",
]


def _shape_tuple(value):
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(x) for x in shape)
    except TypeError:
        return tuple(shape)


def _shape_text(value) -> str:
    shape = _shape_tuple(value)
    if shape is not None:
        return str(shape)
    if isinstance(value, list):
        return f"list[{len(value)}]"
    return type(value).__name__


def _field_origin(result, name: str) -> str:
    if name in result.point_data:
        return "native:point_data"
    if name in result.cell_data:
        return "native:cell_data"
    if name in result.sampled_data:
        return "sampled"
    return "absent"


def _field_note(result, name: str, origin: str) -> Optional[str]:
    if origin == "absent":
        return None
    if name == "u" and origin.startswith("native"):
        return "native"
    source = result.meta.get(f"{name}_source")
    if source:
        return f"source={source}"
    return origin


def _format_values(values, *, fmt: str) -> str:
    return ", ".join(format(v, fmt) for v in values)


def _safe_max(value) -> float:
    arr = np.asarray(value)
    if arr.size == 0:
        return float("nan")
    return float(arr.max())


def _safe_abs_max(value) -> float:
    arr = np.asarray(value)
    if arr.size == 0:
        return float("nan")
    return float(np.abs(arr).max())


def _has_stacked_sample_axis(value) -> bool:
    shape = _shape_tuple(value)
    return shape is not None and len(shape) >= 2 and shape[1] > 0


def _as_config_dict(cfg) -> dict[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return dict(cfg)
    for name in ("to_full_json_dict", "to_json_dict", "to_dict"):
        if hasattr(cfg, name):
            try:
                out = getattr(cfg, name)()
            except Exception:
                continue
            if isinstance(out, dict):
                return out
    return {}


def _volume_selection_legend(cfg) -> dict[int, dict[str, Any]]:
    cfg_dict = _as_config_dict(cfg)
    geometry = cfg_dict.get("geometry")
    if not isinstance(geometry, list):
        return {}
    legend: dict[int, dict[str, Any]] = {}
    for i, entry in enumerate(geometry):
        if not isinstance(entry, dict):
            continue
        volume_selection = entry.get("volume_selection")
        if volume_selection is None:
            continue
        if isinstance(volume_selection, float) and volume_selection.is_integer():
            volume_selection = int(volume_selection)
        try:
            key = int(volume_selection)
        except (TypeError, ValueError):
            continue
        mesh = entry.get("mesh", "")
        legend[key] = {
            "geometry_index": i,
            "volume_selection": key,
            "mesh_stem": Path(str(mesh)).stem if mesh else "",
        }
    return legend


def _row_count(value) -> int:
    arr = np.asarray(value)
    if arr.ndim == 0:
        return int(arr.size)
    if arr.ndim >= 1:
        return int(arr.shape[0])
    return 0


def _vector_magnitude(value) -> np.ndarray:
    arr = np.asarray(value)
    if arr.size == 0:
        return np.empty((0,), dtype=np.float64)
    if arr.ndim == 1:
        return np.abs(arr.astype(np.float64, copy=False))
    return np.linalg.norm(arr.astype(np.float64, copy=False), axis=-1)


def _vm_stats(value) -> dict[str, float | int]:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return {"n": 0, "mean": float("nan"), "max": float("nan"), "p95": float("nan")}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "max": float(arr.max()),
        "p95": float(np.percentile(arr, 95)),
    }


def _step_metrics(u_step, vm_step, stress_step) -> dict[str, float | int]:
    u_mag = _vector_magnitude(u_step)
    vm_stats = _vm_stats(vm_step)
    stress_arr = np.asarray(stress_step)
    return {
        "n_points": int(max(_row_count(u_step), _row_count(vm_step), _row_count(stress_step))),
        "u_max": float(u_mag.max()) if u_mag.size > 0 else float("nan"),
        "vm_mean": float(vm_stats["mean"]),
        "vm_max": float(vm_stats["max"]),
        "vm_p95": float(vm_stats["p95"]),
        "stress_abs_max": _safe_abs_max(stress_arr),
    }


def summarize_history_bundle(result, *, cfg=None) -> dict[str, Any]:
    """Structured per-step summary similar to the older training-bundle export.

    The output is derived from ``result.history`` and can therefore work without
    VTU files when in-memory history is available.
    """
    history = result.history
    bundle: dict[str, Any] = {
        "available": bool(history),
        "history_source": result.meta.get("history_source", "unknown"),
        "fields": {
            "u": bool(np.asarray(history.u).size > 0),
            "von_mises": bool(np.asarray(history.vm).size > 0),
            "stress": bool(np.asarray(history.stress).size > 0),
        },
        "body_legend": {},
        "steps": [],
        "steps_by_body": [],
    }
    if not history.available:
        return bundle

    legend = _volume_selection_legend(cfg)
    body_ids = result.body_ids
    u_by_body = vm_by_body = stress_by_body = None
    if body_ids is not None:
        body_ids = np.asarray(body_ids)
        bundle["body_legend"] = {
            int(bid): legend.get(int(bid), {"body_id": int(bid)})
            for bid in np.unique(body_ids)
        }
        try:
            u_by_body = history.field_by_body("u", body_ids) if _has_stacked_sample_axis(history.u) else None
        except (TypeError, ValueError):
            u_by_body = None
        try:
            vm_by_body = history.field_by_body("vm", body_ids) if _has_stacked_sample_axis(history.vm) else None
        except (TypeError, ValueError):
            vm_by_body = None
        try:
            stress_by_body = (
                history.field_by_body("stress", body_ids)
                if _has_stacked_sample_axis(history.stress)
                else None
            )
        except (TypeError, ValueError):
            stress_by_body = None

    n_steps = len(history)
    times = history.times.tolist() if getattr(history, "times", None) is not None else list(range(n_steps))
    for i in range(n_steps):
        row = {
            "step": i,
            "time": float(times[i]),
            **_step_metrics(history.u[i], history.vm[i], history.stress[i]),
        }
        bundle["steps"].append(row)

        if body_ids is None or vm_by_body is None:
            continue
        for bid in sorted(vm_by_body):
            per_body = {
                "step": i,
                "time": float(times[i]),
                "body_id": int(bid),
                **legend.get(int(bid), {}),
                **_step_metrics(
                    u_by_body[bid][i] if u_by_body is not None else np.empty((0,)),
                    vm_by_body[bid][i],
                    stress_by_body[bid][i] if stress_by_body is not None else np.empty((0,)),
                ),
            }
            bundle["steps_by_body"].append(per_body)

    return bundle


def _format_scalar(value) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if value is None:
        return ""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if np.isnan(f):
        return "nan"
    return f"{f:.6g}"


def _format_tsv_rows(columns: list[str], rows: list[dict[str, Any]]) -> list[str]:
    lines = ["\t".join(columns)]
    for row in rows:
        lines.append("\t".join(_format_scalar(row.get(col, "")) for col in columns))
    return lines


def format_history_bundle_txt(result, *, cfg=None) -> str:
    """Human-readable TSV-style bundle for per-step result history."""
    bundle = summarize_history_bundle(result, cfg=cfg)
    if not bundle["available"]:
        return (
            "# PolyFEM History Bundle\n"
            "history_available: false\n"
            "note: result.history is empty\n"
        )

    lines = [
        "# PolyFEM History Bundle",
        f"history_available: true",
        f"history_source: {bundle['history_source']}",
        (
            "fields: "
            f"u={bundle['fields']['u']} "
            f"von_mises={bundle['fields']['von_mises']} "
            f"stress={bundle['fields']['stress']}"
        ),
        "",
        "[body_legend]",
    ]

    if bundle["body_legend"]:
        legend_cols = ["body_id", "geometry_index", "volume_selection", "mesh_stem"]
        legend_rows = []
        for body_id in sorted(bundle["body_legend"]):
            meta = dict(bundle["body_legend"][body_id])
            meta["body_id"] = body_id
            legend_rows.append(meta)
        lines.extend(_format_tsv_rows(legend_cols, legend_rows))
    else:
        lines.append("body_id\tgeometry_index\tvolume_selection\tmesh_stem")

    lines.extend(
        [
            "",
            "[steps]",
        ]
    )
    lines.extend(
        _format_tsv_rows(
            ["step", "time", "n_points", "u_max", "vm_mean", "vm_max", "vm_p95", "stress_abs_max"],
            bundle["steps"],
        )
    )

    lines.extend(
        [
            "",
            "[steps_by_body]",
        ]
    )
    lines.extend(
        _format_tsv_rows(
            [
                "step",
                "time",
                "body_id",
                "geometry_index",
                "volume_selection",
                "mesh_stem",
                "n_points",
                "u_max",
                "vm_mean",
                "vm_max",
                "vm_p95",
                "stress_abs_max",
            ],
            bundle["steps_by_body"],
        )
    )
    return "\n".join(lines) + "\n"


def write_history_bundle_txt(result, path, *, cfg=None) -> Path:
    """Write :func:`format_history_bundle_txt` to ``path`` and return it."""
    out_path = Path(path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(format_history_bundle_txt(result, cfg=cfg), encoding="utf-8")
    return out_path


def summarize_result(
    result,
    *,
    fields: Iterable[str] = ("u", "stress", "von_mises"),
    include_history: bool = True,
    include_body_stats: bool = True,
):
    """Build a structured, human-oriented summary of a solve result.

    The returned dict is intended for scripting/reporting, while
    :func:`format_result_summary` turns the same information into a compact
    multi-line string for CLI examples and experiments.
    """
    summary = {"fields": [], "history": None}

    for name in fields:
        arr = result.field(name)
        origin = _field_origin(result, name)
        summary["fields"].append(
            {
                "name": name,
                "available": arr is not None,
                "shape": _shape_tuple(arr),
                "shape_text": _shape_text(arr) if arr is not None else None,
                "origin": origin,
                "note": _field_note(result, name, origin),
            }
        )

    if not include_history:
        return summary

    history = result.history
    history_summary = {
        "available": bool(history),
        "frame_count": len(history),
        "u_shape": _shape_tuple(history.u),
        "u_shape_text": _shape_text(history.u),
        "vm_shape": _shape_tuple(history.vm),
        "vm_shape_text": _shape_text(history.vm),
        "stress_shape": _shape_tuple(history.stress),
        "stress_shape_text": _shape_text(history.stress),
        "times": history.times.tolist() if getattr(history, "times", None) is not None else [],
        "vm_max_by_body": {},
        "final_stress_by_body": {},
    }
    summary["history"] = history_summary

    if not history.available or not include_body_stats:
        return summary

    body_ids = result.body_ids
    if body_ids is None:
        return summary

    body_ids = np.asarray(body_ids)
    vm_by_body = None
    if _has_stacked_sample_axis(history.vm):
        try:
            vm_by_body = history.field_by_body("vm", body_ids)
        except (TypeError, ValueError):
            vm_by_body = None
        else:
            history_summary["vm_max_by_body"] = {
                int(bid): np.asarray(values).max(axis=1).tolist()
                for bid, values in vm_by_body.items()
            }

    if _has_stacked_sample_axis(history.stress):
        try:
            stress_by_body = history.field_by_body("stress", body_ids)
        except (TypeError, ValueError):
            stress_by_body = None
        else:
            final_stress = {}
            for bid, values in stress_by_body.items():
                sigma_final = np.asarray(values)[-1]
                vm_final = None if vm_by_body is None else np.asarray(vm_by_body[bid])[-1]
                final_stress[int(bid)] = {
                    "n_points": int(sigma_final.shape[0]) if sigma_final.ndim >= 1 else 0,
                    "max_abs_sigma": _safe_abs_max(sigma_final),
                    "max_vm": _safe_max(vm_final) if vm_final is not None else float("nan"),
                }
            history_summary["final_stress_by_body"] = final_stress

    return summary


def format_result_summary(
    result,
    *,
    elapsed: Optional[float] = None,
    fields: Iterable[str] = ("u", "stress", "von_mises"),
    include_history: bool = True,
    include_body_stats: bool = True,
) -> str:
    """Render a compact CLI-friendly summary for ``Result`` objects."""
    summary = summarize_result(
        result,
        fields=fields,
        include_history=include_history,
        include_body_stats=include_body_stats,
    )
    lines = []

    if elapsed is not None:
        lines.append(f"solve() took {elapsed:.2f}s")

    for field in summary["fields"]:
        name = field["name"]
        if not field["available"]:
            lines.append(f"  {name:10s}: (unavailable)")
            continue
        line = f"  {name:10s}: {field['shape_text']}"
        if field["note"]:
            line += f"  ({field['note']})"
        lines.append(line)

    history = summary["history"]
    if not include_history or history is None:
        return "\n".join(lines)

    if not history["available"]:
        lines.append("")
        lines.append("  (history not populated — enable save_time_sequence in the config)")
        return "\n".join(lines)

    lines.append("")
    lines.append(
        f"  history: {history['frame_count']} frames "
        f"(u {history['u_shape_text']}, vm {history['vm_shape_text']}, "
        f"stress {history['stress_shape_text']})"
    )
    lines.append(f"  history times: [{_format_values(history['times'], fmt='.3f')}]")

    if history["vm_max_by_body"]:
        lines.append("  per-body max(vm) over time:")
        for bid in sorted(history["vm_max_by_body"]):
            values = history["vm_max_by_body"][bid]
            lines.append(f"    body_id={bid}: [{_format_values(values, fmt='.3e')}]")

    if history["final_stress_by_body"]:
        lines.append("  per-body final-step stress summary:")
        for bid in sorted(history["final_stress_by_body"]):
            stats = history["final_stress_by_body"][bid]
            lines.append(
                f"    body_id={bid}: {stats['n_points']:>6d} sampled points  "
                f"max|σ|={stats['max_abs_sigma']:.3e}  max(vm)={stats['max_vm']:.3e}"
            )

    return "\n".join(lines)
