"""Training-sample export helpers for differentiable experiments."""

# pyright: reportMissingImports=false

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch

from ...api.report import summarize_history_bundle
from ..optimization.summary import gradient_norm


def _tensor_to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def _finite_float(value: Any, default: float = float("-inf")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _json_float_or_none(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _array_record(path: Path, arr: np.ndarray) -> dict[str, Any]:
    return {
        "path": path.name,
        "shape": [int(x) for x in arr.shape],
        "dtype": str(arr.dtype),
    }


def _select_body_history_row(
    bundle: dict[str, Any],
    *,
    body_id: int,
    select_by: str = "vm_max",
) -> Optional[dict[str, Any]]:
    rows = [
        row
        for row in bundle.get("steps_by_body", [])
        if int(row.get("body_id", -1)) == int(body_id)
    ]
    if not rows:
        return None
    return max(rows, key=lambda row: _finite_float(row.get(select_by)))


def save_training_sample(
    *,
    result: Any,
    loss: torch.Tensor,
    workspace: Union[str, Path],
    cfg: Any = None,
    gradient: Optional[torch.Tensor] = None,
    gradient_body_id: Optional[int] = None,
    gradient_body_name: Optional[str] = None,
    body_id: int = 1,
    body_name: str = "lattice",
    sample_dir_name: str = "training_sample",
) -> dict[str, Path]:
    """Save one differentiable run as a small ML training sample.

    The JSON file stores scalar labels and points to ``.npy`` arrays. The
    default body id ``1`` matches the lattice in experiment 02. ``body_id``
    selects the body used for reported von Mises labels. Pass ``gradient`` when
    you want to save a masked gradient, such as a block loss gradient restricted
    to lattice vertices.
    """
    out_dir = Path(workspace) / sample_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    gradient_tensor = result.shape_gradient if gradient is None else gradient
    if gradient_tensor is None:
        raise ValueError("shape gradient is missing; call loss.backward() before save_training_sample()")

    gradient_np = _tensor_to_numpy(gradient_tensor)
    gradient_path = out_dir / "shape_gradient.npy"
    np.save(gradient_path, gradient_np)

    vertices_path = None
    vertices_np = None
    if getattr(result, "vertices", None) is not None:
        vertices_np = _tensor_to_numpy(result.vertices)
        vertices_path = out_dir / "vertices.npy"
        np.save(vertices_path, vertices_np)

    loss_value = float(loss.detach().cpu().item())
    grad_norm_value = gradient_norm(gradient_tensor)

    bundle = summarize_history_bundle(result, cfg=cfg)
    body_row = _select_body_history_row(bundle, body_id=int(body_id), select_by="vm_max")
    if body_row is None:
        body_vm_mean = np.nan
        body_vm_max = np.nan
        body_vm_p95 = np.nan
        selected_step = None
        selected_time = None
    else:
        body_vm_mean = _finite_float(body_row.get("vm_mean"), default=np.nan)
        body_vm_max = _finite_float(body_row.get("vm_max"), default=np.nan)
        body_vm_p95 = _finite_float(body_row.get("vm_p95"), default=np.nan)
        selected_step = int(body_row["step"]) if body_row.get("step") is not None else None
        selected_time = _json_float_or_none(body_row.get("time"))

    scalar_labels = np.asarray(
        [
            loss_value,
            grad_norm_value,
            body_vm_max,
            body_vm_p95,
            body_vm_mean,
        ],
        dtype=np.float64,
    )
    scalar_path = out_dir / "scalars.npy"
    np.save(scalar_path, scalar_labels)

    metadata: dict[str, Any] = {
        "loss": _json_float_or_none(loss_value),
        "grad_norm": _json_float_or_none(grad_norm_value),
        "shape_gradient": _array_record(gradient_path, gradient_np),
        "shape_gradient_source": "result.shape_gradient" if gradient is None else "provided_gradient",
        "scalars": {
            **_array_record(scalar_path, scalar_labels),
            "columns": [
                "loss",
                "grad_norm",
                f"{body_name}_vm_max",
                f"{body_name}_vm_p95",
                f"{body_name}_vm_mean",
            ],
        },
        f"{body_name}_body_id": int(body_id),
        f"{body_name}_history_selection": "row with largest vm_max over time",
        f"{body_name}_selected_step": selected_step,
        f"{body_name}_selected_time": selected_time,
        f"{body_name}_vm_max": _json_float_or_none(body_vm_max),
        f"{body_name}_vm_p95": _json_float_or_none(body_vm_p95),
        f"{body_name}_vm_mean": _json_float_or_none(body_vm_mean),
        "objective_body_id": int(body_id),
        "objective_body_name": body_name,
        "gradient_body_id": None if gradient_body_id is None else int(gradient_body_id),
        "gradient_body_name": gradient_body_name,
        "history": {
            "available": bool(bundle.get("available", False)),
            "source": bundle.get("history_source", "unknown"),
            "steps_by_body_rows": len(list(bundle.get("steps_by_body", []))),
        },
    }
    if vertices_path is not None and vertices_np is not None:
        metadata["vertices"] = _array_record(vertices_path, vertices_np)

    metadata_path = out_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )

    paths = {
        "metadata_json": metadata_path.resolve(),
        "shape_gradient_npy": gradient_path.resolve(),
        "scalars_npy": scalar_path.resolve(),
    }
    if vertices_path is not None:
        paths["vertices_npy"] = vertices_path.resolve()
    return paths


__all__ = ["save_training_sample"]
