"""Shared solve contract helpers.

This internal module owns the parts of ``solve`` that define user-facing
semantics before the C++ backend is touched: config normalization, mesh-source
selection, and backend settings construction.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from . import tensor as T


@dataclass
class MeshSource:
    """Resolved mesh source for a solve call."""

    mode: str
    vertices: Optional[np.ndarray] = None
    cells: Optional[np.ndarray] = None
    body_ids: Optional[np.ndarray] = None
    boundary_ids: Optional[np.ndarray] = None
    v_backend: str = "numpy"


@dataclass
class CanonicalSolveInput:
    """Normalized config, mesh, and backend settings for a solve pipeline."""

    config: Any
    full_json: Optional[Dict[str, Any]]
    mesh_source: MeshSource
    backend_settings: Dict[str, Any]
    metadata: Dict[str, Any]


def cfg_array_mesh_payload(cfg: Any) -> Optional[Dict[str, Any]]:
    extras = getattr(cfg, "extras", None)
    if isinstance(extras, dict):
        payload = extras.get("_mesh_array_mode")
        if isinstance(payload, dict):
            return payload
    return None


def normalize_config(cfg: Any):
    """Accept dict / JSON path / SimulationConfig and return SimulationConfig."""
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


def validate_config(cfg: Any) -> None:
    if hasattr(cfg, "validate"):
        cfg.validate()


def _ensure_i32(cells: np.ndarray) -> np.ndarray:
    return cells.astype(np.int32, copy=False) if cells.dtype != np.int32 else cells


def _promote_materials_to_list(payload: Dict[str, Any], *, infer_type_from_pde: bool = False) -> None:
    materials = payload.get("materials")
    if not isinstance(materials, dict):
        return
    if infer_type_from_pde and "type" not in materials:
        pde = payload.get("pde", "LinearElasticity")
        materials["type"] = "Laplacian" if pde == "Poisson" else "LinearElasticity"
    payload["materials"] = [materials]


def _cfg_to_dict_or_raise(cfg: Any, *, context: str) -> Any:
    try:
        return cfg.to_dict()
    except Exception as exc:
        raise RuntimeError(f"cfg.to_dict() failed while {context}") from exc


def merge_user_cfg_over_full_json(cfg: Any, full_json: Dict[str, Any]) -> Dict[str, Any]:
    cfg_dict = _cfg_to_dict_or_raise(
        cfg,
        context="merging Python-side config over full JSON",
    )

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


def build_full_json(cfg: Any) -> Optional[Dict[str, Any]]:
    validate_config(cfg)

    if cfg_array_mesh_payload(cfg) is not None:
        return None

    if hasattr(cfg, "extras") and cfg.extras and "_full_json_config" in cfg.extras:
        return merge_user_cfg_over_full_json(cfg, cfg.extras["_full_json_config"])

    cfg_dict = _cfg_to_dict_or_raise(
        cfg,
        context="building full JSON settings",
    )

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


def choose_mesh_source(
    vertices: Any,
    cells: Any,
    full_json: Optional[Dict[str, Any]],
    *,
    dtype: Any = None,
    cfg: Any = None,
) -> MeshSource:
    """Resolve the real mesh source for this solve.

    Explicit partial array input is an error even when JSON geometry exists,
    because silently falling back would change the user's requested data binding.
    """
    if (vertices is None) ^ (cells is None):
        raise ValueError("array mode requires both vertices and cells")

    array_payload = cfg_array_mesh_payload(cfg)
    used_cfg_array_payload = False
    if vertices is None and cells is None and array_payload is not None:
        vertices = array_payload.get("vertices")
        cells = array_payload.get("cells")
        used_cfg_array_payload = True
        if vertices is None or cells is None:
            raise ValueError("guided array mesh payload requires both vertices and cells")

    if vertices is not None and cells is not None:
        V_np, v_backend = T.as_numpy(vertices, dtype=dtype)
        C_np, _ = T.as_numpy(cells, dtype=np.int32)
        body_ids_np = None
        boundary_ids_np = None
        if used_cfg_array_payload and array_payload is not None:
            if array_payload.get("body_ids") is not None:
                body_ids_np = np.asarray(array_payload["body_ids"], dtype=np.int32).reshape(-1)
            if array_payload.get("boundary_ids") is not None:
                boundary_ids_np = np.asarray(array_payload["boundary_ids"], dtype=np.int32).reshape(-1)
        return MeshSource(
            mode="guided_array" if used_cfg_array_payload else "array",
            vertices=V_np,
            cells=_ensure_i32(C_np),
            body_ids=body_ids_np,
            boundary_ids=boundary_ids_np,
            v_backend=v_backend,
        )

    if full_json is not None and "geometry" in full_json:
        return MeshSource(mode="json", v_backend="numpy")

    raise ValueError(
        "Either provide vertices/cells arrays, or use JSON config with geometry (mesh files)"
    )


def clean_json_for_cpp(obj: Any):
    if isinstance(obj, dict):
        cleaned = {}
        for key, value in obj.items():
            cleaned_value = clean_json_for_cpp(value)
            if cleaned_value is not None:
                cleaned[key] = cleaned_value
        return cleaned
    if isinstance(obj, list):
        return [clean_json_for_cpp(item) for item in obj if clean_json_for_cpp(item) is not None]
    return obj


def process_json_config(full_json: Dict[str, Any], cfg: Any) -> Dict[str, Any]:
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


def _strip_guided_array_placeholders(settings: Dict[str, Any]) -> None:
    geometry = settings.get("geometry")
    if not isinstance(geometry, list):
        return
    kept = []
    for entry in geometry:
        if isinstance(entry, dict) and str(entry.get("mesh", "")).startswith("__array_body__:"):
            continue
        kept.append(entry)
    if kept:
        settings["geometry"] = kept
    else:
        settings.pop("geometry", None)


def build_canonical_solver_settings(
    cfg: Any,
    *,
    full_json: Optional[Dict[str, Any]],
    mesh_source: MeshSource,
) -> Dict[str, Any]:
    """Build backend JSON/settings after config and mesh source are known."""
    if mesh_source.mode == "json":
        if full_json is None:
            raise ValueError("JSON mesh mode requires full_json settings")
        return clean_json_for_cpp(process_json_config(full_json, cfg))

    if hasattr(cfg, "to_dict"):
        settings = cfg.to_dict()
    elif hasattr(cfg, "to_json_dict"):
        settings = cfg.to_json_dict()
    else:
        raise TypeError("cfg must provide to_dict() or to_json_dict() for array mode.")

    if mesh_source.mode == "guided_array":
        _strip_guided_array_placeholders(settings)

    if "geometry" not in settings:
        settings["geometry"] = [
            {"type": "ground", "height": 0.0, "enabled": True, "is_obstacle": False}
        ]

    _promote_materials_to_list(settings, infer_type_from_pde=True)
    return clean_json_for_cpp(settings)


def prepare_canonical_solve_input(
    *,
    vertices: Any,
    cells: Any,
    cfg: Any,
    dtype: Any = None,
) -> CanonicalSolveInput:
    """Normalize config, resolve mesh source, and build backend settings once."""
    config = normalize_config(cfg)
    full_json = build_full_json(config)
    mesh_source = choose_mesh_source(
        vertices,
        cells,
        full_json,
        dtype=dtype,
        cfg=config,
    )
    backend_settings = build_canonical_solver_settings(
        config,
        full_json=full_json,
        mesh_source=mesh_source,
    )
    return CanonicalSolveInput(
        config=config,
        full_json=full_json,
        mesh_source=mesh_source,
        backend_settings=backend_settings,
        metadata={"mesh_source": mesh_source.mode},
    )
