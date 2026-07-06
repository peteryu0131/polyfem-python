"""Payload helpers for generated ``Root`` objects.

The generated classes own Python-side authoring. This module owns the thin
boundary between ``Root.as_dict()`` and the backend settings dict.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict


def is_generated_config(obj: Any) -> bool:
    """Return True for objects that look like generated config objects."""
    return callable(getattr(obj, "as_dict", None))


def generated_payload_from_config(cfg: Any) -> Dict[str, Any]:
    """Return the plain dict emitted by a generated config object."""
    as_dict = getattr(cfg, "as_dict", None)
    if not callable(as_dict):
        raise TypeError(f"cfg must expose as_dict(), got {type(cfg).__name__}")

    payload = as_dict()
    if not isinstance(payload, dict):
        raise TypeError(
            f"cfg.as_dict() must return dict, got {type(payload).__name__}"
        )
    return payload


def prepare_generated_backend_payload(
    payload: Dict[str, Any],
    *,
    root_path: str | None = None,
) -> Dict[str, Any]:
    """Prepare a generated ``Root.as_dict()`` payload for the backend.

    This is deliberately small: copy the payload, resolve relative geometry
    mesh paths when a root path is available, and drop ``None`` leaves. It does
    not allowlist backend fields.
    """
    backend = copy.deepcopy(payload)
    resolved_root_path = root_path or backend.get("root_path")
    if resolved_root_path:
        _resolve_relative_geometry_mesh_paths(backend, str(resolved_root_path))
    return _drop_none(backend)


def _drop_none(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {
            key: cleaned
            for key, value in obj.items()
            if (cleaned := _drop_none(value)) is not None
        }
    if isinstance(obj, list):
        return [cleaned for value in obj if (cleaned := _drop_none(value)) is not None]
    return obj


def _resolve_relative_geometry_mesh_paths(payload: Dict[str, Any], root_path: str) -> None:
    geometry = payload.get("geometry")
    if not isinstance(geometry, list):
        return

    root = Path(root_path).resolve()
    root_dir = root if root.is_dir() else root.parent
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
