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
    mesh paths when a root path is available, restore generated-class defaults
    that would otherwise change backend semantics, and drop ``None`` leaves.
    """
    backend = copy.deepcopy(payload)
    resolved_root_path = root_path or backend.get("root_path")
    if resolved_root_path:
        _resolve_relative_geometry_mesh_paths(backend, str(resolved_root_path))
    _restore_backend_payload_semantics(backend)
    return _drop_none(backend)


_SOLVER_ADVANCED_GENERATED_DEFAULTS = {
    "cache_size": 900000,
    "lump_mass_matrix": False,
    "lagged_regularization_weight": 0.0,
    "lagged_regularization_iterations": 1,
    "check_inversion": "Discrete",
    "jacobian_threshold": 0.0,
    "characteristic_length": -1.0,
    "characteristic_force_density": 10000.0,
}

_NONLINEAR_GENERATED_DEFAULTS = {
    "rel_grad_norm_tol": 1e-10,
    "newton_decrement_tol": 0.0,
    "rel_x_delta_tol": 0.0,
    "first_grad_norm_tol": 1e-12,
    "norm_type": "L2",
    "max_iterations": 500,
    "allow_out_of_iterations": False,
}

_LINE_SEARCH_GENERATED_DEFAULTS = {
    "use_grad_norm_tol": 1e-06,
    "min_step_size": 1e-10,
    "max_step_size_iter": 30,
    "min_step_size_final": 1e-20,
    "max_step_size_iter_final": 100,
    "default_init_step_size": 1.0,
    "step_ratio": 0.5,
}

_NONLINEAR_ADVANCED_GENERATED_DEFAULTS = {
    "f_delta": 0.0,
    "f_delta_step_tol": 100,
    "derivative_along_delta_x_tol": 0,
    "apply_gradient_fd": "None",
    "gradient_fd_eps": 1e-07,
}


def _restore_backend_payload_semantics(payload: Dict[str, Any]) -> None:
    _restore_transformation_semantics(payload)
    _restore_nonlinear_solver_semantics(payload)
    _drop_solver_advanced_generated_defaults(payload)


def _restore_transformation_semantics(payload: Dict[str, Any]) -> None:
    geometry = payload.get("geometry")
    if not isinstance(geometry, list):
        return

    for entry in geometry:
        if not isinstance(entry, dict):
            continue

        transformation = entry.get("transformation")
        if not isinstance(transformation, dict):
            continue

        for scalar_or_vector_key in ("scale", "rotation"):
            value = transformation.get(scalar_or_vector_key)
            if isinstance(value, list) and len(value) == 1:
                transformation[scalar_or_vector_key] = value[0]

        for empty_list_key in ("translation", "rotation", "scale"):
            if transformation.get(empty_list_key) == []:
                transformation.pop(empty_list_key)

        if transformation.get("rotation_mode") == "xyz":
            transformation.pop("rotation_mode")

        if not transformation:
            entry.pop("transformation")


def _drop_solver_advanced_generated_defaults(payload: Dict[str, Any]) -> None:
    solver = payload.get("solver")
    if not isinstance(solver, dict):
        return

    advanced = solver.get("advanced")
    if not isinstance(advanced, dict):
        return

    for key, default_value in _SOLVER_ADVANCED_GENERATED_DEFAULTS.items():
        if advanced.get(key) == default_value:
            advanced.pop(key)

    if not advanced:
        solver.pop("advanced")


def _restore_nonlinear_solver_semantics(payload: Dict[str, Any]) -> None:
    solver = payload.get("solver")
    if not isinstance(solver, dict):
        return

    nonlinear = solver.get("nonlinear")
    if not isinstance(nonlinear, dict):
        return

    _rename_key(nonlinear, "x_delta_tol", "x_delta")
    _rename_key(nonlinear, "grad_norm_tol", "grad_norm")

    for key, default_value in _NONLINEAR_GENERATED_DEFAULTS.items():
        if nonlinear.get(key) == default_value:
            nonlinear.pop(key)

    line_search = nonlinear.get("line_search")
    if isinstance(line_search, dict):
        for key, default_value in _LINE_SEARCH_GENERATED_DEFAULTS.items():
            if line_search.get(key) == default_value:
                line_search.pop(key)
        if not line_search:
            nonlinear.pop("line_search")

    advanced = nonlinear.get("advanced")
    if isinstance(advanced, dict):
        _rename_key(advanced, "f_delta_tol", "f_delta")
        for key, default_value in _NONLINEAR_ADVANCED_GENERATED_DEFAULTS.items():
            if advanced.get(key) == default_value:
                advanced.pop(key)
        if not advanced:
            nonlinear.pop("advanced")


def _rename_key(payload: Dict[str, Any], old: str, new: str) -> None:
    if old in payload and new not in payload:
        payload[new] = payload.pop(old)
    else:
        payload.pop(old, None)


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
