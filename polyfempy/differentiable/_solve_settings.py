"""Config/settings helpers shared by differentiable solve entry points."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

from ..api.config import SimulationConfig
from .cpp_ext import get_cpp_polyfempy


def _console_log_level_from_settings(settings: Dict[str, Any]) -> int:
    """Map ``output.log.level`` string to int 0-6.

    The integer contract matches ``api.solve`` -> ``solver.solve(log_level=...)``.
    """
    log_level = 2
    out = settings.get("output")
    if isinstance(out, dict):
        log = out.get("log")
        if isinstance(log, dict):
            raw = log.get("level", "info")
            log_level_str = raw.strip().lower() if isinstance(raw, str) else str(raw).strip().lower()
            log_level_map = {
                "trace": 0,
                "debug": 1,
                "info": 2,
                "warn": 3,
                "warning": 3,
                "error": 4,
                "critical": 5,
                "off": 6,
            }
            if log_level_str in log_level_map:
                log_level = log_level_map[log_level_str]
    return log_level


def _solver_set_log_level_off(solver: Any) -> None:
    """Best-effort: some PolyFEM Python bindings omit ``set_log_level``."""
    if hasattr(solver, "set_log_level"):
        try:
            solver.set_log_level(6)  # 6=off when supported
        except Exception:
            pass


def _apply_internal_differentiable_runtime_patches(
    *,
    config: Optional[SimulationConfig],
    settings: Dict[str, Any],
) -> None:
    """Apply small runtime fixes needed by differentiable solves.

    This keeps user-facing examples clean: callers should not need to know that
    differentiable contact currently expects a constant barrier stiffness.
    """
    solver_settings = settings.get("solver")
    if not isinstance(solver_settings, dict):
        return

    contact_settings = solver_settings.get("contact")
    if isinstance(contact_settings, dict) and "barrier_stiffness" in contact_settings:
        contact_settings["barrier_stiffness"] = 1e3

    if config is None:
        return

    solver_cfg = getattr(config, "solver", None)
    contact_cfg = getattr(solver_cfg, "contact", None)
    if contact_cfg is not None and hasattr(contact_cfg, "barrier_stiffness"):
        contact_cfg.barrier_stiffness = 1e3


def _geometry_uses_only_absolute_mesh_paths(settings: Dict[str, Any]) -> bool:
    """Return True when every geometry mesh path is already absolute.

    In that case, ``root_path`` is not needed even in config+mesh mode.
    """
    geometry = settings.get("geometry")
    if not isinstance(geometry, list) or not geometry:
        return False

    found_mesh = False
    for item in geometry:
        if not isinstance(item, dict):
            return False
        mesh = item.get("mesh")
        if mesh is None:
            continue
        if not isinstance(mesh, str) or not mesh.strip():
            return False
        found_mesh = True
        if not Path(mesh).is_absolute():
            return False
    return found_mesh


def _cfg_array_mesh_payload(config: SimulationConfig) -> Optional[Dict[str, Any]]:
    extras = getattr(config, "extras", None)
    if isinstance(extras, dict):
        payload = extras.get("_mesh_array_mode")
        if isinstance(payload, dict):
            return payload
    return None


def _differentiable_config_and_settings(
    cfg: Union[str, Dict[str, Any], SimulationConfig],
    *,
    root_path: Optional[str] = None,
    apply_runtime_patches: bool = True,
) -> tuple[SimulationConfig, Optional[SimulationConfig], Dict[str, Any], Optional[str]]:
    """Normalize user configs for differentiable solve entry points."""
    config_root = None
    config_obj: Optional[SimulationConfig] = None
    if isinstance(cfg, str):
        config = SimulationConfig.from_json_file(cfg)
        config_obj = config
        config_root = str(Path(cfg).resolve())
    elif isinstance(cfg, dict):
        config = SimulationConfig.from_json_dict(cfg)
        config_root = cfg.get("root_path") or (config.extras or {}).get("_root_path")
    elif isinstance(cfg, SimulationConfig):
        config = cfg
        config_obj = config
        config_root = (getattr(config, "extras", None) or {}).get("_root_path")
        if not config_root and hasattr(config, "to_dict"):
            config_root = config.to_dict().get("root_path")
    else:
        raise ValueError(f"cfg must be str, dict, or SimulationConfig, got {type(cfg)}")

    settings = config.to_dict()
    if apply_runtime_patches:
        _apply_internal_differentiable_runtime_patches(
            config=config_obj,
            settings=settings,
        )

    root_path_resolved = root_path if root_path is not None else config_root
    if not root_path_resolved:
        root_path_resolved = settings.get("root_path")
    if root_path_resolved:
        settings["root_path"] = root_path_resolved

    return config, config_obj, settings, root_path_resolved


def build_solver_from_settings(
    settings: Dict[str, Any],
    *,
    quiet_polyfem_setup: bool = True,
) -> Any:
    """Build a solver from config-style settings for repeated differentiable solves."""
    pf = get_cpp_polyfempy()
    solver = pf.Solver()
    if quiet_polyfem_setup:
        _solver_set_log_level_off(solver)
    solver.set_settings(json.dumps(settings), strict_validation=False)
    solver.load_mesh_from_settings()
    solver.build_basis()
    if hasattr(solver, "assemble"):
        solver.assemble()
    return solver


__all__ = [
    "build_solver_from_settings",
]
