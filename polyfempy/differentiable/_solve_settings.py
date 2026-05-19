"""Config/settings helpers shared by differentiable solve entry points."""

from __future__ import annotations

import json
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

from ..api._solve_contract import (
    cfg_array_mesh_payload,
    MeshSource,
    NoMeshSourceError,
    normalize_config,
    prepare_canonical_solve_input,
)
from ..api.config import SimulationConfig
from .cpp_ext import get_cpp_polyfempy


@dataclass
class DifferentiableSolveContract:
    """Resolved config, settings, and mesh source for differentiable solves."""

    config: SimulationConfig
    settings: Dict[str, Any]
    root_path: Optional[str]
    diagnostics: Dict[str, Any]
    mesh_source: MeshSource


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
) -> list[Dict[str, Any]]:
    """Apply small runtime fixes needed by differentiable solves.

    This keeps user-facing examples clean: callers should not need to know that
    differentiable contact currently expects a constant barrier stiffness.
    """
    solver_settings = settings.get("solver")
    if not isinstance(solver_settings, dict):
        return []

    patches: list[Dict[str, Any]] = []
    contact_settings = solver_settings.get("contact")
    if isinstance(contact_settings, dict) and "barrier_stiffness" in contact_settings:
        old_value = contact_settings["barrier_stiffness"]
        contact_settings["barrier_stiffness"] = 1e3
        patches.append(
            {
                "path": "solver.contact.barrier_stiffness",
                "old": old_value,
                "new": 1e3,
                "reason": "differentiable_contact_requires_constant_barrier_stiffness",
            }
        )

    # Do not mutate ``config``. Runtime patches are backend-setting concerns;
    # callers may reuse the same SimulationConfig after a differentiable solve.
    return patches


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
    return cfg_array_mesh_payload(config)


def _root_path_from_config_and_settings(
    *,
    config: SimulationConfig,
    settings: Dict[str, Any],
) -> Optional[str]:
    extras = getattr(config, "extras", None)
    if isinstance(extras, dict) and extras.get("_root_path"):
        return extras["_root_path"]
    if settings.get("root_path"):
        return settings["root_path"]
    if hasattr(config, "to_dict"):
        try:
            config_dict = config.to_dict()
        except Exception:
            return None
        if isinstance(config_dict, dict):
            return config_dict.get("root_path")
    return None


def _config_obj_for_legacy_return(
    *,
    original_cfg: Union[str, Dict[str, Any], SimulationConfig],
    config: SimulationConfig,
) -> Optional[SimulationConfig]:
    if isinstance(original_cfg, (str, SimulationConfig)):
        return config
    return None


def _is_settings_only_no_mesh_error(exc: ValueError) -> bool:
    return isinstance(exc, NoMeshSourceError)


def prepare_differentiable_solve_contract(
    *,
    V: Any = None,
    C: Any = None,
    cfg: Union[str, Dict[str, Any], SimulationConfig],
    root_path: Optional[str] = None,
    apply_runtime_patches: bool = True,
) -> DifferentiableSolveContract:
    """Prepare the shared solve contract for differentiable entry points."""
    canonical = prepare_canonical_solve_input(
        vertices=V,
        cells=C,
        cfg=cfg,
        dtype=float,
    )
    settings = copy.deepcopy(canonical.backend_settings)
    diagnostics: Dict[str, Any] = dict(canonical.metadata)
    diagnostics.setdefault("runtime_patches", [])

    if apply_runtime_patches:
        diagnostics["runtime_patches"] = _apply_internal_differentiable_runtime_patches(
            config=None,
            settings=settings,
        )

    root_path_resolved = root_path
    if root_path_resolved is None:
        root_path_resolved = _root_path_from_config_and_settings(
            config=canonical.config,
            settings=settings,
        )
    if root_path_resolved:
        settings["root_path"] = root_path_resolved

    return DifferentiableSolveContract(
        config=canonical.config,
        settings=settings,
        root_path=root_path_resolved,
        diagnostics=diagnostics,
        mesh_source=canonical.mesh_source,
    )


def prepare_settings_only_differentiable_contract(
    *,
    cfg: Union[str, Dict[str, Any], SimulationConfig],
    reason: str,
    root_path: Optional[str] = None,
    apply_runtime_patches: bool = True,
) -> DifferentiableSolveContract:
    """Prepare the explicit legacy settings-only differentiable contract."""
    config = normalize_config(cfg)
    settings = config.to_dict()
    diagnostics: Dict[str, Any] = {
        "runtime_patches": [],
        "mesh_source": "settings_only",
        "contract_path": "settings_only_compatibility",
        "fallback_reason": reason,
    }

    if apply_runtime_patches:
        diagnostics["runtime_patches"] = _apply_internal_differentiable_runtime_patches(
            config=None,
            settings=settings,
        )

    root_path_resolved = root_path
    if root_path_resolved is None:
        root_path_resolved = _root_path_from_config_and_settings(
            config=config,
            settings=settings,
        )
    if not root_path_resolved:
        root_path_resolved = settings.get("root_path")
    if root_path_resolved:
        settings["root_path"] = root_path_resolved

    return DifferentiableSolveContract(
        config=config,
        settings=settings,
        root_path=root_path_resolved,
        diagnostics=diagnostics,
        mesh_source=MeshSource(mode="settings_only"),
    )


def _differentiable_config_and_settings(
    cfg: Union[str, Dict[str, Any], SimulationConfig],
    *,
    root_path: Optional[str] = None,
    apply_runtime_patches: bool = True,
    return_diagnostics: bool = False,
):
    """Normalize user configs for differentiable solve entry points."""
    try:
        contract = prepare_differentiable_solve_contract(
            cfg=cfg,
            root_path=root_path,
            apply_runtime_patches=apply_runtime_patches,
        )
    except ValueError as exc:
        if not _is_settings_only_no_mesh_error(exc):
            raise
        settings_only_reason = str(exc)
    else:
        config_obj = _config_obj_for_legacy_return(
            original_cfg=cfg,
            config=contract.config,
        )
        if return_diagnostics:
            return (
                contract.config,
                config_obj,
                contract.settings,
                contract.root_path,
                contract.diagnostics,
            )
        return contract.config, config_obj, contract.settings, contract.root_path

    contract = prepare_settings_only_differentiable_contract(
        cfg=cfg,
        reason=settings_only_reason,
        root_path=root_path,
        apply_runtime_patches=apply_runtime_patches,
    )
    config_obj = _config_obj_for_legacy_return(original_cfg=cfg, config=contract.config)

    if return_diagnostics:
        return (
            contract.config,
            config_obj,
            contract.settings,
            contract.root_path,
            contract.diagnostics,
        )
    return contract.config, config_obj, contract.settings, contract.root_path


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
