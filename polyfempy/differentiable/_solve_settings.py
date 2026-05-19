"""Compatibility shim for ``polyfempy.differentiable.runtime.settings``."""

from .runtime.settings import *  # noqa: F401,F403
from .runtime.settings import (  # noqa: F401
    _apply_internal_differentiable_runtime_patches,
    _console_log_level_from_settings,
    _differentiable_config_and_settings,
    _geometry_uses_only_absolute_mesh_paths,
    _is_settings_only_no_mesh_error,
    _solver_set_log_level_off,
)
