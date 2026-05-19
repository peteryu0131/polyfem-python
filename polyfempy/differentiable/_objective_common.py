"""Compatibility shim for ``polyfempy.differentiable.objectives.common``."""

from .objectives.common import *  # noqa: F401,F403
from .objectives.common import (  # noqa: F401
    _auto_smooth_max_sharpness,
    _resolve_smooth_max_sharpness,
    _resolve_time_aggregation,
    _resolve_volume_selection,
)
