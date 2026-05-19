"""Compatibility shim for ``polyfempy.differentiable.optimization.result``."""

from .optimization.result import *  # noqa: F401,F403
from .optimization.result import _completion_status, _path_or_none, _step_loss_float  # noqa: F401
