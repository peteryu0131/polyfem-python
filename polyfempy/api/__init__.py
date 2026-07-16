"""Public Python API for PolyFEM.

The recommended user-facing surface is intentionally small:

- ``solve`` for running simulations
- ``Result`` for structured solver output

Generated configuration helpers are the preferred direction for new authoring
APIs. Legacy config, output, report, and runtime helpers remain importable from
their owning modules, but they are not re-exported here.
"""

from ._runtime import (
    configure_windows_runtime as _configure_windows_runtime,
    should_auto_configure_windows as _should_auto_configure_windows,
)

if _should_auto_configure_windows():
    _configure_windows_runtime()

from .result import Result
from .solve import solve

CORE_API = [
    "solve",
    "Result",
]

__all__ = list(CORE_API)
