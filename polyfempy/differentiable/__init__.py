"""Differentiable simulation support for PolyFEM.

Recommended user scripts should stay on the small public surface below.
Lower-level diagnostics and finite-difference checks live under
``polyfempy.differentiable.advanced``. Compatibility names are still available
by explicit import when PyTorch is installed, but they are excluded from
``__all__`` so the recommended API is easy to inspect.
"""

# pyright: reportMissingImports=false
# torch is an optional dependency, so type checkers may not find it.

from __future__ import annotations

from importlib import import_module
import os
import sys
from typing import Callable

from ._exports import (
    ADVANCED_API,
    ADVANCED_COMPAT_API,
    COMPATIBILITY_API,
    EXPORT_MODULES,
    PUBLIC_API,
)


# Fix OpenMP library conflicts on Windows before importing torch.
if sys.platform == "win32" and "KMP_DUPLICATE_LIB_OK" not in os.environ:
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    import torch  # pyright: ignore[reportMissingImports]  # noqa: F401

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

__all__ = list(PUBLIC_API)


_EXPORT_MODULES = EXPORT_MODULES


def _missing_torch_stub(name: str) -> Callable[..., None]:
    def _stub(*args, **kwargs):
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )

    _stub.__name__ = name
    _stub.__qualname__ = name
    _stub.__doc__ = "Unavailable placeholder used when PyTorch is not installed."
    return _stub


def _load_export(name: str):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, package=__package__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __getattr__(name: str):
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if not _TORCH_AVAILABLE:
        value = _missing_torch_stub(name)
        globals()[name] = value
        return value
    return _load_export(name)


def __dir__() -> list[str]:
    return sorted([*globals(), *PUBLIC_API])


for _name in PUBLIC_API:
    globals()[_name] = _load_export(_name) if _TORCH_AVAILABLE else _missing_torch_stub(_name)
del _name
