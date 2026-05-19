"""Resolve the compiled ``polyfempy.polyfempy`` extension.

``import polyfempy`` can expose the pure-Python :class:`polyfempy.api.config.Solver`
(settings DTO) when the C++ submodule fails to load. Differentiable code must use the
compiled module directly so ``Solver()`` is the FEM backend.
"""

from __future__ import annotations

import importlib
from types import ModuleType

_cached: ModuleType | None = None


def get_cpp_polyfempy() -> ModuleType:
    global _cached
    if _cached is not None:
        return _cached
    try:
        _cached = importlib.import_module("polyfempy.polyfempy")
    except Exception as e:
        raise ImportError(
            "Differentiable PolyFEM requires the compiled extension module "
            "'polyfempy.polyfempy' (real C++ Solver + adjoint). "
            "Build/install the bindings; on Windows, ensure DLL dependencies are on PATH. "
            f"Import error: {e}"
        ) from e
    return _cached


__all__ = ["get_cpp_polyfempy"]
