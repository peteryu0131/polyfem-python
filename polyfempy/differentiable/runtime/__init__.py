"""Runtime pieces for differentiable PolyFEM solves.

This package is kept lazy so importing unrelated differentiable reference
modules does not load the legacy solve/settings path.
"""

from __future__ import annotations

from importlib import import_module


_EXPORT_MODULES = {
    "DifferentiableMaterialResult": ".result",
    "DifferentiableResult": ".result",
    "DifferentiableSolveContract": ".settings",
    "build_solver_from_settings": ".settings",
    "prepare_differentiable_solve_contract": ".settings",
    "prepare_settings_only_differentiable_contract": ".settings",
    "prepare_differentiable_simulation": ".solve",
    "solve_differentiable": ".solve",
    "solve_differentiable_material": ".solve",
    "solve_differentiable_material_from_youngs": ".solve",
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, package=__package__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *_EXPORT_MODULES])
