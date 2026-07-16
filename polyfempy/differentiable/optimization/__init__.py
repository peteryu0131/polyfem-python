"""Generic optimization entry points and reporting helpers."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORT_MODULES = {
    "OptimizationKind": ".problem",
    "OptimizationProblem": ".problem",
    "OptimizationRunResult": ".problem",
    "make_optimizer": ".problem",
    "make_torch_optimizer": ".optimizers",
    "prepare_optimization_baseline_simulation": ".problem",
    "prepare_optimization_problem": ".problem",
    "report_optimization_baseline": ".problem",
    "run_optimization": ".problem",
}

__all__ = sorted(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, package=__name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *__all__])
