"""Clean entry point for the new differentiable PolyFEM API."""

from importlib import import_module

from .model import DifferentiableModel, model
from .objectives import MaxStress, ObjectiveSpec, StressNorm, objectives
from .parameter import parameter
from .result import DifferentiableResult
from .shape import shape_solve
from .solve import solve
from .state import State, state

__all__ = [
    "DifferentiableModel",
    "DifferentiableResult",
    "MaxStress",
    "ObjectiveSpec",
    "ShapeOpt",
    "State",
    "StressNorm",
    "model",
    "objectives",
    "parameter",
    "shape_solve",
    "solve",
    "state",
]

_LAZY_EXPORTS = {
    "ShapeOpt": ".torch_ops",
}


def __getattr__(name: str):
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, package=__package__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *(__all__)])
