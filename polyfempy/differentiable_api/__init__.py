"""Clean entry point for the new differentiable PolyFEM API."""

from importlib import import_module

from .initial_condition import InitialConditionOptimization, initial_condition_opt
from .model import DifferentiableModel, model
from .material import MaterialOptimization, material_opt
from .objectives import (
    MaxStress,
    Objective,
    ObjectiveSpec,
    StressNorm,
    objectives,
)
from .parameter import parameter
from .result import DifferentiableResult
from .shape import (
    ShapeOptimization,
    ShapeOptimizationResult,
    shape_opt,
    shape_solve,
)
from .solve import solve
from .state import State, state

__all__ = [
    "DifferentiableModel",
    "DifferentiableResult",
    "InitialConditionOpt",
    "InitialConditionOptimization",
    "MaxStress",
    "MaterialOpt",
    "MaterialOptimization",
    "Objective",
    "ObjectiveSpec",
    "ShapeOpt",
    "ShapeOptimization",
    "ShapeOptimizationResult",
    "State",
    "StressNorm",
    "initial_condition_opt",
    "model",
    "material_opt",
    "objectives",
    "parameter",
    "shape_opt",
    "shape_solve",
    "solve",
    "state",
]

_LAZY_EXPORTS = {
    "InitialConditionOpt": ".torch_ops",
    "MaterialOpt": ".torch_ops",
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
