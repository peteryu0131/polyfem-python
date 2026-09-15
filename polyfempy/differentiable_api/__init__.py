"""Clean entry point for the new differentiable PolyFEM API."""

from .model import DifferentiableModel, model
from .parameter import parameter
from .result import DifferentiableResult
from .solve import solve
from .state import State, state

__all__ = [
    "DifferentiableModel",
    "DifferentiableResult",
    "State",
    "model",
    "parameter",
    "solve",
    "state",
]
