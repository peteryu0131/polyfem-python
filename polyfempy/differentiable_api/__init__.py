"""Clean entry point for the new differentiable PolyFEM API."""

from .parameter import parameter
from .result import DifferentiableResult
from .solve import solve
from .state import State, state

__all__ = [
    "DifferentiableResult",
    "State",
    "parameter",
    "solve",
    "state",
]

