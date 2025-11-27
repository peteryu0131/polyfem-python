"""Differentiable simulation support for PolyFEM.

This module provides tools for gradient-based optimization and PyTorch integration.
Most users don't need this - only use if you need gradients.

Main exports:
    - solve_differentiable: Differentiable version of solve()
    - PolyFEMFunction: PyTorch Function wrapper for custom use cases
"""

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

if _TORCH_AVAILABLE:
    from .solve import solve_differentiable
    from .torch_integration import PolyFEMFunction
    from .result import DifferentiableResult
    from .helpers import create_shape_optimizer, gradient_check
    __all__ = [
        "solve_differentiable",
        "PolyFEMFunction",
        "DifferentiableResult",
        "create_shape_optimizer",
        "gradient_check"
    ]
else:
    __all__ = []
    
    def solve_differentiable(*args, **kwargs):
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )
    
    def PolyFEMFunction(*args, **kwargs):
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )

