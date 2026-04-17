"""Differentiable simulation support for PolyFEM.

This module provides tools for gradient-based optimization and PyTorch integration.
Most users don't need this - only use if you need gradients.

Main exports:
    - solve_differentiable: Differentiable version of solve()
    - solve_differentiable_material: Per-element Lamé (λ, μ) as differentiable inputs
    - PolyFEMFunction: PyTorch Function wrapper (use solve_differentiable in most cases)
    - DifferentiableResult: result container with .u, .vertices, .vertices.grad after backward
"""

# pyright: reportMissingImports=false
# torch is an optional dependency, so type checker may not find it

import os
import sys

# Fix OpenMP library conflicts on Windows (set before importing torch)
if sys.platform == 'win32' and 'KMP_DUPLICATE_LIB_OK' not in os.environ:
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

try:
    import torch  # pyright: ignore[reportMissingImports]
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

if _TORCH_AVAILABLE:
    from .solve_diff import solve_differentiable, solve_differentiable_material
    from .torch_integration import PolyFEMFunction, PolyFEMPerElementMaterialFunction
    from .result_diff import DifferentiableResult, DifferentiableMaterialResult
    __all__ = [
        "solve_differentiable",
        "solve_differentiable_material",
        "PolyFEMFunction",
        "PolyFEMPerElementMaterialFunction",
        "DifferentiableResult",
        "DifferentiableMaterialResult",
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

    def solve_differentiable_material(*args, **kwargs):
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )

    def PolyFEMPerElementMaterialFunction(*args, **kwargs):
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )

    def DifferentiableMaterialResult(*args, **kwargs):
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )

