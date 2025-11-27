"""Helper functions and utilities for differentiable simulations.

This module provides convenience functions to reduce boilerplate code
for common differentiable simulation scenarios.
"""

from typing import Optional, List, Dict, Any, Union
import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def create_shape_optimizer(V, C, cfg):
    """Create a shape optimization helper.
    
    This function creates a callable that can be used for shape optimization
    with automatic gradient computation.
    
    Args:
        V: Initial vertex coordinates
        C: Cell connectivity
        cfg: Configuration
    
    Returns:
        Callable function that takes vertices and returns (loss, gradient)
    
    Example:
        >>> optimizer = create_shape_optimizer(V, C, cfg)
        >>> vertices = torch.tensor(V, requires_grad=True)
        >>> loss, grad = optimizer(vertices)
    """
    from .solve import solve_differentiable
    
    def optimize_fn(vertices):
        """Optimization function."""
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        result = solve_differentiable(vertices, C, cfg, derivative_type="shape")
        
        # Compute loss (example: norm of final displacement)
        if result.u.ndim == 2:
            # Transient: use last time step
            loss = torch.norm(result.u[:, -1])
        else:
            # Static: use all
            loss = torch.norm(result.u)
        
        # Compute gradient
        loss.backward()
        grad = vertices.grad
        
        return loss, grad
    
    return optimize_fn


def gradient_check(
    fn,
    x: "torch.Tensor",
    grad: "torch.Tensor",
    epsilon: float = 1e-6,
    rtol: float = 1e-4
):
    """Check gradient using finite differences.
    
    Args:
        fn: Function that takes x and returns a scalar loss
        x: Input tensor
        grad: Computed gradient
        epsilon: Perturbation size for finite differences
        rtol: Relative tolerance for comparison
    
    Returns:
        (is_correct, relative_error): Whether gradient is correct and error
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required")
    
    # Random direction
    direction = torch.randn_like(x)
    direction = direction / torch.norm(direction)
    
    # Finite difference
    x_plus = x + epsilon * direction
    x_minus = x - epsilon * direction
    
    with torch.no_grad():
        f_plus = fn(x_plus)
        f_minus = fn(x_minus)
        fd_grad = (f_plus - f_minus) / (2 * epsilon)
    
    # Analytical gradient
    analytical = torch.dot(grad.flatten(), direction.flatten())
    
    # Compare
    relative_error = abs(analytical - fd_grad) / (abs(analytical) + 1e-10)
    is_correct = relative_error < rtol
    
    return is_correct, relative_error.item()

