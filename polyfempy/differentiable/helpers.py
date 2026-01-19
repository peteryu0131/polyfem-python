"""Helper functions for differentiable simulations."""

from typing import Optional, List, Dict, Any, Union
import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def create_shape_optimizer(V, C, cfg):
    """Create shape optimization callable.

    Returns function that takes vertices and returns (loss, gradient).
    """
    from .solve import solve_differentiable
    
    def optimize_fn(vertices):
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        result = solve_differentiable(vertices, C, cfg, derivative_type="shape")
        
        if result.u.ndim == 2:
            loss = torch.norm(result.u[:, -1])
        else:
            loss = torch.norm(result.u)
        
        loss.backward()
        grad = vertices.grad
        
        return loss, grad
    
    return optimize_fn


def gradient_check(fn, x: "torch.Tensor", grad: "torch.Tensor", 
                   epsilon: float = 1e-6, rtol: float = 1e-4):
    """Check gradient using finite differences.

    Returns (is_correct, relative_error).
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required")
    
    direction = torch.randn_like(x)
    direction = direction / torch.norm(direction)
    
    x_plus = x + epsilon * direction
    x_minus = x - epsilon * direction
    
    with torch.no_grad():
        f_plus = fn(x_plus)
        f_minus = fn(x_minus)
        fd_grad = (f_plus - f_minus) / (2 * epsilon)
    
    analytical = torch.dot(grad.flatten(), direction.flatten())
    relative_error = abs(analytical - fd_grad) / (abs(analytical) + 1e-10)
    is_correct = relative_error < rtol
    
    return is_correct, relative_error.item()

