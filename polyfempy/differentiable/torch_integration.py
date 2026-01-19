"""PyTorch integration for PolyFEM differentiable simulations.

Route A only:
- Always `import polyfempy as pf` (stable C++ extension module name).
- nanobind vs pybind11 is a build-time backend choice and must not change Python imports.

Important:
- The C++ binding's `Solver.solve()` returns `(sol, pressure)`. In forward we therefore
  prefer parsing the return value; only fall back to getters when needed.
"""

import numpy as np
from typing import Optional, List, Dict, Any, Union

try:
    import torch
    from torch.autograd import Function
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    Function = object  # Placeholder for type hints


class PolyFEMFunction(Function):
    """PyTorch Function wrapper for PolyFEM with automatic gradient computation.

    Handles cache management, adjoint solving, and derivative computation.
    Use solve_differentiable() instead of this directly.
    """
    
    @staticmethod
    def forward(ctx, solver, vertices: "torch.Tensor", 
                derivative_type: str = "shape") -> "torch.Tensor":
        """Forward pass: run simulation and cache results."""
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for differentiable simulations")
        
        vertices_np = vertices.detach().cpu().numpy()
        solver.mesh().set_vertices(vertices_np)
        
        # Enable derivative caching (required for adjoint)
        import polyfempy as pf
        solver.set_cache_level(pf.CacheLevel.Derivatives)
        
        ret = solver.solve()

        solutions_np = None
        if isinstance(ret, (tuple, list)) and len(ret) > 0:
            solutions_np = np.asarray(ret[0])
        elif hasattr(solver, "get_solutions"):
            solutions_np = np.asarray(solver.get_solutions())
        else:
            # Last-resort compatibility: try cache object if exposed
            if hasattr(solver, "get_solution_cache"):
                cache = solver.get_solution_cache()
                if hasattr(cache, "solution"):
                    solutions_np = np.asarray(cache.solution(0))
                elif hasattr(cache, "displacement"):
                    solutions_np = np.asarray(cache.displacement(0))

        if solutions_np is None:
            raise RuntimeError(
                "Failed to retrieve solution after solve(): no return tuple and no known getters."
            )
        
        sol_tensor = torch.tensor(solutions_np, dtype=vertices.dtype, device=vertices.device)
        
        ctx.solver = solver
        ctx.derivative_type = derivative_type
        ctx.vertices_shape = vertices.shape
        ctx.vertices_dtype = vertices.dtype
        ctx.vertices_device = vertices.device
        
        return sol_tensor
    
    @staticmethod
    def backward(ctx, grad_output: "torch.Tensor") -> tuple:
        """Backward pass: solve adjoint and compute derivatives."""
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for differentiable simulations")
        
        import polyfempy as pf
        
        grad_output_np = grad_output.detach().cpu().numpy()
        ctx.solver.solve_adjoint(grad_output_np)
        
        if ctx.derivative_type == "shape":
            grad_np = pf.shape_derivative(ctx.solver)
        elif ctx.derivative_type == "material":
            grad_np = pf.elastic_material_derivative(ctx.solver)
        elif ctx.derivative_type == "initial_velocity":
            grad_dict = pf.initial_velocity_derivative(ctx.solver)
            grad_np = np.array(list(grad_dict.values())).flatten()
        else:
            raise ValueError(f"Unknown derivative type: {ctx.derivative_type}")
        
        grad_tensor = torch.tensor(
            grad_np, 
            dtype=ctx.vertices_dtype,
            device=ctx.vertices_device
        )
        
        if grad_tensor.shape != ctx.vertices_shape:
            if grad_tensor.numel() == np.prod(ctx.vertices_shape):
                grad_tensor = grad_tensor.reshape(ctx.vertices_shape)
        
        return None, grad_tensor, None

