"""PyTorch integration for PolyFEM differentiable simulations.

This module provides PyTorch Function wrappers that automatically handle
cache management, adjoint solving, and gradient computation.
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
    """PyTorch Function wrapper for PolyFEM simulations with automatic gradient computation.
    
    This class encapsulates the forward and backward passes for differentiable simulations,
    automatically handling cache management, adjoint solving, and derivative computation.
    
    Users typically don't need to use this directly - use solve_differentiable() instead.
    This class is provided for advanced users who need custom control.
    
    Example:
        >>> class MySimulate(PolyFEMFunction):
        ...     @staticmethod
        ...     def forward(ctx, solver, vertices):
        ...         # Custom forward logic
        ...         return PolyFEMFunction.forward(ctx, solver, vertices)
    """
    
    @staticmethod
    def forward(ctx, solver, vertices: "torch.Tensor", 
                derivative_type: str = "shape") -> "torch.Tensor":
        """Forward pass: run simulation and cache results.
        
        Args:
            ctx: PyTorch context for storing intermediate values
            solver: PolyFEM Solver object (from pf.Solver())
            vertices: Vertex coordinates as torch.Tensor, shape (N, dim)
            derivative_type: Type of derivative to compute, one of:
                - "shape": Shape derivatives (default)
                - "material": Material parameter derivatives
                - "initial_velocity": Initial velocity derivatives
                - etc.
        
        Returns:
            torch.Tensor: Solution array, shape (n_dof, n_time_steps)
        """
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for differentiable simulations")
        
        # Convert torch.Tensor to numpy
        vertices_np = vertices.detach().cpu().numpy()
        
        # Update mesh vertices
        solver.mesh().set_vertices(vertices_np)
        
        # Enable derivative caching (critical!)
        import polyfempy as pf
        solver.set_cache_level(pf.CacheLevel.Derivatives)
        
        # Run simulation
        solver.solve()
        
        # Get solutions (all time steps)
        solutions = solver.get_solutions()  # numpy array, shape (n_dof, n_time_steps)
        
        # Convert to torch.Tensor
        sol_tensor = torch.tensor(solutions, dtype=vertices.dtype, device=vertices.device)
        
        # Store in context for backward pass
        ctx.solver = solver
        ctx.derivative_type = derivative_type
        ctx.vertices_shape = vertices.shape
        ctx.vertices_dtype = vertices.dtype
        ctx.vertices_device = vertices.device
        
        return sol_tensor
    
    @staticmethod
    def backward(ctx, grad_output: "torch.Tensor") -> tuple:
        """Backward pass: solve adjoint and compute derivatives.
        
        Args:
            ctx: PyTorch context with stored values from forward pass
            grad_output: Gradient of loss w.r.t. solution, shape (n_dof, n_time_steps)
        
        Returns:
            tuple: Gradients w.r.t. inputs (None for solver, gradient for vertices)
        """
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for differentiable simulations")
        
        import polyfempy as pf
        
        # Convert grad_output to numpy
        grad_output_np = grad_output.detach().cpu().numpy()
        
        # Solve adjoint equation
        ctx.solver.solve_adjoint(grad_output_np)
        
        # Compute derivative based on type
        if ctx.derivative_type == "shape":
            grad_np = pf.shape_derivative(ctx.solver)
        elif ctx.derivative_type == "material":
            grad_np = pf.elastic_material_derivative(ctx.solver)
        elif ctx.derivative_type == "initial_velocity":
            grad_dict = pf.initial_velocity_derivative(ctx.solver)
            # Convert dict to array (simplified - may need more handling)
            grad_np = np.array(list(grad_dict.values())).flatten()
        else:
            raise ValueError(f"Unknown derivative type: {ctx.derivative_type}")
        
        # Convert to torch.Tensor with correct shape and device
        grad_tensor = torch.tensor(
            grad_np, 
            dtype=ctx.vertices_dtype,
            device=ctx.vertices_device
        )
        
        # Reshape to match input vertices shape
        if grad_tensor.shape != ctx.vertices_shape:
            # Try to reshape (may need more sophisticated handling)
            if grad_tensor.numel() == np.prod(ctx.vertices_shape):
                grad_tensor = grad_tensor.reshape(ctx.vertices_shape)
            else:
                # If shapes don't match, return flattened version
                # User should handle reshaping based on derivative type
                pass
        
        # Return gradients: 
        # - None for solver (no gradient needed)
        # - grad_tensor for vertices (the actual gradient)
        # - None for derivative_type (string parameter, no gradient)
        return None, grad_tensor, None

