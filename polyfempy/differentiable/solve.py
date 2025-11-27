"""Differentiable solve function for PolyFEM.

This module provides solve_differentiable(), a differentiable version of solve()
that automatically handles PyTorch integration and gradient computation.
"""

import numpy as np
from typing import Optional, List, Dict, Any, Union

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from ..api.config import SimulationConfig
from ..api import errors as err
from .torch_integration import PolyFEMFunction
from .result import DifferentiableResult


def solve_differentiable(
    V: Union[np.ndarray, "torch.Tensor"],
    C: Union[np.ndarray, "torch.Tensor"],
    cfg: Union[Dict[str, Any], SimulationConfig],
    differentiable_params: Optional[List[str]] = None,
    derivative_type: str = "shape",
    backend: str = "nanobind"
) -> DifferentiableResult:
    """Solve PolyFEM problem with automatic gradient computation support.
    
    This is a differentiable version of solve() that returns PyTorch tensors
    and automatically handles gradient computation through the adjoint method.
    
    Args:
        V: Vertex coordinates, shape (N, dim). Can be numpy array or torch.Tensor.
           If torch.Tensor with requires_grad=True, gradients will be computed.
        C: Cell connectivity, shape (M, k). Can be numpy array or torch.Tensor.
        cfg: Configuration dict or SimulationConfig object.
        differentiable_params: List of parameter names to make differentiable.
            Options: ["geometry", "materials.E", "materials.nu", etc.]
            If None, defaults to ["geometry"].
        derivative_type: Type of derivative to compute. Options:
            - "shape": Shape derivatives (dJ/d(vertices))
            - "material": Material parameter derivatives
            - "initial_velocity": Initial velocity derivatives
            Default: "shape"
        backend: Backend to use. Must be "nanobind" (dummy doesn't support differentiable).
            Default: "nanobind"
    
    Returns:
        DifferentiableResult: Result object with PyTorch tensors that support .backward()
    
    Raises:
        ImportError: If PyTorch is not installed
        ValueError: If backend is not "nanobind" or if inputs are invalid
        RuntimeError: If simulation fails
    
    Example:
        >>> import torch
        >>> from polyfempy.differentiable import solve_differentiable
        >>> 
        >>> V = torch.tensor([[0., 0.], [1., 0.], [1., 1.]], requires_grad=True)
        >>> C = np.array([[0, 1, 2]], dtype=np.int32)
        >>> cfg = {"materials": [{"type": "LinearElasticity", "E": 1e6, "nu": 0.3}]}
        >>> 
        >>> result = solve_differentiable(V, C, cfg)
        >>> loss = torch.norm(result.u)
        >>> loss.backward()
        >>> grad = V.grad  # Automatic gradient computation!
    """
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )
    
    if backend != "nanobind":
        raise ValueError(
            f"Differentiable simulations require 'nanobind' backend, got '{backend}'. "
            "The 'dummy' backend does not support differentiable operations."
        )
    
    # Check if nanobind backend is available
    try:
        import polyfempy as pf
    except ImportError:
        raise ImportError(
            "PolyFEM C++ module is required for differentiable simulations. "
            "Please build the C++ module first."
        )
    
    # Normalize inputs
    if isinstance(V, torch.Tensor):
        V_np = V.detach().cpu().numpy()
        V_requires_grad = V.requires_grad
        V_device = V.device
        V_dtype = V.dtype
    else:
        V_np = np.asarray(V, dtype=np.float64)
        V_requires_grad = False
        V_device = torch.device("cpu")
        V_dtype = torch.float64
    
    if isinstance(C, torch.Tensor):
        C_np = C.detach().cpu().numpy().astype(np.int32)
    else:
        C_np = np.asarray(C, dtype=np.int32)
    
    # Normalize config
    if isinstance(cfg, dict):
        config = SimulationConfig(**cfg)
    elif isinstance(cfg, SimulationConfig):
        config = cfg
    else:
        raise ValueError(f"cfg must be dict or SimulationConfig, got {type(cfg)}")
    
    # Convert config to dict for C++ backend
    settings = config.to_dict()
    
    # Default differentiable params
    if differentiable_params is None:
        differentiable_params = ["geometry"]
    
    # Create Solver object (using old API for direct access to differentiable features)
    # Note: The new solve() API doesn't expose Solver object directly,
    # so we use the old API here for differentiable operations
    try:
        import polyfempy as pf
    except ImportError:
        raise ImportError(
            "PolyFEM C++ module is required for differentiable simulations. "
            "Please build the C++ module first."
        )
    
    solver = pf.Solver()
    
    # Convert settings to JSON string (required by old API)
    import json
    settings_json = json.dumps(settings)
    solver.set_settings(settings_json, strict_validation=False)
    
    # Set mesh from V, C (instead of loading from file)
    # Note: This uses the old API's set_mesh method
    solver.set_mesh(V_np, C_np.astype(np.int32))
    
    # Build basis and assemble (required before solve)
    solver.build_basis()
    solver.assemble()
    
    # For transient problems, initialize time stepping
    if config.time:
        t0 = config.time.get("t0", 0.0)
        dt = config.time.get("dt", 0.01)
        solver.init_timestepping(t0, dt)
    
    # Create PolyFEMFunction and run forward pass
    # Convert V back to torch.Tensor if it was originally
    if V_requires_grad:
        V_torch = torch.tensor(V_np, requires_grad=True, dtype=V_dtype, device=V_device)
    else:
        V_torch = torch.tensor(V_np, requires_grad=False, dtype=V_dtype, device=V_device)
    
    # Run forward pass
    solutions = PolyFEMFunction.apply(solver, V_torch, derivative_type)
    
    # Create result object
    return DifferentiableResult(
        u=solutions,
        solver=solver,
        derivative_type=derivative_type,
        differentiable_params=differentiable_params
    )

