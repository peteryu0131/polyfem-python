"""Differentiable solve function for PolyFEM."""

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
    """Solve with automatic gradient computation.

    Returns PyTorch tensors supporting .backward() via adjoint method.

    Args:
        V: (N, dim) vertices. If torch.Tensor with requires_grad=True, computes gradients.
        C: (M, k) cells.
        cfg: Configuration dict or SimulationConfig.
        differentiable_params: Parameter names to make differentiable. Default: ["geometry"].
        derivative_type: "shape", "material", "initial_velocity", etc. Default: "shape".
        backend: Must be "nanobind" (dummy doesn't support differentiable).

    Returns:
        DifferentiableResult with PyTorch tensors.
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
    
    # Check if nanobind backend is available and import
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
    
    if isinstance(cfg, dict):
        config = SimulationConfig(**cfg)
    elif isinstance(cfg, SimulationConfig):
        config = cfg
    else:
        raise ValueError(f"cfg must be dict or SimulationConfig, got {type(cfg)}")
    
    settings = config.to_dict()
    
    if differentiable_params is None:
        differentiable_params = ["geometry"]
    
    # Use old API for direct Solver access (required for differentiable operations)
    try:
        import polyfempy as pf
    except ImportError:
        raise ImportError(
            "PolyFEM C++ module is required for differentiable simulations. "
            "Please build the C++ module first."
        )
    
    solver = pf.Solver()
    
    import json
    settings_json = json.dumps(settings)
    solver.set_settings(settings_json, strict_validation=False)
    solver.set_mesh(V_np, C_np.astype(np.int32))
    solver.build_basis()
    solver.assemble()
    
    # Initialize time stepping for transient problems
    if config.time:
        t0 = config.time.get("t0", 0.0)
        dt = config.time.get("dt", 0.01)
        solver.init_timestepping(t0, dt)
    
    if V_requires_grad:
        V_torch = torch.tensor(V_np, requires_grad=True, dtype=V_dtype, device=V_device)
    else:
        V_torch = torch.tensor(V_np, requires_grad=False, dtype=V_dtype, device=V_device)
    
    solutions = PolyFEMFunction.apply(solver, V_torch, derivative_type)
    
    return DifferentiableResult(
        u=solutions,
        solver=solver,
        derivative_type=derivative_type,
        differentiable_params=differentiable_params
    )

