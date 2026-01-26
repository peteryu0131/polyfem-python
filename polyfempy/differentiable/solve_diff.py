"""Differentiable solve function for PolyFEM."""

# pyright: reportMissingImports=false
# torch is an optional dependency, so type checker may not find it

import numpy as np
from typing import Optional, List, Dict, Any, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

try:
    import torch  # pyright: ignore[reportMissingImports]
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from ..api.config import SimulationConfig
from .torch_integration import PolyFEMFunction
from .result_diff import DifferentiableResult


def solve_differentiable(
    V: Union[np.ndarray, "torch.Tensor"],
    C: Union[np.ndarray, "torch.Tensor"],
    cfg: Union[Dict[str, Any], SimulationConfig],
    differentiable_params: Optional[List[str]] = None,
    derivative_type: str = "shape",
    backend: str = "nanobind",
    sidesets_func: Optional[callable] = None
) -> DifferentiableResult:
    """Solve with automatic gradient computation.

    Returns PyTorch tensors supporting .backward() via adjoint method.

    Args:
        V: (N, dim) vertices. If torch.Tensor with requires_grad=True, computes gradients.
        C: (M, k) cells.
        cfg: Configuration dict or SimulationConfig.
        differentiable_params: Parameter names to make differentiable. Default: ["geometry"].
        derivative_type: "shape", "periodic_shape", "material", "initial_velocity", etc. 
            Default: "shape". Use "periodic_shape" for periodic boundary conditions or periodic contact.
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
    # Note: polyfempy is imported here and reused later
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
        config = SimulationConfig.from_json_dict(cfg)
    elif isinstance(cfg, SimulationConfig):
        config = cfg
    else:
        raise ValueError(f"cfg must be dict or SimulationConfig, got {type(cfg)}")
    
    settings = config.to_dict()
    
    # Add placeholder geometry field if not present (required by JSON validator)
    # Since we're setting mesh via set_mesh(), we don't need actual geometry data
    # Use a minimal valid geometry object (matches pattern in solve.py)
    if "geometry" not in settings:
        settings["geometry"] = [{
            "type": "ground",
            "height": 0.0,
            "enabled": True,
            "is_obstacle": False
        }]
    
    # C++ JSON schema requires "materials" to be an array with "type" field
    # Convert materials dict to array format if needed (matches pattern in solve.py)
    if "materials" in settings:
        materials = settings["materials"]
        if isinstance(materials, dict) and not isinstance(materials, list):
            # Ensure materials has "type" field
            if "type" not in materials:
                # Infer type from pde if available
                pde = settings.get("pde", "LinearElasticity")
                if pde == "Poisson":
                    materials["type"] = "Laplacian"
                else:
                    materials["type"] = "LinearElasticity"
            # Convert to array format
            settings["materials"] = [materials]
    
    # Convert "selection" to "id" in boundary conditions (required by JSON validator)
    # In array mode, we use boundary IDs (sideset IDs) directly
    if "boundary_conditions" in settings:
        bc = settings["boundary_conditions"]
        
        # Convert dirichlet_boundary
        if "dirichlet_boundary" in bc:
            for bc_item in bc["dirichlet_boundary"]:
                if isinstance(bc_item, dict) and "selection" in bc_item and "id" not in bc_item:
                    # Use selection value as id (they should match in array mode)
                    bc_item["id"] = bc_item.pop("selection")
        
        # Convert neumann_boundary
        if "neumann_boundary" in bc:
            for bc_item in bc["neumann_boundary"]:
                if isinstance(bc_item, dict) and "selection" in bc_item and "id" not in bc_item:
                    # Use selection value as id (they should match in array mode)
                    bc_item["id"] = bc_item.pop("selection")
    
    if differentiable_params is None:
        differentiable_params = ["geometry"]
    
    # Use old API for direct Solver access (required for differentiable operations)
    # Note: pf is already imported above
    solver = pf.Solver()
    
    import json
    # First set settings (including boundary conditions)
    settings_json = json.dumps(settings)
    solver.set_settings(settings_json, strict_validation=False)
    
    # Then set mesh (array mode)
    solver.set_mesh(V_np, C_np.astype(np.int32))
    
    # Set boundary IDs if sidesets_func is provided (required for boundary conditions to work)
    if sidesets_func is not None:
        try:
            mesh = solver.mesh()
            if hasattr(mesh, "set_boundary_ids") and hasattr(mesh, "n_boundary_elements"):
                n_boundary = mesh.n_boundary_elements()
                boundary_ids = []
                
                for i in range(n_boundary):
                    try:
                        v0 = mesh.boundary_element_vertex(i, 0)
                        v1 = mesh.boundary_element_vertex(i, 1)
                        p0 = mesh.point(v0)
                        p1 = mesh.point(v1)
                        center = (p0 + p1) / 2.0
                        bid = sidesets_func(center, True)
                        boundary_ids.append(bid)
                    except Exception:
                        boundary_ids.append(-1)
                
                if boundary_ids:
                    boundary_ids_array = np.array(boundary_ids, dtype=np.int32)
                    mesh.set_boundary_ids(boundary_ids_array)
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to set boundary IDs: {e}", RuntimeWarning)
    
    # Re-apply settings after set_mesh() to ensure boundary conditions are recognized
    # This matches the pattern in solve.py and legacy tests
    solver.set_settings(settings_json, strict_validation=False)
    
    # Build basis and assemble to initialize the solver properly.
    # This ensures boundary conditions are recognized and the solver is in a valid state.
    # Note: solve() will call build_basis() and assemble() again internally, but we need
    # to do it here first to ensure boundary conditions are properly set up.
    # When set_vertices() is called in forward(), we'll rebuild basis and assemble again.
    solver.build_basis()
    solver.assemble()
    
    # Set cache level for differentiable operations (must be set before solve())
    # Note: pf is already imported above
    solver.set_cache_level(pf.CacheLevel.Derivatives)
    
    # Initialize time stepping for transient problems (if needed)
    if "time" in settings and settings["time"]:
        time_config = settings["time"]
        t0 = time_config.get("t0", 0.0) if isinstance(time_config, dict) else 0.0
        dt = time_config.get("dt", 0.01) if isinstance(time_config, dict) else 0.01
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

