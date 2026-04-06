"""Differentiable solve function for PolyFEM."""

# pyright: reportMissingImports=false
# torch is an optional dependency, so type checker may not find it

import numpy as np
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

try:
    import torch  # pyright: ignore[reportMissingImports]
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from ..api.config import SimulationConfig
from .cpp_ext import get_cpp_polyfempy
from .torch_integration import PolyFEMFunction
from .result_diff import DifferentiableResult


def _solver_set_log_level_off(solver: Any) -> None:
    """Best-effort: some PolyFEM Python bindings omit ``set_log_level``."""
    if hasattr(solver, "set_log_level"):
        try:
            solver.set_log_level(6)  # 6=off when supported
        except Exception:
            pass


def solve_differentiable(
    V: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    C: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    cfg: Optional[Union[str, Dict[str, Any], SimulationConfig]] = None,
    root_path: Optional[str] = None,
    differentiable_params: Optional[List[str]] = None,
    derivative_type: str = "shape",
    sidesets_func: Optional[callable] = None
) -> DifferentiableResult:
    """Solve with automatic gradient computation.

    Returns PyTorch tensors supporting .backward() via adjoint method.
    API 与 api.solve() 对齐：cfg 支持 JSON 路径、dict、或 SimulationConfig（含你用 API 类构建的配置）。

    Two usage modes:

    1) Config + mesh file (recommended): pass cfg only. Mesh is loaded via
       load_mesh_from_settings(). cfg 可以是:
       - str: JSON 文件路径（自动用该路径作为 root_path 解析 mesh 相对路径）
       - SimulationConfig: 用 API 类构建，如 SimulationConfig(geometry=Geometry(...), materials=...).
         若 geometry 里 mesh 是相对路径，需设 cfg.extras["_root_path"] = "目录路径" 或传 root_path=...
       - dict: 与 SimulationConfig.to_dict() 格式一致，可含 root_path 或由 extras["_root_path"] 提供

    2) Array mode: pass V, C, and cfg. Mesh is set via set_mesh(V, C). You may need
       sidesets_func to assign boundary IDs if the mesh has no sidesets.

    Args:
        V: (N, dim) vertices, or None when using config+mesh mode.
        C: (M, k) cells, or None when using config+mesh mode.
        cfg: JSON path (str), dict, or SimulationConfig (e.g. from API classes).
        root_path: Optional. When using config+mesh with relative paths, overrides/sets
            root_path for resolving mesh files (e.g. root_path=str(Path("data").resolve())).
        differentiable_params: Parameter names to make differentiable. Default: ["geometry"].
        derivative_type: "shape", "periodic_shape", "material", "initial_velocity", etc.

    Returns:
        DifferentiableResult with .u, .vertices (backward 后 .vertices.grad 为形状导数).
    """
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )

    pf = get_cpp_polyfempy()

    import json
    from pathlib import Path

    # --- Resolve config and decide: load_mesh (file) vs set_mesh (array) ---
    if cfg is None:
        raise ValueError("cfg is required (JSON path, dict, or SimulationConfig)")
    config_root = None  # 从 cfg 解析出的 root_path（用于解析 mesh 相对路径）
    if isinstance(cfg, str):
        config = SimulationConfig.from_json_file(cfg)
        config_root = str(Path(cfg).resolve())
    elif isinstance(cfg, dict):
        config = SimulationConfig.from_json_dict(cfg)
        config_root = cfg.get("root_path") or (config.extras or {}).get("_root_path")
    elif isinstance(cfg, SimulationConfig):
        config = cfg
        config_root = (getattr(config, "extras", None) or {}).get("_root_path")
        if not config_root and hasattr(config, "to_dict"):
            config_root = config.to_dict().get("root_path")
    else:
        raise ValueError(f"cfg must be str, dict, or SimulationConfig, got {type(cfg)}")
    
    settings = config.to_dict()
    # root_path：用户传入的 root_path 参数优先；否则用 config 里的
    root_path_resolved = root_path if root_path is not None else config_root
    if not root_path_resolved:
        root_path_resolved = settings.get("root_path")
    if root_path_resolved:
        settings["root_path"] = root_path_resolved

    use_load_mesh = (V is None and C is None) and settings.get("geometry")
    if use_load_mesh and not settings.get("root_path"):
        raise ValueError(
            "Config+mesh mode requires root_path so mesh paths resolve. "
            "Pass cfg as JSON path, or cfg.extras['_root_path'] = ..., or root_path=..."
        )

    if use_load_mesh:
        # --- Config + mesh file path (same as legacy / differentiable_minimal) ---
        solver = pf.Solver()
        _solver_set_log_level_off(solver)
        solver.set_settings(json.dumps(settings), strict_validation=False)
        solver.load_mesh_from_settings()
        solver.build_basis()
        solver.assemble()
        solver.set_cache_level(pf.CacheLevel.Derivatives)
        if "time" in settings and settings["time"]:
            tcfg = settings["time"]
            t0 = tcfg.get("t0", 0.0) if isinstance(tcfg, dict) else 0.0
            dt = tcfg.get("dt", 0.01) if isinstance(tcfg, dict) else 0.01
            solver.init_timestepping(t0, dt)
        mesh = solver.mesh()
        V_np = np.asarray(mesh.vertices(), dtype=np.float64)
        V_requires_grad = True
        V_device = torch.device("cpu")
        V_dtype = torch.float64
        if differentiable_params is None:
            differentiable_params = ["geometry"]
    else:
        # --- Array mode: set_mesh(V, C) ---
        if V is None or C is None:
            raise ValueError("When cfg has no geometry or root_path, both V and C are required")
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

        # Normalize settings for array mode
        if "geometry" not in settings:
            settings["geometry"] = [{"type": "ground", "height": 0.0, "enabled": True, "is_obstacle": False}]
        if "materials" in settings:
            materials = settings["materials"]
            if isinstance(materials, dict) and not isinstance(materials, list):
                if "type" not in materials:
                    pde = settings.get("pde", "LinearElasticity")
                    materials["type"] = "Laplacian" if pde == "Poisson" else "LinearElasticity"
                settings["materials"] = [materials]
        if "boundary_conditions" in settings:
            bc = settings["boundary_conditions"]
            for key in ("dirichlet_boundary", "neumann_boundary"):
                if key not in bc:
                    continue
                for item in bc[key]:
                    if isinstance(item, dict) and "selection" in item and "id" not in item:
                        item["id"] = item.pop("selection")
        if differentiable_params is None:
            differentiable_params = ["geometry"]

        solver = pf.Solver()
        _solver_set_log_level_off(solver)
        settings_json = json.dumps(settings)
        solver.set_settings(settings_json, strict_validation=False)
        solver.set_mesh(V_np, C_np.astype(np.int32))
        if sidesets_func is not None:
            try:
                mesh = solver.mesh()
                if hasattr(mesh, "set_boundary_ids") and hasattr(mesh, "n_boundary_elements"):
                    n_b = mesh.n_boundary_elements()
                    ids = []
                    for i in range(n_b):
                        try:
                            p0 = mesh.point(mesh.boundary_element_vertex(i, 0))
                            p1 = mesh.point(mesh.boundary_element_vertex(i, 1))
                            bid = sidesets_func((p0 + p1) / 2.0, True)
                            ids.append(bid)
                        except Exception:
                            ids.append(-1)
                    if ids:
                        mesh.set_boundary_ids(np.array(ids, dtype=np.int32))
            except Exception as e:
                import warnings
                warnings.warn(f"Failed to set boundary IDs: {e}", RuntimeWarning)
        solver.set_settings(settings_json, strict_validation=False)
        solver.build_basis()
        solver.assemble()
        solver.set_cache_level(pf.CacheLevel.Derivatives)
        if "time" in settings and settings["time"]:
            tcfg = settings["time"]
            t0 = tcfg.get("t0", 0.0) if isinstance(tcfg, dict) else 0.0
            dt = tcfg.get("dt", 0.01) if isinstance(tcfg, dict) else 0.01
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
        differentiable_params=differentiable_params,
        vertices=V_torch,
    )

