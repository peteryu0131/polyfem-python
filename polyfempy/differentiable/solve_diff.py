"""Differentiable solve function for PolyFEM."""

# pyright: reportMissingImports=false
# torch is an optional dependency, so type checker may not find it

import json
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

try:
    import torch  # pyright: ignore[reportMissingImports]
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from ..api.config import SimulationConfig
from .cpp_ext import get_cpp_polyfempy
from .torch_integration import PolyFEMFunction, PolyFEMPerElementMaterialFunction
from .result_diff import DifferentiableResult, DifferentiableMaterialResult
from ._material_parameters import (
    _expand_material_parameter_to_slots,
    _solver_n_element_assembly_slots,
    build_lame_from_youngs,
    solver_body_ids_for_assembly,
    solver_body_slot_mask,
    youngs_to_lame,
    youngs_value_to_internal,
)
from ._solve_settings import (
    _cfg_array_mesh_payload,
    _console_log_level_from_settings,
    _differentiable_config_and_settings,
    _geometry_uses_only_absolute_mesh_paths,
    _solver_set_log_level_off,
    build_solver_from_settings,
)


def solve_differentiable(
    V: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    C: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    cfg: Optional[Union[str, Dict[str, Any], SimulationConfig]] = None,
    root_path: Optional[str] = None,
    body_ids: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    boundary_ids: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    differentiable_params: Optional[List[str]] = None,
    derivative_type: str = "shape",
    sidesets_func: Optional[callable] = None,
    *,
    quiet_polyfem_setup: bool = True,
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
        body_ids: Array-mode only. Optional per-element body ids to copy onto the mesh after ``set_mesh``.
        boundary_ids: Array-mode only. Optional per-boundary-element ids to copy after ``set_mesh``.
        differentiable_params: Parameter names to make differentiable. Default: ["geometry"].
        derivative_type: "shape", "periodic_shape", "material", "initial_velocity", etc.
        quiet_polyfem_setup: If True (default), call ``set_log_level(6)`` right after ``Solver()``
            so setup is quiet; if False, skip that so console verbosity can follow JSON ``log.level``
            during ``solve()`` (see ``PolyFEMFunction`` / ``solve_log_level``).

    Returns:
        DifferentiableResult with .u, .vertices (backward 后 .vertices.grad 为形状导数).
    """
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )

    pf = get_cpp_polyfempy()

    # --- Resolve config and decide: load_mesh (file) vs set_mesh (array) ---
    if cfg is None:
        raise ValueError("cfg is required (JSON path, dict, or SimulationConfig)")
    config, _, settings, _ = _differentiable_config_and_settings(
        cfg,
        root_path=root_path,
    )
    array_payload = _cfg_array_mesh_payload(config)
    if V is None and C is None and array_payload is not None:
        V = array_payload.get("vertices")
        C = array_payload.get("cells")
        if body_ids is None:
            body_ids = array_payload.get("body_ids")
        if boundary_ids is None:
            boundary_ids = array_payload.get("boundary_ids")

    use_load_mesh = (V is None and C is None) and settings.get("geometry")
    if use_load_mesh and not settings.get("root_path") and not _geometry_uses_only_absolute_mesh_paths(settings):
        raise ValueError(
            "Config+mesh mode requires root_path so mesh paths resolve. "
            "Pass cfg as JSON path, or cfg.extras['_root_path'] = ..., or root_path=..."
        )

    if use_load_mesh:
        # --- Config + mesh file path (disk geometry) ---
        #
        # Do **not** call assemble(), set_cache_level(Derivatives), or init_timestepping() here.
        # PolyFEMFunction.forward runs a single legal sequence: set_vertices → build_basis →
        # assemble → set_cache_level → solve(log_level=...), and solve()'s internals own
        # transient timestep setup. A second assemble + init_timestepping before forward used to
        # duplicate / reorder state vs solve_problem, which on large Neo-Hookean + transient
        # could leave State inconsistent or SIGSEGV. Same spirit as api.solve: avoid extra
        # timestep init outside the one solve path.
        #
        solver = pf.Solver()
        if quiet_polyfem_setup:
            _solver_set_log_level_off(solver)
        solver.set_settings(json.dumps(settings), strict_validation=False)
        solver.load_mesh_from_settings()
        solver.build_basis()
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
        if quiet_polyfem_setup:
            _solver_set_log_level_off(solver)
        settings_json = json.dumps(settings)
        solver.set_settings(settings_json, strict_validation=False)
        solver.set_mesh(V_np, C_np.astype(np.int32))
        mesh = solver.mesh()
        if body_ids is not None and hasattr(mesh, "set_body_ids"):
            if isinstance(body_ids, torch.Tensor):
                body_ids_np = body_ids.detach().cpu().numpy().astype(np.int32).reshape(-1)
            else:
                body_ids_np = np.asarray(body_ids, dtype=np.int32).reshape(-1)
            mesh.set_body_ids(body_ids_np)
        if boundary_ids is not None and hasattr(mesh, "set_boundary_ids"):
            if isinstance(boundary_ids, torch.Tensor):
                boundary_ids_np = boundary_ids.detach().cpu().numpy().astype(np.int32).reshape(-1)
            else:
                boundary_ids_np = np.asarray(boundary_ids, dtype=np.int32).reshape(-1)
            mesh.set_boundary_ids(boundary_ids_np)
        if sidesets_func is not None:
            try:
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
        # PolyFEMFunction.forward owns the geometry-dependent sequence:
        # set_vertices -> build_basis -> assemble -> set_cache_level -> solve.
        # Pre-assembling here duplicates large transient matrices in array mode
        # and can trigger std::bad_alloc on optimization runs.
    
    if not use_load_mesh and isinstance(V, torch.Tensor):
        V_torch = V
    elif V_requires_grad:
        V_torch = torch.tensor(V_np, requires_grad=True, dtype=V_dtype, device=V_device)
    else:
        V_torch = torch.tensor(V_np, requires_grad=False, dtype=V_dtype, device=V_device)

    solve_log_level = _console_log_level_from_settings(settings)
    solutions = PolyFEMFunction.apply(solver, V_torch, derivative_type, solve_log_level)

    return DifferentiableResult(
        u=solutions,
        solver=solver,
        derivative_type=derivative_type,
        differentiable_params=differentiable_params,
        vertices=V_torch,
        meta={"_solve_settings": settings},
    )


def prepare_differentiable_simulation(
    V: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    C: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    cfg: Optional[Union[str, Dict[str, Any], SimulationConfig]] = None,
    root_path: Optional[str] = None,
    body_ids: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    boundary_ids: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    differentiable_params: Optional[List[str]] = None,
    derivative_type: str = "shape",
    sidesets_func: Optional[callable] = None,
    *,
    quiet_polyfem_setup: bool = True,
) -> DifferentiableResult:
    """Prepare a differentiable simulation result for torch losses.

    This is the preferred user-facing name for one-off loss/gradient examples:
    it runs the forward simulation, keeps the PolyFEM solver state needed by
    autograd, and returns a ``DifferentiableResult`` that can be passed to
    ``make_von_mises_loss(...)`` followed by ``loss.backward()``.

    ``solve_differentiable(...)`` remains as the backward-compatible low-level
    alias.
    """
    return solve_differentiable(
        V=V,
        C=C,
        cfg=cfg,
        root_path=root_path,
        body_ids=body_ids,
        boundary_ids=boundary_ids,
        differentiable_params=differentiable_params,
        derivative_type=derivative_type,
        sidesets_func=sidesets_func,
        quiet_polyfem_setup=quiet_polyfem_setup,
    )


def solve_differentiable_material(
    solver: Any,
    lam: "torch.Tensor",
    mu: "torch.Tensor",
    *,
    forward_solve_cache: str = "derivatives",
    solve_log_level: int = 2,
) -> DifferentiableMaterialResult:
    """Differentiable solve with **per-element Lamé parameters** as PyTorch inputs.

    This does **not** replace ``solve_differentiable`` (vertex-based). Use it when you need
    ``dL/dλ`` and ``dL/dμ`` from ``elastic_material_derivative`` after ``solve_adjoint``.

    **Scalar Young modulus** ``E``: build ``lam, mu`` from ``E`` (and masks) with ordinary
    torch ops, then call this function; ``loss.backward()`` propagates to ``E`` via the
    chain rule.

    Args:
        solver: Built ``polyfempy`` ``Solver`` (mesh loaded, ``build_basis`` / transient init
            as required by your problem — same state you would pass to ``solve()``).
        lam: 1D tensor, length ``n_element_assembly_slots()`` (same as ``set_per_element_material``).
        mu: Same shape as ``lam``.
        forward_solve_cache: ``"derivatives"`` (default, adjoint), ``"solution"``, or ``"none"``.
        solve_log_level: Passed to ``solver.solve(log_level=...)`` when supported.

    Returns:
        DifferentiableMaterialResult with ``.u`` (displacement) and references ``.lam``, ``.mu``,
        ``.solver``. After ``loss.backward()``, ``lam.grad`` and ``mu.grad`` are filled.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )

    u = PolyFEMPerElementMaterialFunction.apply(
        solver,
        lam,
        mu,
        int(solve_log_level),
        str(forward_solve_cache),
    )
    return DifferentiableMaterialResult(
        u=u,
        solver=solver,
        lam=lam,
        mu=mu,
    )


def solve_differentiable_material_from_youngs(
    solver: Any,
    E: Union[float, "torch.Tensor"],
    nu: Union[float, "torch.Tensor"],
    *,
    slot_mask: Optional["torch.Tensor"] = None,
    other_E: Optional[Union[float, "torch.Tensor"]] = None,
    other_nu: Optional[Union[float, "torch.Tensor"]] = None,
    forward_solve_cache: str = "derivatives",
    solve_log_level: int = 2,
) -> DifferentiableMaterialResult:
    """Differentiable material solve using ``E, nu`` instead of explicit ``lam, mu``.

    This is the user-facing wrapper most examples should prefer. Internally it
    still converts ``E, nu`` to Lamé parameters because the current low-level
    PolyFEM material path is built on ``set_per_element_material(lambda, mu)``.
    """
    if slot_mask is None:
        n_slots = _solver_n_element_assembly_slots(solver)
        E_t = torch.as_tensor(E, dtype=torch.get_default_dtype())
        E_full = _expand_material_parameter_to_slots(
            E,
            n_slots=n_slots,
            dtype=E_t.dtype,
            device=E_t.device,
        )
        nu_full = _expand_material_parameter_to_slots(
            nu,
            n_slots=n_slots,
            dtype=E_t.dtype,
            device=E_t.device,
        )
        lam, mu = youngs_to_lame(E_full, nu_full)
    else:
        lam, mu = build_lame_from_youngs(
            E,
            nu,
            slot_mask=slot_mask,
            other_E=other_E,
            other_nu=other_nu,
        )
    return solve_differentiable_material(
        solver=solver,
        lam=lam,
        mu=mu,
        forward_solve_cache=forward_solve_cache,
        solve_log_level=solve_log_level,
    )
