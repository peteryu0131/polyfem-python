"""Differentiable solve function for PolyFEM."""

# pyright: reportMissingImports=false
# torch is an optional dependency, so type checker may not find it

import json
from pathlib import Path
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


def _console_log_level_from_settings(settings: Dict[str, Any]) -> int:
    """Map ``output.log.level`` string to int 0–6 (same contract as ``api.solve`` → ``solver.solve``)."""
    log_level = 2
    out = settings.get("output")
    if isinstance(out, dict):
        log = out.get("log")
        if isinstance(log, dict):
            raw = log.get("level", "info")
            log_level_str = raw.strip().lower() if isinstance(raw, str) else str(raw).strip().lower()
            log_level_map = {
                "trace": 0,
                "debug": 1,
                "info": 2,
                "warn": 3,
                "warning": 3,
                "error": 4,
                "critical": 5,
                "off": 6,
            }
            if log_level_str in log_level_map:
                log_level = log_level_map[log_level_str]
    return log_level


def _solver_set_log_level_off(solver: Any) -> None:
    """Best-effort: some PolyFEM Python bindings omit ``set_log_level``."""
    if hasattr(solver, "set_log_level"):
        try:
            solver.set_log_level(6)  # 6=off when supported
        except Exception:
            pass


def _apply_internal_differentiable_runtime_patches(
    *,
    config: Optional[SimulationConfig],
    settings: Dict[str, Any],
) -> None:
    """Apply small runtime fixes needed by differentiable solves.

    This keeps user-facing examples clean: callers should not need to know that
    differentiable contact currently expects a constant barrier stiffness.
    """
    solver_settings = settings.get("solver")
    if not isinstance(solver_settings, dict):
        return

    contact_settings = solver_settings.get("contact")
    if isinstance(contact_settings, dict) and "barrier_stiffness" in contact_settings:
        contact_settings["barrier_stiffness"] = 1e3

    if config is None:
        return

    solver_cfg = getattr(config, "solver", None)
    contact_cfg = getattr(solver_cfg, "contact", None)
    if contact_cfg is not None and hasattr(contact_cfg, "barrier_stiffness"):
        contact_cfg.barrier_stiffness = 1e3


def _geometry_uses_only_absolute_mesh_paths(settings: Dict[str, Any]) -> bool:
    """Return True when every geometry mesh path is already absolute.

    In that case, ``root_path`` is not needed even in config+mesh mode.
    """
    geometry = settings.get("geometry")
    if not isinstance(geometry, list) or not geometry:
        return False

    found_mesh = False
    for item in geometry:
        if not isinstance(item, dict):
            return False
        mesh = item.get("mesh")
        if mesh is None:
            continue
        if not isinstance(mesh, str) or not mesh.strip():
            return False
        found_mesh = True
        if not Path(mesh).is_absolute():
            return False
    return found_mesh


def _cfg_array_mesh_payload(config: SimulationConfig) -> Optional[Dict[str, Any]]:
    extras = getattr(config, "extras", None)
    if isinstance(extras, dict):
        payload = extras.get("_mesh_array_mode")
        if isinstance(payload, dict):
            return payload
    return None


def _differentiable_config_and_settings(
    cfg: Union[str, Dict[str, Any], SimulationConfig],
    *,
    root_path: Optional[str] = None,
    apply_runtime_patches: bool = True,
) -> tuple[SimulationConfig, Optional[SimulationConfig], Dict[str, Any], Optional[str]]:
    """Normalize user configs for differentiable solve entry points.

    ``prepare_differentiable_simulation`` and material optimization both need
    the same config coercion, root-path handling, and differentiable runtime
    patches. Keeping that here avoids example-specific patch calls.
    """
    config_root = None
    config_obj: Optional[SimulationConfig] = None
    if isinstance(cfg, str):
        config = SimulationConfig.from_json_file(cfg)
        config_obj = config
        config_root = str(Path(cfg).resolve())
    elif isinstance(cfg, dict):
        config = SimulationConfig.from_json_dict(cfg)
        config_root = cfg.get("root_path") or (config.extras or {}).get("_root_path")
    elif isinstance(cfg, SimulationConfig):
        config = cfg
        config_obj = config
        config_root = (getattr(config, "extras", None) or {}).get("_root_path")
        if not config_root and hasattr(config, "to_dict"):
            config_root = config.to_dict().get("root_path")
    else:
        raise ValueError(f"cfg must be str, dict, or SimulationConfig, got {type(cfg)}")

    settings = config.to_dict()
    if apply_runtime_patches:
        _apply_internal_differentiable_runtime_patches(
            config=config_obj,
            settings=settings,
        )

    root_path_resolved = root_path if root_path is not None else config_root
    if not root_path_resolved:
        root_path_resolved = settings.get("root_path")
    if root_path_resolved:
        settings["root_path"] = root_path_resolved

    return config, config_obj, settings, root_path_resolved


def build_solver_from_settings(
    settings: Dict[str, Any],
    *,
    quiet_polyfem_setup: bool = True,
) -> Any:
    """Build a solver from config-style settings for repeated differentiable solves."""
    pf = get_cpp_polyfempy()
    solver = pf.Solver()
    if quiet_polyfem_setup:
        _solver_set_log_level_off(solver)
    solver.set_settings(json.dumps(settings), strict_validation=False)
    solver.load_mesh_from_settings()
    solver.build_basis()
    if hasattr(solver, "assemble"):
        solver.assemble()
    return solver


def solver_body_ids_for_assembly(solver: Any) -> np.ndarray:
    """Return body ids aligned with per-element assembly slots."""
    n_slots = _solver_n_element_assembly_slots(solver)
    if hasattr(solver, "get_body_ids_for_assembly"):
        ids = np.asarray(solver.get_body_ids_for_assembly(), dtype=np.int32).reshape(-1)
        if int(ids.size) != int(n_slots):
            raise RuntimeError(
                f"get_body_ids_for_assembly length {ids.size} != n_element_assembly_slots {n_slots}"
            )
        return ids

    mesh = solver.mesh()
    body_ids = np.asarray(mesh.get_body_ids(), dtype=np.int32).reshape(-1)
    n_el = int(mesh.n_elements())
    if body_ids.size != n_el:
        raise RuntimeError(
            f"get_body_ids length {body_ids.size} != n_elements {n_el}; check mesh / PolyFEM version."
        )
    if n_el < n_slots:
        raise RuntimeError(
            f"mesh n_elements={n_el} < assembly slots {n_slots}; cannot build material vectors."
        )
    return body_ids[:n_slots]


def solver_body_slot_mask(solver: Any, *, body_id: int) -> "torch.Tensor":
    """Boolean mask over assembly slots for one body id."""
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )
    body_ids = solver_body_ids_for_assembly(solver)
    return torch.as_tensor(body_ids == int(body_id), dtype=torch.bool)


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

    import json
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


def youngs_to_lame(
    E: "torch.Tensor",
    nu: Union[float, "torch.Tensor"],
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Convert isotropic ``E, nu`` parameters to Lamé ``lambda, mu``.

    This helper is intentionally public so user-facing code can stay in the
    more familiar ``E, nu`` parameterization while the current low-level
    differentiable material path still works with per-element Lamé tensors.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )

    nu_t = torch.as_tensor(nu, dtype=E.dtype, device=E.device)
    lam = E * nu_t / ((1.0 + nu_t) * (1.0 - 2.0 * nu_t))
    mu = E / (2.0 * (1.0 + nu_t))
    return lam, mu


def _pressure_unit_to_internal_scale(
    *,
    units_length: str,
    units_mass: str,
    units_time: str,
    pressure_unit: str,
) -> float:
    """Scale a named pressure unit into the current solver unit system."""
    length_to_m = {"m": 1.0, "cm": 1e-2, "mm": 1e-3}
    mass_to_kg = {"kg": 1.0, "g": 1e-3}
    time_to_s = {"s": 1.0}
    pressure_to_pa = {"Pa": 1.0, "kPa": 1e3, "MPa": 1e6, "GPa": 1e9}

    try:
        length_scale = length_to_m[str(units_length)]
        mass_scale = mass_to_kg[str(units_mass)]
        time_scale = time_to_s[str(units_time)]
        pressure_pa = pressure_to_pa[str(pressure_unit)]
    except KeyError as exc:
        raise ValueError(
            "unsupported unit conversion for differentiable material solve: "
            f"length={units_length!r}, mass={units_mass!r}, time={units_time!r}, "
            f"pressure={pressure_unit!r}"
        ) from exc

    internal_pressure_pa = mass_scale / (length_scale * (time_scale ** 2))
    return pressure_pa / internal_pressure_pa


def _units_triplet_from_settings(settings: dict[str, Any]) -> tuple[str, str, str]:
    units = settings.get("units", {})
    if not isinstance(units, dict):
        units = {}
    return (
        str(units.get("length", "m")),
        str(units.get("mass", "kg")),
        str(units.get("time", "s")),
    )


def youngs_value_to_internal(
    value: Union[float, "torch.Tensor"],
    *,
    pressure_unit: str,
    solver_settings: dict[str, Any],
) -> Union[float, "torch.Tensor"]:
    """Convert user-facing ``E`` values (MPa/GPa/...) into solver internal units."""
    units_length, units_mass, units_time = _units_triplet_from_settings(solver_settings)
    scale = _pressure_unit_to_internal_scale(
        units_length=units_length,
        units_mass=units_mass,
        units_time=units_time,
        pressure_unit=str(pressure_unit),
    )
    return value * scale


def _expand_material_parameter_to_slots(
    value: Union[float, "torch.Tensor"],
    *,
    n_slots: int,
    dtype: "torch.dtype",
    device: "torch.device",
) -> "torch.Tensor":
    """Expand a scalar or per-slot material parameter to ``(n_slots,)``."""
    t = torch.as_tensor(value, dtype=dtype, device=device)
    if t.ndim == 0:
        return t.reshape(()).expand(n_slots)
    if t.numel() != int(n_slots):
        raise ValueError(
            f"material parameter has {t.numel()} values but expected 1 or {n_slots}"
        )
    return t.reshape(n_slots)


def _solver_n_element_assembly_slots(solver: Any) -> int:
    """Length required by ``set_per_element_material``."""
    if hasattr(solver, "n_element_assembly_slots"):
        return int(solver.n_element_assembly_slots())
    if hasattr(solver, "n_bases"):
        return int(solver.n_bases())
    raise AttributeError("solver does not expose n_element_assembly_slots() or n_bases()")


def build_lame_from_youngs(
    E: Union[float, "torch.Tensor"],
    nu: Union[float, "torch.Tensor"],
    *,
    slot_mask: Optional["torch.Tensor"] = None,
    other_E: Optional[Union[float, "torch.Tensor"]] = None,
    other_nu: Optional[Union[float, "torch.Tensor"]] = None,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Build Lamé tensors from ``E, nu`` for one or two material regions.

    Args:
        E: Primary Young's modulus. May be a scalar tensor/float or a per-slot tensor.
        nu: Primary Poisson ratio. Scalar or per-slot.
        slot_mask: Optional bool mask over assembly slots. ``True`` entries use
            ``E, nu``; ``False`` entries use ``other_E, other_nu``.
        other_E: Secondary Young's modulus when ``slot_mask`` is provided.
        other_nu: Secondary Poisson ratio when ``slot_mask`` is provided.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )

    E_t = torch.as_tensor(E, dtype=torch.get_default_dtype())
    if slot_mask is None:
        return youngs_to_lame(E_t, nu)

    if other_E is None or other_nu is None:
        raise ValueError("slot_mask requires other_E and other_nu for the non-masked region")

    mask = torch.as_tensor(slot_mask, dtype=torch.bool, device=E_t.device).reshape(-1)
    n_slots = int(mask.numel())

    E_primary = _expand_material_parameter_to_slots(
        E,
        n_slots=n_slots,
        dtype=E_t.dtype,
        device=E_t.device,
    )
    nu_primary = _expand_material_parameter_to_slots(
        nu,
        n_slots=n_slots,
        dtype=E_t.dtype,
        device=E_t.device,
    )
    E_secondary = _expand_material_parameter_to_slots(
        other_E,
        n_slots=n_slots,
        dtype=E_t.dtype,
        device=E_t.device,
    )
    nu_secondary = _expand_material_parameter_to_slots(
        other_nu,
        n_slots=n_slots,
        dtype=E_t.dtype,
        device=E_t.device,
    )

    E_full = torch.where(mask, E_primary, E_secondary)
    nu_full = torch.where(mask, nu_primary, nu_secondary)
    return youngs_to_lame(E_full, nu_full)


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
