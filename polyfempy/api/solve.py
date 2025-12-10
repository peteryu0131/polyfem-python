import numpy as np
from . import tensor as T
from .result import Result


def _first_attr(obj, *names):
    """Return the first available attribute name on an object.

    Args:
        obj: Any Python object to inspect.
        *names: Candidate attribute names to probe in order.

    Returns:
        The first name that exists on `obj`, or None if none are present.

    Notes:
        - Useful for adapting across multiple polyfempy versions with
          slightly different method names.
    """
    for n in names:
        if hasattr(obj, n):
            return n
    return None


def _ensure_i32(cells):
    """Ensure cell connectivity array has dtype int32.

    If `cells` already has dtype int32, it is returned as-is; otherwise
    a view/copy with dtype int32 is returned (no copy if possible).

    Args:
        cells: Array-like cell connectivity of shape (M, k).

    Returns:
        NumPy array with dtype `np.int32`.
    """
    return cells.astype(np.int32, copy=False) if cells.dtype != np.int32 else cells


def solve(vertices=None, cells=None, cfg=None, sidesets_func=None, dtype=None):
    """Unified high-level solve entry point.

    This adapter normalizes user arrays (NumPy / Torch / JAX) to a CPU,
    C-contiguous NumPy representation, constructs a backend Settings/Problem
    from `SimulationConfig`, adapts to different polyfempy versions via
    light reflection, applies minimal boundary data, runs the solver, and
    wraps outputs into a `Result`.

    Args:
        vertices: Point coordinates of shape (N, dim). Can be NumPy / Torch / JAX.
                  If None and cfg contains geometry (mesh files), mesh will be loaded from files.
        cells: Cell connectivity of shape (M, k). Can be NumPy / Torch / JAX.
               If None and cfg contains geometry (mesh files), mesh will be loaded from files.
        cfg: A `SimulationConfig` instance, dict, or str (file path) containing 
             full PolyFEM JSON configuration. If dict, it will be converted to 
             SimulationConfig automatically. If str, treated as JSON file path.
        sidesets_func: Optional callable to build side sets from geometry if
            supported by the current polyfempy version.
        dtype: Optional NumPy dtype for vertices (e.g., `np.float64`). If
            provided, `vertices` are cast to this dtype during normalization.

    Returns:
        Result: A result object with:
            - backend: Original backend of `vertices` ('numpy'|'torch'|'jax').
            - vertices/cells: NumPy arrays suitable for export/post-processing.
            - fields: Dict with at least "u" (displacement or generic solution).
            - meta: Auxiliary info (e.g., solver_type).

    Raises:
        RuntimeError: If C++ bindings are missing, polyfempy cannot be imported,
            no compatible solver constructor/methods are found, or solution
            getters are unknown.

    Notes:
        - Numerical work is fully executed by the polyfempy C++ bindings.
          This function only handles data normalization and version bridging.
        - Dummy safety guard: if `cfg.to_settings()` returns an object with
          `_is_dummy=True`, an error is raised immediately to prevent "fake runs".
        - Multi-version compatibility uses `_first_attr(...)` to locate
          available method names (e.g., `settings`/`set_settings`,
          `set_mesh`/`set_mesh_data`).
        - Minimal BCs supported here:
            * Dirichlet via `dirichlet_boundary`: [{'id': int, 'value': ...}, ...]
            * Body force / RHS via `rhs`.
          Extensions (Neumann/pressure, etc.) can be added later.
    """
    import json
    from .config import SimulationConfig
    
    # 0) Handle input - validate and convert to SimulationConfig
    if cfg is None:
        raise ValueError("cfg (configuration) is required")
    
    # 0.1) Handle dict/str input - convert to SimulationConfig
    if isinstance(cfg, dict):
        cfg = SimulationConfig.from_json_dict(cfg)
    elif isinstance(cfg, str):
        # Assume it's a file path
        cfg = SimulationConfig.from_json_file(cfg)
    elif not isinstance(cfg, SimulationConfig):
        raise TypeError(f"cfg must be SimulationConfig, dict, or str (file path), got {type(cfg).__name__}")
    
    # Check if we have full JSON config (geometry, time, etc.)
    full_json = None
    if hasattr(cfg, "extras") and cfg.extras and "_full_json_config" in cfg.extras:
        full_json = cfg.extras["_full_json_config"]
    
    # Check if full_json contains geometry (indicates mesh should be loaded from files)
    has_geometry_in_json = full_json is not None and "geometry" in full_json
    
    # If we have full JSON with geometry and no vertices/cells, use JSON mode (load mesh from files)
    use_json_mode = has_geometry_in_json and (vertices is None or cells is None)
    
    # 1) Normalize inputs to NumPy (if provided)
    if vertices is not None and cells is not None:
        V_np, v_backend = T.as_numpy(vertices, dtype=dtype)
        C_np, _ = T.as_numpy(cells, dtype=np.int32)
        C_np = _ensure_i32(C_np)
    else:
        V_np, C_np, v_backend = None, None, "numpy"
        if not use_json_mode:
            raise ValueError("Either provide vertices/cells arrays, or use JSON config with geometry (mesh files)")

    # 2) Import polyfempy (required for real execution).
    try:
        import polyfempy as pf
    except Exception:
        raise RuntimeError("polyfempy bindings not found. Please install/compile them first.")

    # 3) Construct solver/state (support different versions).
    solver = None
    for ctor in ("Solver", "State"):
        if hasattr(pf, ctor):
            try:
                solver = getattr(pf, ctor)()
                break
            except Exception:
                # Some versions may require extra args; try next name.
                pass
    if solver is None:
        raise RuntimeError("No usable Solver/State constructor found in polyfempy.")

    # 4) Apply Settings - use JSON mode if we have full JSON config
    if use_json_mode:
        # Use full JSON configuration directly
        config_json = json.dumps(full_json)
        name = _first_attr(solver, "settings", "set_settings")
        if not name:
            raise RuntimeError("Missing settings(...) / set_settings(...) on solver.")
        getattr(solver, name)(config_json, strict_validation=False)
        
        # Load mesh from settings (geometry in JSON)
        if hasattr(solver, "load_mesh_from_settings"):
            solver.load_mesh_from_settings()
        else:
            raise RuntimeError("JSON mode requires load_mesh_from_settings() method")
    else:
        # Use SimulationConfig mode (existing behavior)
        settings = cfg.to_settings()
        if getattr(settings, "_is_dummy", False):
            raise RuntimeError("DummySettings: C++ bindings not found; cannot run a real solve.")
        
        name = _first_attr(solver, "settings", "set_settings")
        if not name:
            raise RuntimeError("Missing settings(...) / set_settings(...) on solver.")
        getattr(solver, name)(settings)

    # 5) Set mesh - only if not using JSON mode (mesh already loaded)
    if not use_json_mode:
        set_mesh_ok = False
        for name in ("set_mesh", "set_mesh_data", "load_mesh_from_points"):
            if hasattr(solver, name):
                fn = getattr(solver, name)
                try:
                    fn(V_np, C_np)  # common signature
                    set_mesh_ok = True
                    break
                except TypeError:
                    # Some versions require keywords.
                    try:
                        fn(points=V_np, cells=C_np)
                        set_mesh_ok = True
                        break
                    except Exception:
                        pass
        if not set_mesh_ok:
            raise RuntimeError("No mesh setter found (set_mesh / set_mesh_data / load_mesh_from_points).")

    # 6) Build basis and assemble (required for JSON mode or if needed)
    if use_json_mode:
        if hasattr(solver, "build_basis"):
            solver.build_basis()
        if hasattr(solver, "assemble"):
            solver.assemble()
    
    # 7) Side sets (optional).
    if sidesets_func is not None:
        name = _first_attr(solver, "set_sidesets_from_function", "build_sidesets_from_function")
        if name:
            getattr(solver, name)(sidesets_func)
    
    # 8) Boundary conditions - only apply if not using JSON mode (BCs in JSON)
    if not use_json_mode:
        bc_raw = getattr(cfg, "boundary_conditions", {}) or {}
        # Convert BoundaryConditions class to dict if needed
        if hasattr(bc_raw, "to_dict"):
            bc = bc_raw.to_dict()
        else:
            bc = bc_raw if isinstance(bc_raw, dict) else {}

        # 8.1 Dirichlet boundaries: [{'id': 4, 'value': [0,0(,0)]}, ...]
        entries = bc.get("dirichlet_boundary")
        if entries:
            add_name = _first_attr(solver, "add_dirichlet_boundary", "set_dirichlet_boundary")
            if not add_name:
                raise RuntimeError("No Dirichlet boundary API found on solver.")
            add = getattr(solver, add_name)
            for ent in entries:
                sid = int(ent.get("id", -1))
                val = ent.get("value", 0.0)
                add(sid, val)

        # 8.2 Body force / RHS: scalar or per-dimension vector.
        rhs = bc.get("rhs")
        if rhs is not None:
            name = _first_attr(solver, "set_body_force", "set_rhs", "add_rhs")
            if name:
                getattr(solver, name)(rhs)

    # 9) Run solve (`solve` / `solve_problem` / `run`).
    name = _first_attr(solver, "solve", "solve_problem", "run")
    if not name:
        raise RuntimeError("No solver entry point found (solve / solve_problem / run).")
    getattr(solver, name)()

    # 10) Fetch solution and additional fields (displacement, stress, strain, energy, etc.)
    fields = {}
    meta = {"solver_type": type(solver).__name__}
    
    # 10.1 Sampled solution: typically returns (pts, tris, el_id, bid, fun).
    if hasattr(solver, "get_sampled_solution"):
        out = solver.get_sampled_solution()
        if isinstance(out, (list, tuple)) and len(out) >= 5:
            pts = np.asarray(out[0])
            fun = np.asarray(out[4])
            fields["u"] = fun
            
            # Get cells from solver if available
            cells_result = C_np
            if hasattr(solver, "get_cells"):
                try:
                    cells_result = np.asarray(solver.get_cells())
                except Exception:
                    pass
            
            # Try to get additional fields
            _extract_additional_fields(solver, fields, meta)
            
            return Result(
                v_backend,
                pts,
                cells_result,
                fields,
                meta=meta,
            )

    # 10.2 Direct solution: `get_solution` / `get_displacement` / `get_u`.
    u = None
    for name in ("get_solution", "get_displacement", "get_u"):
        if hasattr(solver, name):
            u = np.asarray(getattr(solver, name)())
            fields["u"] = u
            break
    
    if u is None:
        raise RuntimeError("Failed to retrieve solution: no known getters (sampled or direct).")
    
    # Prefer solver-provided vertices if available
    V_pts = None
    for vn in ("get_vertices", "get_points"):
        if hasattr(solver, vn):
            V_pts = np.asarray(getattr(solver, vn)())
            break
    if V_pts is None:
        V_pts = V_np if V_np is not None else np.array([])
    
    # Get cells from solver if available
    cells_result = C_np
    if hasattr(solver, "get_cells"):
        try:
            cells_result = np.asarray(solver.get_cells())
        except Exception:
            pass
    elif cells_result is None:
        cells_result = np.array([])
    
    # Extract additional fields (stress, strain, energy, etc.)
    _extract_additional_fields(solver, fields, meta)
    
    return Result(
        v_backend,
        V_pts,
        cells_result,
        fields,
        meta=meta,
    )


def _extract_additional_fields(solver, fields: dict, meta: dict):
    """Extract additional fields from solver (stress, strain, energy, etc.).
    
    Args:
        solver: Solver object with potential getter methods.
        fields: Dictionary to populate with field data.
        meta: Dictionary to populate with metadata.
    """
    # Try to get stress
    for name in ("get_stress", "get_cauchy_stress", "stress"):
        if hasattr(solver, name):
            try:
                stress = getattr(solver, name)()
                if stress is not None:
                    fields["stress"] = np.asarray(stress)
                    break
            except Exception:
                pass
    
    # Try to get strain
    for name in ("get_strain", "strain"):
        if hasattr(solver, name):
            try:
                strain = getattr(solver, name)()
                if strain is not None:
                    fields["strain"] = np.asarray(strain)
                    break
            except Exception:
                pass
    
    # Try to get energy
    for name in ("get_energy", "energy", "total_energy"):
        if hasattr(solver, name):
            try:
                energy = getattr(solver, name)()
                if energy is not None:
                    if isinstance(energy, (int, float)):
                        meta["energy"] = float(energy)
                    else:
                        fields["energy"] = np.asarray(energy)
                    break
            except Exception:
                pass
    
    # Try to get pressure (for Stokes/fluid problems)
    for name in ("get_pressure", "pressure"):
        if hasattr(solver, name):
            try:
                pressure = getattr(solver, name)()
                if pressure is not None:
                    fields["p"] = np.asarray(pressure)
                    break
            except Exception:
                pass
    
    # Try to get velocity (for fluid problems)
    for name in ("get_velocity", "velocity"):
        if hasattr(solver, name):
            try:
                velocity = getattr(solver, name)()
                if velocity is not None:
                    fields["v"] = np.asarray(velocity)
                    break
            except Exception:
                pass
    
    # Try to get solver statistics
    for name in ("get_stats", "stats", "get_log"):
        if hasattr(solver, name):
            try:
                stats = getattr(solver, name)()
                if stats is not None:
                    if isinstance(stats, dict):
                        meta.update(stats)
                    break
            except Exception:
                pass
