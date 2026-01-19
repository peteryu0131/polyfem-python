"""
Route A only:
- Always `import polyfempy as pf` (stable C++ extension module name).
- nanobind vs pybind11 is a build-time backend choice and must not change Python imports.

Important:
- The C++ binding's `Solver.solve()` returns a tuple `(sol, pressure)`, so Python must
  capture and parse the return value instead of assuming getters exist.
"""

import numpy as np
from . import tensor as T
from .result import Result


def _first_attr(obj, *names):
    """Return first available attribute name.

    Supports multiple polyfempy versions with different method names.
    """
    for n in names:
        if hasattr(obj, n):
            return n
    return None


def _ensure_i32(cells):
    """Ensure cells are int32 (no copy if already int32)."""
    return cells.astype(np.int32, copy=False) if cells.dtype != np.int32 else cells


def solve(vertices=None, cells=None, cfg=None, sidesets_func=None, dtype=None):
    """High-level solver wrapper.

    Normalizes inputs, loads mesh or config, dispatches to PolyFEM.
    Handles both JSON-based and array-based setups.

    Args:
        vertices: (N, dim) array or None. If None and cfg has geometry, loads from files.
        cells: (M, k) array or None.
        cfg: SimulationConfig, dict, or JSON file path.
        sidesets_func: Optional callable for side sets.
        dtype: Optional dtype for vertices.

    Returns:
        Result with fields, vertices, cells, meta.
    """
    import json
    from .config import SimulationConfig
    
    if cfg is None:
        raise ValueError("cfg (configuration) is required")
    
    if isinstance(cfg, dict):
        cfg = SimulationConfig.from_json_dict(cfg)
    elif isinstance(cfg, str):
        cfg = SimulationConfig.from_json_file(cfg)
    elif not isinstance(cfg, SimulationConfig):
        raise TypeError(f"cfg must be SimulationConfig, dict, or str (file path), got {type(cfg).__name__}")
    
    # Check for full JSON config (geometry, time, etc.)
    full_json = None
    if hasattr(cfg, "extras") and cfg.extras and "_full_json_config" in cfg.extras:
        full_json = cfg.extras["_full_json_config"]
    
    # JSON mode: load mesh from files if geometry in JSON and no vertices/cells provided
    use_json_mode = (full_json is not None and "geometry" in full_json and 
                     (vertices is None or cells is None))
    
    # Normalize inputs to NumPy
    if vertices is not None and cells is not None:
        V_np, v_backend = T.as_numpy(vertices, dtype=dtype)
        C_np, _ = T.as_numpy(cells, dtype=np.int32)
        C_np = _ensure_i32(C_np)
    else:
        V_np, C_np, v_backend = None, None, "numpy"
        if not use_json_mode:
            raise ValueError("Either provide vertices/cells arrays, or use JSON config with geometry (mesh files)")

    try:
        import polyfempy as pf
    except Exception:
        raise RuntimeError("polyfempy bindings not found. Please install/compile them first.")

    # Construct solver (support different versions)
    solver = None
    for ctor in ("Solver", "State"):
        if hasattr(pf, ctor):
            try:
                solver = getattr(pf, ctor)()
                break
            except Exception:
                pass
    if solver is None:
        raise RuntimeError("No usable Solver/State constructor found in polyfempy.")

    # Apply settings
    if use_json_mode:
        config_json = json.dumps(full_json)
        # Prefer explicit setter; some older variants used `settings(json, ...)`.
        if hasattr(solver, "set_settings"):
            solver.set_settings(config_json, strict_validation=False)
        elif hasattr(solver, "settings"):
            try:
                solver.settings(config_json, strict_validation=False)
            except TypeError as e:
                raise RuntimeError(
                    "Found solver.settings() but it does not accept arguments; "
                    "expected a settings setter or set_settings()."
                ) from e
        else:
            raise RuntimeError("Missing set_settings(...) on solver.")
        
        if hasattr(solver, "load_mesh_from_settings"):
            solver.load_mesh_from_settings()
        else:
            raise RuntimeError("JSON mode requires load_mesh_from_settings() method")
    else:
        # The C++ binding expects a JSON string (it does `json::parse(str(settings))`).
        # Do NOT pass Python dict repr or placeholder objects here.
        if hasattr(cfg, "to_dict"):
            settings_dict = cfg.to_dict()
        elif hasattr(cfg, "to_json_dict"):
            settings_dict = cfg.to_json_dict()
        else:
            raise TypeError("cfg must provide to_dict() or to_json_dict() for non-JSON mode.")

        settings_json = json.dumps(settings_dict)

        if hasattr(solver, "set_settings"):
            solver.set_settings(settings_json, strict_validation=False)
        elif hasattr(solver, "settings"):
            try:
                solver.settings(settings_json, strict_validation=False)
            except TypeError as e:
                raise RuntimeError(
                    "Found solver.settings() but it does not accept arguments; "
                    "expected a settings setter or set_settings()."
                ) from e
        else:
            raise RuntimeError("Missing set_settings(...) on solver.")

    # Set mesh (skip if JSON mode already loaded)
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

    # Build basis and assemble (required for JSON mode)
    if use_json_mode:
        if hasattr(solver, "build_basis"):
            solver.build_basis()
        if hasattr(solver, "assemble"):
            solver.assemble()
    
    # Side sets (optional)
    if sidesets_func is not None:
        name = _first_attr(solver, "set_sidesets_from_function", "build_sidesets_from_function")
        if name:
            getattr(solver, name)(sidesets_func)
    
    # Boundary conditions (skip if JSON mode - BCs in JSON)
    if not use_json_mode:
        bc_raw = getattr(cfg, "boundary_conditions", {}) or {}
        if hasattr(bc_raw, "to_dict"):
            bc = bc_raw.to_dict()
        else:
            bc = bc_raw if isinstance(bc_raw, dict) else {}

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

        rhs = bc.get("rhs")
        if rhs is not None:
            name = _first_attr(solver, "set_body_force", "set_rhs", "add_rhs")
            if name:
                getattr(solver, name)(rhs)

    # Run solve
    name = _first_attr(solver, "solve", "solve_problem", "run")
    if not name:
        raise RuntimeError("No solver entry point found (solve / solve_problem / run).")
    ret = getattr(solver, name)()

    # Fetch solution and additional fields
    fields = {}
    meta = {"solver_type": type(solver).__name__}

    # Preferred: parse return value from C++ binding (`solve() -> (sol, pressure)`).
    if isinstance(ret, (tuple, list)) and len(ret) >= 1:
        sol = np.asarray(ret[0])
        fields["u"] = sol
        if len(ret) >= 2 and ret[1] is not None:
            try:
                pressure = np.asarray(ret[1])
                # Keep consistent with `_extract_additional_fields` ("p" for pressure).
                fields["p"] = pressure
            except Exception:
                pass

        # Prefer solver-provided mesh if available (esp. JSON mode).
        V_pts = None
        for vn in ("get_vertices", "get_points"):
            if hasattr(solver, vn):
                try:
                    V_pts = np.asarray(getattr(solver, vn)())
                    break
                except Exception:
                    pass
        if V_pts is None:
            V_pts = V_np if V_np is not None else np.array([])

        cells_result = C_np
        for cn in ("get_elements", "get_cells"):
            if hasattr(solver, cn):
                try:
                    cells_result = np.asarray(getattr(solver, cn)())
                    break
                except Exception:
                    pass
        if cells_result is None:
            cells_result = np.array([], dtype=np.int32)

        _extract_additional_fields(solver, fields, meta)
        return Result(v_backend, V_pts, cells_result, fields, meta=meta)
    
    # Try sampled solution first
    if hasattr(solver, "get_sampled_solution"):
        out = solver.get_sampled_solution()
        if isinstance(out, (list, tuple)) and len(out) >= 5:
            pts = np.asarray(out[0])
            fun = np.asarray(out[4])
            fields["u"] = fun
            
            cells_result = C_np
            for cn in ("get_elements", "get_cells"):
                if hasattr(solver, cn):
                    try:
                        cells_result = np.asarray(getattr(solver, cn)())
                        break
                    except Exception:
                        pass
            
            _extract_additional_fields(solver, fields, meta)
            
            return Result(v_backend, pts, cells_result, fields, meta=meta)

    # Fallback to direct solution
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
    
    cells_result = C_np
    for cn in ("get_elements", "get_cells"):
        if hasattr(solver, cn):
            try:
                cells_result = np.asarray(getattr(solver, cn)())
                break
            except Exception:
                pass
    if cells_result is None:
        cells_result = np.array([], dtype=np.int32)
    
    _extract_additional_fields(solver, fields, meta)
    
    return Result(
        v_backend,
        V_pts,
        cells_result,
        fields,
        meta=meta,
    )


def _extract_additional_fields(solver, fields: dict, meta: dict):
    """Extract optional fields (stress, strain, energy, etc.)."""
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
