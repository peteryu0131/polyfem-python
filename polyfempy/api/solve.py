"""
Route A only:
- Always `import polyfempy as pf` (stable C++ extension module name).

Important:
- The C++ binding's `Solver.solve()` returns a tuple `(sol, pressure)`, so Python must
  capture and parse the return value instead of assuming getters exist.
"""

import numpy as np
import json
from pathlib import Path
from . import tensor as T
from .result import Result


def _merge_json_dicts(base, override):
    """Recursively merge two JSON dictionaries.
    
    Values from override take precedence over base.
    """
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_json_dicts(result[key], value)
        else:
            result[key] = value
    return result


def _process_json_config(full_json, cfg):
    """Process JSON configuration: handle common.json references and merge configs.
    
    Args:
        full_json: Full JSON configuration dict
        cfg: SimulationConfig object (may contain root_path info)
    
    Returns:
        Processed JSON configuration with common.json merged
    """
    import copy
    processed = copy.deepcopy(full_json)
    
    # Check for common.json reference
    if "common" in processed:
        common_path = processed.pop("common")
        
        # Get root_path (JSON file location) to resolve relative paths
        root_path = None
        if hasattr(cfg, "extras") and cfg.extras and "_root_path" in cfg.extras:
            root_path = Path(cfg.extras["_root_path"]).parent
        elif "root_path" in processed:
            root_path = Path(processed["root_path"]).parent
        
        # Resolve common.json path relative to JSON file location
        if root_path is not None:
            # common_path is relative to JSON file (e.g., "../../common.json")
            # root_path is the directory containing the JSON file
            common_file = (root_path / common_path).resolve()
        else:
            # Fallback: try relative to current working directory
            common_file = Path(common_path)
            if not common_file.is_absolute():
                common_file = Path.cwd() / common_file
        
        # Try to load common.json
        common_loaded = False
        if common_file.exists() and common_file.is_file():
            try:
                with open(common_file, "r") as f:
                    common_config = json.load(f)
                # Merge: common.json as base, main config as override
                processed = _merge_json_dicts(common_config, processed)
                common_loaded = True
            except Exception as e:
                import warnings
                warnings.warn(f"Failed to load common.json from '{common_file}': {e}", RuntimeWarning)
        
        if not common_loaded:
            # Try fallback locations
            possible_paths = [
                Path(common_path),  # Direct path
                root_path / common_path if root_path else None,  # Relative to JSON file
            ]
            # Filter out None values
            possible_paths = [p for p in possible_paths if p is not None]
            
            for path in possible_paths:
                if path.exists() and path.is_file():
                    try:
                        with open(path, "r") as f:
                            common_config = json.load(f)
                        processed = _merge_json_dicts(common_config, processed)
                        common_loaded = True
                        break
                    except Exception:
                        continue
        
        if not common_loaded:
            # Warn but continue - C++ backend might handle it
            import warnings
            warnings.warn(
                f"Could not load common.json from '{common_path}' (resolved from {root_path}). "
                f"C++ backend will try to resolve it, but this may cause errors.",
                RuntimeWarning
            )
    
    # Optimize output settings to reduce unnecessary VTM files
    # VTM files are only useful when multiple datasets are exported (volume + surface + contact, etc.)
    # If only one dataset type is enabled, VTM files are unnecessary
    if "output" in processed and "paraview" in processed["output"]:
        paraview_config = processed["output"]["paraview"]
        
        # Count how many dataset types are enabled
        dataset_count = 0
        if paraview_config.get("volume", True):  # Default is True
            dataset_count += 1
        if paraview_config.get("surface", False):
            dataset_count += 1
        if paraview_config.get("wireframe", False):
            dataset_count += 1
        if paraview_config.get("points", False):
            dataset_count += 1
        
        # Check for contact-related outputs
        options = paraview_config.get("options", {})
        if options.get("contact_forces", False) or options.get("friction_forces", False):
            dataset_count += 1
        
        # Note: We cannot directly disable VTM generation from Python API
        # as it's hardcoded in C++ save_vtu() function. However, we can optimize
        # the output configuration to minimize unnecessary datasets.
        # The C++ code should be modified to only generate VTM when dataset_count > 1
        # For now, we'll leave a note in the config for future reference
        if dataset_count <= 1:
            # Only one dataset type - VTM file is unnecessary but will still be generated by C++
            # This is a known issue in the C++ backend that should be fixed
            pass
    
    return processed


def _clean_json_for_cpp(obj):
    """Recursively clean JSON object to remove null values and ensure C++ compatibility.
    
    C++ JSON parser may not accept null values in places where strings are expected.
    """
    if isinstance(obj, dict):
        cleaned = {}
        for key, value in obj.items():
            cleaned_value = _clean_json_for_cpp(value)
            # Skip null values (but keep empty strings and other falsy values)
            if cleaned_value is not None:
                cleaned[key] = cleaned_value
        return cleaned
    elif isinstance(obj, list):
        return [_clean_json_for_cpp(item) for item in obj if _clean_json_for_cpp(item) is not None]
    elif obj is None:
        # Replace None with empty string for string fields, or remove from dict
        return None  # Will be filtered out in dict/list processing
    else:
        return obj


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
        # Process common.json references and merge configurations
        processed_json = _process_json_config(full_json, cfg)
        
        # Remove null values and ensure all strings are non-null
        processed_json = _clean_json_for_cpp(processed_json)
        
        config_json = json.dumps(processed_json)
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

        # Extract boundary conditions early (needed later)
        bc_raw = getattr(cfg, "boundary_conditions", {}) or {}
        if hasattr(bc_raw, "to_dict"):
            bc = bc_raw.to_dict()
        else:
            bc = bc_raw if isinstance(bc_raw, dict) else {}

        # C++ JSON schema requires "geometry" field, but to_dict() doesn't include it
        # We'll add a placeholder that will be replaced by set_mesh()
        if "geometry" not in settings_dict:
            settings_dict["geometry"] = [{
                "type": "ground",
                "height": 0.0,
                "enabled": True,
                "is_obstacle": False
            }]
        
        # C++ JSON schema requires "materials" to be an array with "type" field
        # Convert materials dict to array format if needed
        if "materials" in settings_dict:
            materials = settings_dict["materials"]
            if isinstance(materials, dict) and not isinstance(materials, list):
                # Ensure materials has "type" field
                if "type" not in materials:
                    # Infer type from pde if available
                    pde = settings_dict.get("pde", "LinearElasticity")
                    if pde == "Poisson":
                        materials["type"] = "Laplacian"
                    else:
                        materials["type"] = "LinearElasticity"
                # Convert to array format
                settings_dict["materials"] = [materials]
        
        # Add boundary conditions to JSON config (C++ backend expects BCs in JSON)
        if bc:
            # Merge boundary conditions into settings_dict
            if "boundary_conditions" not in settings_dict:
                settings_dict["boundary_conditions"] = {}
            settings_dict["boundary_conditions"].update(bc)

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
        
        # After set_mesh(), re-apply settings to ensure problem is re-initialized
        # with boundary conditions. set_mesh() calls load_mesh() which may reset
        # the problem state, so we need to re-initialize it with the full config
        # including boundary conditions.
        if not use_json_mode:
            # Re-apply the full settings (including boundary conditions) to ensure
            # the problem is properly initialized before build_basis()
            try:
                current_settings = solver.settings()
                if isinstance(current_settings, dict):
                    # Ensure boundary conditions are present
                    if bc:
                        if "boundary_conditions" not in current_settings:
                            current_settings["boundary_conditions"] = {}
                        if not isinstance(current_settings["boundary_conditions"], dict):
                            current_settings["boundary_conditions"] = {}
                        current_settings["boundary_conditions"].update(bc)
                    # Re-apply settings to re-initialize the problem
                    settings_json = json.dumps(current_settings)
                    solver.set_settings(settings_json, strict_validation=False)
            except Exception:
                # Fallback: re-apply the original settings_dict with boundary conditions
                try:
                    if bc:
                        if "boundary_conditions" not in settings_dict:
                            settings_dict["boundary_conditions"] = {}
                        settings_dict["boundary_conditions"].update(bc)
                    settings_json = json.dumps(settings_dict)
                    solver.set_settings(settings_json, strict_validation=False)
                except Exception:
                    pass
    
    # Side sets (optional) - must be set BEFORE build_basis
    # Boundary IDs need to be assigned before basis functions are built
    if sidesets_func is not None:
        # Note: mesh.compute_boundary_ids() is not exposed in the C++ bindings
        # Instead, we manually compute boundary IDs and use set_boundary_ids()
        try:
            mesh = solver.mesh()
            if hasattr(mesh, "set_boundary_ids") and hasattr(mesh, "n_boundary_elements"):
                # Get number of boundary elements
                n_boundary = mesh.n_boundary_elements()
                boundary_ids = []
                
                # Compute boundary ID for each boundary element
                # Note: In 2D, some "boundary elements" might be internal edges (diagonals)
                # where both vertices are on the boundary but the edge itself is not.
                # We still assign IDs to all boundary elements, but the user's sidesets_func
                # should handle this correctly based on edge center coordinates.
                for i in range(n_boundary):
                    try:
                        # Get boundary element vertices
                        v0 = mesh.boundary_element_vertex(i, 0)
                        v1 = mesh.boundary_element_vertex(i, 1)
                        
                        # Get vertex coordinates
                        p0 = mesh.point(v0)
                        p1 = mesh.point(v1)
                        
                        # Calculate edge center
                        center = (p0 + p1) / 2.0
                        
                        # Call sidesets_func to get boundary ID based on edge center
                        # The function should handle internal edges correctly (return -1 or ignore)
                        bid = sidesets_func(center, True)
                        boundary_ids.append(bid)
                    except Exception as e:
                        # Default to invalid ID if computation fails
                        boundary_ids.append(-1)
                
                # Set boundary IDs (only if we have valid IDs)
                if boundary_ids:
                    boundary_ids_array = np.array(boundary_ids, dtype=np.int32)
                    mesh.set_boundary_ids(boundary_ids_array)
        except Exception as e:
            # If setting boundary IDs fails, log but continue
            # The boundary conditions in JSON might still work if IDs are auto-assigned
            import warnings
            warnings.warn(f"Failed to set boundary IDs: {e}", RuntimeWarning)
        
        # Re-apply settings to ensure boundary conditions are recognized
        # Note: solver.settings() returns C++ JSON type, so we use the original settings_dict
        if not use_json_mode and bc:
            try:
                # Re-apply original settings_dict with boundary conditions
                if bc:
                    if "boundary_conditions" not in settings_dict:
                        settings_dict["boundary_conditions"] = {}
                    settings_dict["boundary_conditions"].update(bc)
                settings_json = json.dumps(settings_dict)
                solver.set_settings(settings_json, strict_validation=False)
            except Exception:
                pass

    # Build basis BEFORE calling solve() to ensure boundary conditions are recognized
    # This is critical: boundary conditions must be set up before build_basis() is called
    # Note: solver.solve() internally calls build_basis() and assemble() again,
    # but we need to build basis first to ensure boundary conditions are properly recognized.
    if hasattr(solver, "build_basis"):
        solver.build_basis()
    if hasattr(solver, "assemble"):
        solver.assemble()

    # Run solve
    # Note: solver.solve() will call build_basis() and assemble() again internally.
    # This is intentional - it ensures the problem state is properly initialized.
    # The duplicate calls should be safe as PolyFEM checks if basis is already built.
    name = _first_attr(solver, "solve", "run")
    if not name:
        raise RuntimeError("No solver entry point found (solve / run).")
    
    # The C++ solve() method signature is: solve(log_level=3)
    # spdlog level mapping: 0=trace, 1=debug, 2=info, 3=warn, 4=error, 5=critical, 6=off
    # We use log_level=2 (info) for less verbose output, or respect JSON config if available
    # Check if JSON config has log level set, otherwise use info (2)
    log_level = 2  # Default to info (2) for less verbose output
    if full_json and "output" in full_json and "log" in full_json["output"]:
        log_level_str = full_json["output"]["log"].get("level", "info")
        # Map string to spdlog level enum
        log_level_map = {"trace": 0, "debug": 1, "info": 2, "warn": 3, "warning": 3, 
                         "error": 4, "critical": 5, "off": 6}
        if log_level_str in log_level_map:
            log_level = log_level_map[log_level_str]
    
    try:
        ret = getattr(solver, name)(log_level=log_level)
    except TypeError:
        # Fallback: call without arguments (if signature doesn't match)
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
