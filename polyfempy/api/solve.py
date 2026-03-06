"""Solver entry point. Uses polyfempy C++ backend; solve() returns (sol, pressure)."""

import numpy as np
import json
from pathlib import Path
from . import tensor as T
from .result import Result


def _process_json_config(full_json, cfg):
    """Normalize JSON config for the solver. No separate common config file is used."""
    import copy
    processed = copy.deepcopy(full_json)
    # Drop "common" key if present; solver uses only this config (no external common.json).
    processed.pop("common", None)
    return processed


def _strip_solver_defaults(processed):
    """Remove default solver config blocks that cause C++ validation errors."""
    unwanted = ["ADAM", "L-BFGS", "L-BFGS-B", "Newton", "StochasticADAM", "StochasticGradientDescent"]
    if "solver" not in processed or not isinstance(processed["solver"], dict):
        return
    solver_dict = processed["solver"]
    if "nonlinear" not in solver_dict or not isinstance(solver_dict["nonlinear"], dict):
        return
    nonlinear = solver_dict["nonlinear"]
    for k in unwanted:
        nonlinear.pop(k, None)
    # line_search: keep only method + common fields
    if "line_search" in nonlinear and isinstance(nonlinear["line_search"], dict):
        ls = nonlinear["line_search"]
        common_keys = ["default_init_step_size", "max_step_size_iter", "max_step_size_iter_final",
                      "min_step_size", "min_step_size_final", "step_ratio", "use_grad_norm_tol"]
        if "method" in ls:
            nonlinear["line_search"] = {"method": ls["method"], **{k: ls[k] for k in common_keys if k in ls}}
        else:
            nonlinear["line_search"] = {k: ls[k] for k in common_keys if k in ls}


def _clean_json_for_cpp(obj, path=""):
    """Remove nulls and solver default blocks for C++ JSON schema."""
    if isinstance(obj, dict):
        cleaned = {}
        for key, value in obj.items():
            current_path = f"{path}.{key}" if path else key
            if key in ["ADAM", "L-BFGS", "L-BFGS-B", "Newton", "StochasticADAM", "StochasticGradientDescent"]:
                continue
            if key == "line_search" and isinstance(value, dict):
                cleaned_line_search = {}
                if "method" in value:
                    cleaned_line_search["method"] = value["method"]
                    for common_key in ["default_init_step_size", "max_step_size_iter", "max_step_size_iter_final",
                                     "min_step_size", "min_step_size_final", "step_ratio", "use_grad_norm_tol"]:
                        if common_key in value:
                            cleaned_line_search[common_key] = value[common_key]
                else:
                    cleaned_line_search = {}
                    for common_key in ["default_init_step_size", "max_step_size_iter", "max_step_size_iter_final",
                                     "min_step_size", "min_step_size_final", "step_ratio", "use_grad_norm_tol"]:
                        if common_key in value:
                            cleaned_line_search[common_key] = value[common_key]
                cleaned_value = cleaned_line_search
            else:
                cleaned_value = _clean_json_for_cpp(value, current_path)
            if cleaned_value is not None:
                cleaned[key] = cleaned_value
        return cleaned
    elif isinstance(obj, list):
        return [_clean_json_for_cpp(item, f"{path}[{i}]") for i, item in enumerate(obj) if _clean_json_for_cpp(item, f"{path}[{i}]") is not None]
    elif obj is None:
        return None
    else:
        return obj


def _first_attr(obj, *names):
    for n in names:
        if hasattr(obj, n):
            return n
    return None


def _ensure_i32(cells):
    return cells.astype(np.int32, copy=False) if cells.dtype != np.int32 else cells


def solve(vertices=None, cells=None, cfg=None, sidesets_func=None, dtype=None):
    """Run PolyFEM solve. Provide vertices/cells or cfg with geometry."""
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
    else:
        # Fallback: if cfg has geometry and user didn't provide vertices/cells,
        # synthesize a JSON config from cfg.to_dict() and normalize it.
        try:
            cfg_dict = cfg.to_dict()
        except Exception:
            cfg_dict = None
        if isinstance(cfg_dict, dict) and "geometry" in cfg_dict:
            full_json = cfg_dict
            # JSON schema expects materials to be an array
            mats = full_json.get("materials")
            if isinstance(mats, dict):
                full_json["materials"] = [mats]
            # Promote root_path from extras if present (for resolving relative paths)
            if hasattr(cfg, "extras") and cfg.extras and "root_path" not in full_json and "_root_path" in cfg.extras:
                full_json["root_path"] = cfg.extras["_root_path"]
    
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

    # Construct solver (must be C++ backend, not Python config.Solver)
    # pf.Solver can be either: (1) C++ State binding with set_settings, or
    # (2) Python config.Solver from api (when C++ extension failed to load)
    solver = None
    if getattr(pf, "cpp_backend_available", lambda: False)():
        # C++ backend loaded: get solver from C++ module directly to avoid name collision
        import importlib
        _core = importlib.import_module("polyfempy.polyfempy")
        for ctor in ("Solver", "State"):
            if hasattr(_core, ctor):
                try:
                    solver = getattr(_core, ctor)()
                    break
                except Exception:
                    pass
    if solver is None:
        # Fallback: try pf.Solver/pf.State (may be Python config if C++ not loaded)
        for ctor in ("Solver", "State"):
            if hasattr(pf, ctor):
                try:
                    candidate = getattr(pf, ctor)()
                    if hasattr(candidate, "set_settings") or hasattr(candidate, "settings"):
                        solver = candidate
                        break
                except Exception:
                    pass
    if solver is None:
        if not getattr(pf, "cpp_backend_available", lambda: False)():
            err = getattr(pf, "cpp_backend_error", lambda: None)()
            raise RuntimeError(
                "C++ backend not loaded. JSON/array mode requires the compiled extension. "
                f"Error: {err}. Build with: pip install -e . --no-build-isolation"
            )
        raise RuntimeError("No usable Solver/State constructor found in polyfempy.")

    if use_json_mode:
        processed_json = _process_json_config(full_json, cfg)
        processed_json.pop("common", None)
        processed_json = _clean_json_for_cpp(processed_json)
        _strip_solver_defaults(processed_json)
        config_json = json.dumps(processed_json)
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
        if hasattr(cfg, "to_dict"):
            settings_dict = cfg.to_dict()
        elif hasattr(cfg, "to_json_dict"):
            settings_dict = cfg.to_json_dict()
        else:
            raise TypeError("cfg must provide to_dict() or to_json_dict() for non-JSON mode.")

        bc_raw = getattr(cfg, "boundary_conditions", {}) or {}
        if hasattr(bc_raw, "to_dict"):
            bc = bc_raw.to_dict()
        else:
            bc = bc_raw if isinstance(bc_raw, dict) else {}

        if "geometry" not in settings_dict:
            settings_dict["geometry"] = [{
                "type": "ground",
                "height": 0.0,
                "enabled": True,
                "is_obstacle": False
            }]
        
        if "materials" in settings_dict:
            materials = settings_dict["materials"]
            if isinstance(materials, dict) and not isinstance(materials, list):
                if "type" not in materials:
                    pde = settings_dict.get("pde", "LinearElasticity")
                    if pde == "Poisson":
                        materials["type"] = "Laplacian"
                    else:
                        materials["type"] = "LinearElasticity"
                settings_dict["materials"] = [materials]
        if bc:
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

    if not use_json_mode:
        set_mesh_ok = False
        for name in ("set_mesh", "set_mesh_data", "load_mesh_from_points"):
            if hasattr(solver, name):
                fn = getattr(solver, name)
                try:
                    fn(V_np, C_np)
                    set_mesh_ok = True
                    break
                except TypeError:
                    try:
                        fn(points=V_np, cells=C_np)
                        set_mesh_ok = True
                        break
                    except Exception:
                        pass
        if not set_mesh_ok:
            raise RuntimeError("No mesh setter found (set_mesh / set_mesh_data / load_mesh_from_points).")
        try:
            current_settings = solver.settings()
            if isinstance(current_settings, dict) and bc:
                if "boundary_conditions" not in current_settings:
                    current_settings["boundary_conditions"] = {}
                if not isinstance(current_settings["boundary_conditions"], dict):
                    current_settings["boundary_conditions"] = {}
                current_settings["boundary_conditions"].update(bc)
                settings_json = json.dumps(current_settings)
                solver.set_settings(settings_json, strict_validation=False)
        except Exception:
            try:
                if bc:
                    if "boundary_conditions" not in settings_dict:
                        settings_dict["boundary_conditions"] = {}
                    settings_dict["boundary_conditions"].update(bc)
                    settings_json = json.dumps(settings_dict)
                    solver.set_settings(settings_json, strict_validation=False)
            except Exception:
                pass

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
        if not use_json_mode and bc:
            try:
                if bc:
                    if "boundary_conditions" not in settings_dict:
                        settings_dict["boundary_conditions"] = {}
                    settings_dict["boundary_conditions"].update(bc)
                settings_json = json.dumps(settings_dict)
                solver.set_settings(settings_json, strict_validation=False)
            except Exception:
                pass

    if hasattr(solver, "build_basis"):
        solver.build_basis()
    if hasattr(solver, "assemble"):
        solver.assemble()

    name = _first_attr(solver, "solve", "run")
    if not name:
        raise RuntimeError("No solver entry point found (solve / run).")
    
    log_level = 2
    if full_json and "output" in full_json and "log" in full_json["output"]:
        log_level_str = full_json["output"]["log"].get("level", "info")
        log_level_map = {"trace": 0, "debug": 1, "info": 2, "warn": 3, "warning": 3, 
                         "error": 4, "critical": 5, "off": 6}
        if log_level_str in log_level_map:
            log_level = log_level_map[log_level_str]
    
    try:
        ret = getattr(solver, name)(log_level=log_level)
    except TypeError:
        ret = getattr(solver, name)()

    fields = {}
    meta = {"solver_type": type(solver).__name__}

    if isinstance(ret, (tuple, list)) and len(ret) >= 1:
        sol = np.asarray(ret[0])
        fields["u"] = sol
        if len(ret) >= 2 and ret[1] is not None:
            try:
                pressure = np.asarray(ret[1])
                fields["p"] = pressure
            except Exception:
                pass

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

    u = None
    for name in ("get_solution", "get_displacement", "get_u"):
        if hasattr(solver, name):
            u = np.asarray(getattr(solver, name)())
            fields["u"] = u
            break
    
    if u is None:
        raise RuntimeError("Failed to retrieve solution: no known getters (sampled or direct).")
    
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
    for name in ("get_pressure", "pressure"):
        if hasattr(solver, name):
            try:
                pressure = getattr(solver, name)()
                if pressure is not None:
                    fields["p"] = np.asarray(pressure)
                    break
            except Exception:
                pass
    for name in ("get_velocity", "velocity"):
        if hasattr(solver, name):
            try:
                velocity = getattr(solver, name)()
                if velocity is not None:
                    fields["v"] = np.asarray(velocity)
                    break
            except Exception:
                pass
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
