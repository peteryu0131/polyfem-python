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


def solve(vertices, cells, cfg, sidesets_func=None, dtype=None):
    """Unified high-level solve entry point.

    This adapter normalizes user arrays (NumPy / Torch / JAX) to a CPU,
    C-contiguous NumPy representation, constructs a backend Settings/Problem
    from `SimulationConfig`, adapts to different polyfempy versions via
    light reflection, applies minimal boundary data, runs the solver, and
    wraps outputs into a `Result`.

    Args:
        vertices: Point coordinates of shape (N, dim). Can be NumPy / Torch / JAX.
        cells: Cell connectivity of shape (M, k). Can be NumPy / Torch / JAX.
        cfg: A `SimulationConfig` instance (see `config.py`).
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
    # 1) Normalize inputs to NumPy while remembering original backend.
    V_np, v_backend = T.as_numpy(vertices, dtype=dtype)
    C_np, _ = T.as_numpy(cells, dtype=np.int32)
    C_np = _ensure_i32(C_np)

    # 2) Build Settings from config (early-exit if Dummy).
    settings = cfg.to_settings()
    if getattr(settings, "_is_dummy", False):
        raise RuntimeError("DummySettings: C++ bindings not found; cannot run a real solve.")

    # 3) Import polyfempy (required for real execution).
    try:
        import polyfempy as pf
    except Exception:
        raise RuntimeError("polyfempy bindings not found. Please install/compile them first.")

    # 4) Construct solver/state (support different versions).
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

    # 5) Apply Settings (`settings(...)` / `set_settings(...)`).
    name = _first_attr(solver, "settings", "set_settings")
    if not name:
        raise RuntimeError("Missing settings(...) / set_settings(...) on solver.")
    getattr(solver, name)(settings)

    # 6) Set mesh (`set_mesh` / `set_mesh_data` / `load_mesh_from_points`).
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

    # 7) Side sets (optional).
    if sidesets_func is not None:
        name = _first_attr(solver, "set_sidesets_from_function", "build_sidesets_from_function")
        if name:
            getattr(solver, name)(sidesets_func)

    # 8) Minimal boundary conditions (dirichlet / rhs).
    bc = getattr(cfg, "boundary_conditions", {}) or {}

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

    # 10) Fetch solution (prefer sampled; otherwise direct).
    # 10.1 Sampled solution: typically returns (pts, tris, el_id, bid, fun).
    if hasattr(solver, "get_sampled_solution"):
        out = solver.get_sampled_solution()
        if isinstance(out, (list, tuple)) and len(out) >= 5:
            pts = np.asarray(out[0])
            fun = np.asarray(out[4])
            return Result(
                v_backend,
                pts,
                C_np,
                {"u": fun},
                meta={"solver_type": type(solver).__name__},
            )

    # 10.2 Direct solution: `get_solution` / `get_displacement` / `get_u`.
    for name in ("get_solution", "get_displacement", "get_u"):
        if hasattr(solver, name):
            u = np.asarray(getattr(solver, name)())
            # Prefer solver-provided vertices if available.
            V_pts = None
            for vn in ("get_vertices", "get_points"):
                if hasattr(solver, vn):
                    V_pts = np.asarray(getattr(solver, vn)())
                    break
            if V_pts is None:
                V_pts = V_np
            return Result(
                v_backend,
                V_pts,
                C_np,
                {"u": u},
                meta={"solver_type": type(solver).__name__},
            )

    # Unknown getter set.
    raise RuntimeError("Failed to retrieve solution: no known getters (sampled or direct).")
