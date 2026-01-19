"""DEPRECATED backend shim (Route A only).

Why Route A:
- `polyfempy` is the only stable C++ extension module name in this repo.
- nanobind vs pybind11 is a build-time backend choice and must not affect Python import paths.

Important:
- The C++ binding's `Solver.solve()` returns `(sol, pressure)`, so Python must capture it.

This file is kept for backwards compatibility with the older backend abstraction.
The normal/high-level path should import and use `polyfempy` directly.
"""

import json
import numpy as np


def solve_impl(V, C, settings, callbacks):
    """Backend SPI entry point (Route A: call `polyfempy` C++ module)."""
    try:
        import polyfempy as pf
    except Exception as e:
        raise RuntimeError(
            "polyfempy C++ extension module is required. Build/install it first."
        ) from e

    # Normalize inputs
    V = np.ascontiguousarray(np.asarray(V, dtype=np.float64))
    C = np.ascontiguousarray(np.asarray(C, dtype=np.int32))

    # Settings must be JSON for the C++ binding (`json::parse(str(settings))`)
    settings_json = json.dumps(settings)

    solver = pf.Solver() if hasattr(pf, "Solver") else pf.State()
    if hasattr(solver, "set_settings"):
        solver.set_settings(settings_json, strict_validation=False)
    else:
        solver.settings(settings_json, strict_validation=False)

    # Mesh + solve
    solver.set_mesh(V, C)

    # Optional lifecycle callbacks (best-effort; C++ doesn't expose iteration hooks here)
    if callbacks and "before_solve" in callbacks:
        callbacks["before_solve"](meta={})

    ret = solver.solve()

    if callbacks and "after_solve" in callbacks:
        callbacks["after_solve"](meta={})

    sol = None
    pressure = None
    if isinstance(ret, (tuple, list)) and len(ret) > 0:
        sol = ret[0]
        if len(ret) > 1:
            pressure = ret[1]

    out = {
        "u": None if sol is None else np.ascontiguousarray(np.asarray(sol)),
        "strain": None,
        "stress": None,
        "meta": {
            "backend": "polyfempy",
            "iters": None,
            "residual": None,
            "seed": None,
        },
    }
    if pressure is not None:
        out["p"] = np.ascontiguousarray(np.asarray(pressure))
    return out

