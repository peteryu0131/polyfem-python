from pathlib import Path
import json

import numpy as np
import pytest


def test_backend_forward_solve_smoke(tmp_path):
    import polyfempy as pf
    from polyfempy.runtime import solve

    if not pf.cpp_backend_available():
        pytest.skip(f"C++ backend is unavailable: {pf.cpp_backend_error()}")

    # The test file may run from an installed editable checkout or from the
    # repository root. Use this file location so the example config and meshes
    # are found independently of the current working directory.
    config_path = Path(__file__).resolve().parents[1] / "examples" / "configs" / "contact_impact.json"
    if not config_path.exists():
        pytest.skip(f"backend smoke config is missing: {config_path}")

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    cfg["root_path"] = str(config_path)
    cfg.setdefault("time", {})
    cfg["time"]["tend"] = 0.01
    cfg["time"]["dt"] = 0.01

    output_dir = tmp_path / "backend_smoke"
    cfg["output"] = {
        "directory": str(output_dir),
        "json": "smoke_stats.json",
        "paraview": {"file_name": ""},
        "advanced": {"save_time_sequence": False},
        "result": {"fields": ["u"], "strict": True},
        "fallback": {"sampled_vtu": "never"},
    }

    result = solve(cfg=cfg)
    try:
        vertices = np.asarray(result.vertices)
        u = np.asarray(result.u)

        assert vertices.ndim == 2
        assert vertices.shape[0] > 0
        assert u.size > 0
    finally:
        if hasattr(result, "release_solver"):
            result.release_solver()
