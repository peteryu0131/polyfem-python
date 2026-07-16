"""Smoke test for the current builder-style generated API example."""

from __future__ import annotations

import importlib.util
import py_compile
from pathlib import Path

from polyfempy.api._solve_contract import prepare_canonical_solve_input


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PATH = (
    ROOT
    / "examples"
    / "classic_example"
    / "2D"
    / "new_better_contact_2d_golf_ball_deformable_wall_generated_api.py"
)


def _import_example():
    spec = importlib.util.spec_from_file_location("new_better_contact_2d_example", EXAMPLE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_current_builder_example_compiles():
    py_compile.compile(str(EXAMPLE_PATH), doraise=True)


def test_current_builder_example_builds_generated_payload():
    example = _import_example()
    payload = example.polyfem_config.as_dict()

    assert len(payload["geometry"]) == 2
    assert len(payload["materials"]) == 2
    assert payload["time"] == {"tend": 0.004, "dt": 2e-05}
    assert payload["contact"]["enabled"] is True
    assert "dirichlet_boundary" in payload["boundary_conditions"]
    assert "output" in payload
    assert "solver" in payload


def test_current_builder_example_uses_generated_solve_path():
    example = _import_example()
    canonical = prepare_canonical_solve_input(
        vertices=None,
        cells=None,
        cfg=example.polyfem_config,
        dtype=None,
    )

    assert canonical.metadata == {"mesh_source": "json", "config_source": "generated"}
    assert canonical.mesh_source.mode == "json"
    assert "geometry" in canonical.backend_settings
    assert "materials" in canonical.backend_settings
