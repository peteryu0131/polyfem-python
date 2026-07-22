"""Tests for the minimal runtime ``Result`` container."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from polyfempy.runtime.result import Result  # noqa: E402


def _mesh():
    vertices = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        dtype=np.float64,
    )
    cells = np.array([[0, 1, 2]], dtype=np.int32)
    return vertices, cells


def test_result_stores_backend_mesh_solution_pressure_and_meta():
    vertices, cells = _mesh()
    u = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]])
    p = np.array([0.5, 0.5, 0.5])

    result = Result(vertices, cells, u, p=p, meta={"from_bundle": True})

    np.testing.assert_array_equal(result.vertices, vertices)
    np.testing.assert_array_equal(result.cells, cells)
    np.testing.assert_array_equal(result.u, u)
    np.testing.assert_array_equal(result.p, p)
    assert result.meta == {"from_bundle": True}


def test_result_reshapes_flat_solution_dofs():
    vertices, cells = _mesh()
    result = Result(vertices, cells, np.arange(6))

    np.testing.assert_array_equal(result.u, np.arange(6).reshape(3, 2))


def test_result_keeps_missing_pressure_as_none():
    vertices, cells = _mesh()
    result = Result(vertices, cells, np.zeros((3, 2)))

    assert result.p is None


def test_removed_runtime_output_and_convenience_helpers_are_absent():
    vertices, cells = _mesh()
    result = Result(vertices, cells, np.zeros((3, 2)))

    for name in (
        "backend",
        "V",
        "fields",
        "point_data",
        "cell_data",
        "history",
        "sampled_data",
        "body_ids",
        "von_mises",
        "stress",
        "strain",
        "has_field",
        "require_field",
        "point_field",
        "cell_field",
        "set_field",
        "remove_field",
        "field_names",
        "field",
        "n_vertices",
        "n_cells",
        "available_fields",
        "summary",
        "as_numpy",
        "to_backend",
        "to_torch",
        "set_sampled_field",
        "sampled_field",
        "field_by_body",
        "to_meshio",
        "from_meshio",
        "read",
        "write",
        "to_vtk",
        "get_percentile_from_von_mises",
    ):
        assert not hasattr(result, name)
        assert not hasattr(Result, name)
