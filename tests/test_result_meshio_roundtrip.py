"""Round-trip tests for `Result.from_meshio` / `Result.to_meshio`.

Pinpointed at the multi cell-block data-loss bug:

    * Before T3, ``from_meshio`` kept only ``arr_list[0]`` of each field,
      dropping every subsequent block's cell_data silently.
    * Before T3, ``to_meshio`` only emitted cell_data when the mesh had a
      single block, so a multi-block mesh round-trip dropped cell_data on the
      write side too.

These tests do **not** require ``meshio`` to be installed. A minimal stand-in
with just the attributes ``Result.from_meshio`` / ``to_meshio`` touches
(``points``, ``cells``, ``point_data``, ``cell_data`` for input; a ``Mesh``
class for output) is injected through ``sys.modules``.
"""

from __future__ import annotations

import sys
import types
import unittest
import unittest.mock
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from polyfempy.api.result import Result  # noqa: E402


# ---------------------------------------------------------------------------
# Minimal stand-ins for meshio so the tests don't need the real package.
# ---------------------------------------------------------------------------


class _FakeCellBlock:
    """Mirror of the modern meshio.CellBlock with .type / .data attributes."""

    def __init__(self, cell_type: str, data: np.ndarray):
        self.type = cell_type
        self.data = data


class _FakeInputMesh:
    """Stand-in for a meshio Mesh passed *into* Result.from_meshio()."""

    def __init__(self, points, cells, point_data=None, cell_data=None):
        self.points = np.asarray(points)
        self.cells = cells
        self.point_data = dict(point_data) if point_data else {}
        self.cell_data = dict(cell_data) if cell_data else {}


class _RecordingOutputMesh:
    """Stand-in for the Mesh produced by Result.to_meshio().

    We just record the kwargs and expose the same shape meshio uses so tests
    can introspect them.
    """

    def __init__(self, vertices, cells, point_data=None, cell_data=None):
        self.points = vertices  # meshio names the arg ``points`` positionally
        self.cells = cells
        self.point_data = point_data or {}
        self.cell_data = cell_data or {}


def _install_fake_meshio():
    """Return a context manager that swaps ``meshio`` in sys.modules."""
    fake = types.SimpleNamespace(Mesh=_RecordingOutputMesh)
    return unittest.mock.patch.dict(sys.modules, {"meshio": fake})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _two_block_mesh_inputs():
    """A minimal mixed triangle+quad 2D mesh.

    triangles: 3 cells over vertices 0..3
    quads:     2 cells over vertices 0..5
    total n_cells = 5
    """
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [2.0, 1.0],
        ],
        dtype=np.float64,
    )
    tri = np.array([[0, 1, 2], [1, 3, 2], [1, 4, 3]], dtype=np.int32)
    quad = np.array([[1, 4, 5, 3], [0, 1, 3, 2]], dtype=np.int32)
    return points, tri, quad


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class FromMeshioMultiBlockTests(unittest.TestCase):
    def test_single_block_cell_data_unchanged(self):
        """Sanity: the common single-block case still produces the same flat
        array as before, even after the multi-block fix landed."""
        points = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64)
        tri = np.array([[0, 1, 2]], dtype=np.int32)
        fake = _FakeInputMesh(
            points=points,
            cells=[_FakeCellBlock("triangle", tri)],
            cell_data={"mat_id": [np.array([7])]},
        )
        r = Result.from_meshio(fake)
        self.assertIn("mat_id", r._cell_data)
        np.testing.assert_array_equal(r._cell_data["mat_id"], np.array([7]))

    def test_multi_block_cell_data_is_concatenated(self):
        """This is the actual bug fix: per-block arrays must all be preserved,
        concatenated in block order, instead of dropping blocks 1..N silently.
        """
        points, tri, quad = _two_block_mesh_inputs()
        tri_mat = np.array([10, 11, 12])
        quad_mat = np.array([20, 21])
        fake = _FakeInputMesh(
            points=points,
            cells=[_FakeCellBlock("triangle", tri), _FakeCellBlock("quad", quad)],
            cell_data={"mat_id": [tri_mat, quad_mat]},
        )
        r = Result.from_meshio(fake)
        np.testing.assert_array_equal(
            r._cell_data["mat_id"], np.array([10, 11, 12, 20, 21])
        )

    def test_multi_block_preserves_per_block_order(self):
        """The concatenation must follow cell-block order, not arr_list order
        rearranged by any other key (there is no dict-ordering risk here since
        arr_list is a list, but we lock the invariant anyway)."""
        points, tri, quad = _two_block_mesh_inputs()
        fake = _FakeInputMesh(
            points=points,
            cells=[_FakeCellBlock("triangle", tri), _FakeCellBlock("quad", quad)],
            cell_data={
                "stress_xx": [np.array([1.0, 2.0, 3.0]), np.array([100.0, 200.0])],
            },
        )
        r = Result.from_meshio(fake)
        np.testing.assert_array_equal(
            r._cell_data["stress_xx"],
            np.array([1.0, 2.0, 3.0, 100.0, 200.0]),
        )

    def test_heterogeneous_block_shapes_fall_back_to_first_block(self):
        """If per-block arrays have incompatible non-cell axes, concat fails
        and we preserve legacy behavior (keep block 0 only) rather than crash.
        """
        points, tri, quad = _two_block_mesh_inputs()
        tri_data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # (3, 2)
        quad_data = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])  # (2, 3)
        fake = _FakeInputMesh(
            points=points,
            cells=[_FakeCellBlock("triangle", tri), _FakeCellBlock("quad", quad)],
            cell_data={"weird": [tri_data, quad_data]},
        )
        r = Result.from_meshio(fake)
        np.testing.assert_array_equal(r._cell_data["weird"], tri_data)

    def test_empty_cell_data_list_is_skipped(self):
        points, tri, _ = _two_block_mesh_inputs()
        fake = _FakeInputMesh(
            points=points,
            cells=[_FakeCellBlock("triangle", tri)],
            cell_data={"empty_field": []},
        )
        r = Result.from_meshio(fake)
        self.assertNotIn("empty_field", r._cell_data)


class ToMeshioMultiBlockTests(unittest.TestCase):
    def _build_two_block_result(self) -> Result:
        points, tri, quad = _two_block_mesh_inputs()
        # 5 cells total (3 tri + 2 quad).
        mat_id = np.array([10, 11, 12, 20, 21])
        r = Result(
            backend="numpy",
            vertices=points,
            cells=[("triangle", tri), ("quad", quad)],
            cell_data={"mat_id": mat_id},
            point_data={},
        )
        return r

    def test_single_block_write_still_emits_cell_data_as_list_of_one(self):
        """Backward compatibility: single-block meshes must still produce the
        meshio-expected ``[arr]`` wrapping, no regression."""
        points = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64)
        tri = np.array([[0, 1, 2]], dtype=np.int32)
        r = Result(
            backend="numpy",
            vertices=points,
            cells=[("triangle", tri)],
            cell_data={"mat_id": np.array([7])},
            point_data={},
        )
        with _install_fake_meshio():
            m = r.to_meshio()
        self.assertIn("mat_id", m.cell_data)
        self.assertIsInstance(m.cell_data["mat_id"], list)
        self.assertEqual(len(m.cell_data["mat_id"]), 1)
        np.testing.assert_array_equal(m.cell_data["mat_id"][0], np.array([7]))

    def test_multi_block_write_splits_cell_data_by_block_size(self):
        """The fix: multi-block cell_data must emerge split at the right offsets
        rather than being dropped entirely."""
        r = self._build_two_block_result()
        with _install_fake_meshio():
            m = r.to_meshio()
        self.assertIn("mat_id", m.cell_data)
        chunks = m.cell_data["mat_id"]
        self.assertEqual(len(chunks), 2)
        np.testing.assert_array_equal(chunks[0], np.array([10, 11, 12]))
        np.testing.assert_array_equal(chunks[1], np.array([20, 21]))

    def test_multi_block_write_skips_fields_with_mismatched_length(self):
        """A cell_data entry whose length doesn't match n_cells should be
        filtered out on write (same as the single-block behavior)."""
        r = self._build_two_block_result()
        r._cell_data["bogus"] = np.array([0, 1, 2])  # only 3, but n_cells=5
        with _install_fake_meshio():
            m = r.to_meshio()
        self.assertIn("mat_id", m.cell_data)
        self.assertNotIn("bogus", m.cell_data)

    def test_multi_block_write_handles_2d_cell_data(self):
        """Per-cell vector/tensor data must split along axis 0 only, keeping
        trailing axes intact (e.g. stress (n_cells, 6))."""
        points, tri, quad = _two_block_mesh_inputs()
        stress = np.arange(5 * 3, dtype=np.float64).reshape(5, 3)
        r = Result(
            backend="numpy",
            vertices=points,
            cells=[("triangle", tri), ("quad", quad)],
            cell_data={"stress": stress},
            point_data={},
        )
        with _install_fake_meshio():
            m = r.to_meshio()
        chunks = m.cell_data["stress"]
        self.assertEqual(chunks[0].shape, (3, 3))
        self.assertEqual(chunks[1].shape, (2, 3))
        np.testing.assert_array_equal(chunks[0], stress[:3])
        np.testing.assert_array_equal(chunks[1], stress[3:])


class RoundTripTests(unittest.TestCase):
    """End-to-end: meshio -> Result -> meshio must preserve all blocks."""

    def test_multi_block_with_cell_data_round_trips_losslessly(self):
        points, tri, quad = _two_block_mesh_inputs()
        mat_id_blocks = [np.array([10, 11, 12]), np.array([20, 21])]
        stress_blocks = [
            np.arange(3 * 3, dtype=np.float64).reshape(3, 3),
            np.arange(2 * 3, dtype=np.float64).reshape(2, 3) + 100,
        ]
        fake_in = _FakeInputMesh(
            points=points,
            cells=[_FakeCellBlock("triangle", tri), _FakeCellBlock("quad", quad)],
            point_data={"u": np.zeros((len(points), 2))},
            cell_data={"mat_id": mat_id_blocks, "stress": stress_blocks},
        )

        r = Result.from_meshio(fake_in)
        with _install_fake_meshio():
            m_out = r.to_meshio()

        # Cells and point_data must survive intact.
        self.assertEqual(len(m_out.cells), 2)
        self.assertEqual(m_out.cells[0][0], "triangle")
        self.assertEqual(m_out.cells[1][0], "quad")
        self.assertIn("u", m_out.point_data)

        # Cell data must come back as per-block lists with the original shapes
        # and values, *including the block that used to get silently dropped*.
        for name, original_blocks in (("mat_id", mat_id_blocks), ("stress", stress_blocks)):
            with self.subTest(field=name):
                self.assertIn(name, m_out.cell_data)
                round_tripped = m_out.cell_data[name]
                self.assertEqual(len(round_tripped), 2)
                np.testing.assert_array_equal(round_tripped[0], original_blocks[0])
                np.testing.assert_array_equal(round_tripped[1], original_blocks[1])


if __name__ == "__main__":
    unittest.main()
