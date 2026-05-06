"""Unit tests for the sampled-data namespace on ``Result``.

The sampled-VTU fallback populates stress / von_mises from a probe mesh that
is NOT aligned with ``result.vertices`` / ``result.cells``. Before T4 these
values were shoved into ``point_data``, making ``result.stress`` look like a
per-vertex native field even when it was sampled. That broke both the mental
model and ``to_meshio()`` output.

These tests pin down the new behavior:

    - ``set_sampled_field`` never writes into point_data or cell_data
    - ``result.stress`` / ``result.von_mises`` still find sampled values
    - ``to_meshio()`` excludes sampled fields from its output
    - native point/cell data takes priority over sampled when both exist
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
from polyfempy.api._solve_pipeline import _field_available  # noqa: E402


def _make_native_result(*, with_native_stress: bool = False) -> Result:
    """Result with a 3-vertex / 1-triangle native mesh."""
    V = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    C = np.array([[0, 1, 2]], dtype=np.int32)
    fields = {"u": np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]])}
    if with_native_stress:
        # 3 vertices × Voigt-3 (2D) native stress on the actual mesh.
        fields["stress"] = np.array([[1.0, 2.0, 0.5]] * 3)
    return Result("numpy", V, C, fields=fields)


class SetSampledFieldTests(unittest.TestCase):
    def test_sampled_field_does_not_leak_into_point_data(self):
        r = _make_native_result()
        sampled_stress = np.arange(5 * 3, dtype=np.float64).reshape(5, 3)
        r.set_sampled_field("stress", sampled_stress)

        self.assertNotIn("stress", r._point_data)
        self.assertNotIn("stress", r._cell_data)
        self.assertIn("stress", r._sampled_data)

    def test_sampled_length_matching_nvertices_still_goes_to_sampled(self):
        """Even when the sampled array length happens to equal n_vertices,
        it must NOT be parked in point_data — the meshes are not the same
        just because the counts coincide."""
        r = _make_native_result()
        coincident = np.array([[0.0, 0.0, 0.0]] * r.n_vertices)
        r.set_sampled_field("stress", coincident)

        self.assertNotIn("stress", r._point_data)
        self.assertIn("stress", r._sampled_data)

    def test_returns_self_for_chaining(self):
        r = _make_native_result()
        self.assertIs(r.set_sampled_field("x", np.array([1.0])), r)


class FieldLookupPrecedenceTests(unittest.TestCase):
    def test_stress_property_finds_sampled_value_when_no_native(self):
        r = _make_native_result()
        sampled = np.array([[9.0, 9.0, 9.0]] * 5)
        r.set_sampled_field("stress", sampled)

        got = r.stress
        self.assertIsNotNone(got)
        np.testing.assert_array_equal(got, sampled)

    def test_native_point_data_wins_over_sampled(self):
        r = _make_native_result(with_native_stress=True)
        r.set_sampled_field("stress", np.array([[99.0, 99.0, 99.0]] * 5))
        # Native stress sits in _point_data; it must beat _sampled_data.
        np.testing.assert_array_equal(r.stress, np.array([[1.0, 2.0, 0.5]] * 3))

    def test_field_available_sees_sampled_stress(self):
        r = _make_native_result()
        r.set_sampled_field("stress", np.array([[1.0, 2.0, 3.0]] * 4))
        self.assertTrue(_field_available(r, "stress"))

    def test_von_mises_property_computes_from_sampled_stress(self):
        """``result.von_mises`` must be able to derive von Mises from stress
        even when stress only lives in sampled_data."""
        r = _make_native_result()
        # (n, 3) Voigt: [sxx, syy, sxy], 2D. With sxx=1, syy=0, sxy=0 the
        # 2D plane-stress von Mises formula simplifies to sqrt(sxx^2) = 1.
        sampled = np.array([[1.0, 0.0, 0.0]] * 4)
        r.set_sampled_field("stress", sampled)
        vm = r.von_mises
        self.assertIsNotNone(vm)
        np.testing.assert_allclose(vm, np.array([1.0] * 4))

    def test_von_mises_property_prefers_sampled_von_mises_directly(self):
        r = _make_native_result()
        vm_direct = np.array([7.0, 8.0, 9.0])
        r.set_sampled_field("von_mises", vm_direct)
        np.testing.assert_array_equal(r.von_mises, vm_direct)


class RemoveFieldTests(unittest.TestCase):
    def test_remove_field_clears_sampled_too(self):
        r = _make_native_result()
        r.set_sampled_field("stress", np.array([[1.0, 2.0, 3.0]] * 4))
        r.remove_field("stress")
        self.assertNotIn("stress", r._sampled_data)
        self.assertIsNone(r.field("stress"))


class ToMeshioExcludesSampledTests(unittest.TestCase):
    def test_to_meshio_does_not_attach_sampled_fields_to_native_mesh(self):
        """The core semantic fix: sampled fields must not appear in the
        meshio ``point_data`` or ``cell_data`` dicts, regardless of their
        array length."""
        r = _make_native_result()
        r.set_sampled_field("stress", np.array([[1.0, 2.0, 3.0]] * 4))

        recorded = {}

        class _RecordingMesh:
            def __init__(self, vertices, cells, point_data=None, cell_data=None):
                recorded["vertices"] = vertices
                recorded["cells"] = cells
                recorded["point_data"] = point_data or {}
                recorded["cell_data"] = cell_data or {}

        fake_meshio = types.SimpleNamespace(Mesh=_RecordingMesh)
        with unittest.mock.patch.dict(sys.modules, {"meshio": fake_meshio}):
            r.to_meshio()

        self.assertNotIn("stress", recorded["point_data"])
        self.assertNotIn("stress", recorded["cell_data"])

    def test_to_meshio_still_emits_native_fields(self):
        r = _make_native_result(with_native_stress=True)
        r.set_sampled_field("von_mises", np.array([99.0] * 8))

        recorded = {}

        class _RecordingMesh:
            def __init__(self, vertices, cells, point_data=None, cell_data=None):
                recorded["point_data"] = point_data or {}
                recorded["cell_data"] = cell_data or {}

        fake_meshio = types.SimpleNamespace(Mesh=_RecordingMesh)
        with unittest.mock.patch.dict(sys.modules, {"meshio": fake_meshio}):
            r.to_meshio()

        self.assertIn("u", recorded["point_data"])
        self.assertIn("stress", recorded["point_data"])  # native stress, kept
        self.assertNotIn("von_mises", recorded["point_data"])  # sampled, dropped


class FieldByBodyTests(unittest.TestCase):
    """``Result.field_by_body`` replaces the inline numpy boolean-masking
    pattern. It must: (a) only work when body_ids is available, (b) only
    work on fields whose row count matches body_ids, (c) return a dict
    keyed by int body ids with contiguous sub-arrays."""

    def _sampled_result(self) -> Result:
        r = _make_native_result()
        r.set_sampled_field("stress", np.arange(4 * 3, dtype=np.float64).reshape(4, 3))
        r.set_sampled_field("body_ids", np.array([1, 1, 2, 2], dtype=np.int32))
        return r

    def test_splits_sampled_field_by_body(self):
        r = self._sampled_result()
        out = r.field_by_body("stress")

        self.assertEqual(sorted(out.keys()), [1, 2])
        self.assertEqual(out[1].shape, (2, 3))
        self.assertEqual(out[2].shape, (2, 3))
        np.testing.assert_array_equal(out[1], np.array([[0, 1, 2], [3, 4, 5]]))
        np.testing.assert_array_equal(out[2], np.array([[6, 7, 8], [9, 10, 11]]))

    def test_keys_are_plain_ints_not_numpy_scalars(self):
        """Python code often does ``d[1]``; int keys avoid surprises from
        numpy int scalar types in downstream consumers."""
        r = self._sampled_result()
        out = r.field_by_body("stress")
        for key in out:
            self.assertIsInstance(key, int)

    def test_raises_when_body_ids_absent(self):
        r = _make_native_result()
        r.set_sampled_field("stress", np.zeros((4, 3)))
        # No body_ids -> can't split.
        with self.assertRaisesRegex(RuntimeError, "body_ids"):
            r.field_by_body("stress")

    def test_raises_for_unknown_field(self):
        r = self._sampled_result()
        with self.assertRaises(KeyError):
            r.field_by_body("does_not_exist")

    def test_raises_when_field_row_count_mismatches_body_ids(self):
        """Splitting a native-mesh field (e.g. u, rows=n_vertices) by a
        sampled-mesh body_ids (rows=n_sampled) must fail with a clear error,
        not return a silently-wrong dict."""
        r = self._sampled_result()
        # ``u`` on the native mesh lives in _point_data with 3 rows, while
        # body_ids has 4 rows. Different meshes; splitting doesn't make sense.
        with self.assertRaisesRegex(ValueError, r"lengths do not match"):
            r.field_by_body("u")


class IntrospectionTests(unittest.TestCase):
    def test_flat_displacement_vector_is_reshaped_to_vertex_dim(self):
        V = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        C = np.array([[0, 1, 2]], dtype=np.int32)
        flat_u = np.arange(6, dtype=np.float64).reshape(6, 1)
        r = Result("numpy", V, C, fields={"u": flat_u})

        np.testing.assert_array_equal(
            r.point_field("u"),
            np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]),
        )

    def test_field_names_includes_sampled(self):
        r = _make_native_result(with_native_stress=True)
        r.set_sampled_field("von_mises", np.array([1.0, 2.0, 3.0]))
        names = set(r.field_names())
        self.assertIn("u", names)
        self.assertIn("stress", names)
        self.assertIn("von_mises", names)

    def test_available_fields_groups_names_by_namespace(self):
        r = _make_native_result(with_native_stress=True)
        r.cell_data["mat_id"] = np.array([7], dtype=np.int32)
        r.set_sampled_field("von_mises", np.array([1.0, 2.0, 3.0]))

        self.assertEqual(
            r.available_fields(),
            {
                "point_data": ["stress", "u"],
                "cell_data": ["mat_id"],
                "sampled_data": ["von_mises"],
            },
        )

    def test_namespace_specific_field_accessors_do_not_fall_through(self):
        r = _make_native_result(with_native_stress=True)
        sampled_stress = np.array([[99.0, 99.0, 99.0]] * 5)
        r.set_sampled_field("stress", sampled_stress)

        np.testing.assert_array_equal(
            r.point_field("stress"),
            np.array([[1.0, 2.0, 0.5]] * 3),
        )
        np.testing.assert_array_equal(r.sampled_field("stress"), sampled_stress)
        self.assertIsNone(r.cell_field("stress"))
        self.assertEqual(r.cell_field("stress", default="missing"), "missing")

    def test_summary_reports_sampled_data_namespace(self):
        r = _make_native_result()
        r.set_sampled_field("stress", np.array([[1.0, 2.0, 3.0]] * 4))
        summary = r.summary()
        self.assertIn("sampled_data", summary)
        self.assertEqual(summary["sampled_data"]["stress"], (4, 3))

    def test_sampled_data_proxy_supports_basic_dict_operations(self):
        r = _make_native_result()
        r.sampled_data["von_mises"] = np.array([1.0, 2.0])
        self.assertIn("von_mises", r.sampled_data)
        self.assertEqual(list(r.sampled_data.keys()), ["von_mises"])
        np.testing.assert_array_equal(
            r.sampled_data["von_mises"], np.array([1.0, 2.0])
        )
        del r.sampled_data["von_mises"]
        self.assertNotIn("von_mises", r.sampled_data)

    def test_init_accepts_sampled_data_kwarg(self):
        V = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        C = np.array([[0, 1, 2]], dtype=np.int32)
        r = Result(
            "numpy",
            V,
            C,
            point_data={},
            cell_data={},
            sampled_data={"stress": np.array([[1.0, 2.0, 3.0]] * 4)},
        )
        self.assertIn("stress", r._sampled_data)
        self.assertIsNotNone(r.stress)


if __name__ == "__main__":
    unittest.main()
