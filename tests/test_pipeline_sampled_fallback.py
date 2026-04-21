"""Integration tests for ``apply_sampled_vtu_fallback``.

These tests verify that the fallback path wires sampled stress / von_mises
into ``Result._sampled_data`` (the new post-T4 namespace) and never into
``point_data`` / ``cell_data``, regardless of the sampled arrays' lengths.

We stub out the heavy bits — ``_export_and_read_vtu`` (which would normally
write + read a VTU through meshio) and the two extraction helpers — so the
test focuses only on the routing/plumbing change.
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

from polyfempy.api import _solve_pipeline as _p  # noqa: E402
from polyfempy.api._solve_pipeline import (  # noqa: E402
    NativeOutputs,
    RuntimeOptions,
    apply_sampled_vtu_fallback,
)
from polyfempy.api.result import Result  # noqa: E402


class _FakeSolver:
    """Minimal solver stand-in: just needs ``export_vtu`` to pass the
    ``hasattr(solver, 'export_vtu')`` gate inside the fallback."""

    def export_vtu(self, *_args, **_kwargs):  # pragma: no cover - unused
        raise AssertionError(
            "export_vtu should not be called when _export_and_read_vtu is mocked"
        )


def _native_result_and_outputs():
    V = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    C = np.array([[0, 1, 2]], dtype=np.int32)
    u = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]])
    result = Result("numpy", V, C, fields={"u": u})
    native = NativeOutputs(
        vertices=V, cells=C, fields={"u": u}, meta={"solver_type": "FakeSolver"}
    )
    return result, native


class SampledFallbackRoutingTests(unittest.TestCase):
    def test_fallback_routes_stress_to_sampled_not_point_data(self):
        """The core T4 fix: fallback stress must land in _sampled_data."""
        result, native = _native_result_and_outputs()
        sampled_stress = np.arange(4 * 3, dtype=np.float64).reshape(4, 3)
        fake_mesh = types.SimpleNamespace(
            points=np.zeros((4, 2)),
            point_data={},
            cell_data={},
        )

        with unittest.mock.patch.object(
            _p, "_export_and_read_vtu", return_value=(fake_mesh, None)
        ), unittest.mock.patch.object(
            _p, "_reconstruct_sampled_cauchy_stress", return_value=(sampled_stress, "point")
        ), unittest.mock.patch.object(
            _p, "_extract_meshio_array", return_value=(None, None)
        ):
            r = apply_sampled_vtu_fallback(
                result,
                solver=_FakeSolver(),
                native=native,
                full_json=None,
                runtime=RuntimeOptions(fallback_mode="always"),
            )

        self.assertIn("stress", r._sampled_data)
        np.testing.assert_array_equal(r._sampled_data["stress"], sampled_stress)
        self.assertNotIn("stress", r._point_data)
        self.assertNotIn("stress", r._cell_data)
        self.assertEqual(r.meta.get("stress_source"), "temp_vtu_sampled_cauchy")
        self.assertEqual(r.meta.get("stress_location"), "point")
        self.assertTrue(r.meta.get("sampled_vtu_fallback"))

    def test_fallback_routes_von_mises_to_sampled_not_point_data(self):
        result, native = _native_result_and_outputs()
        sampled_vm = np.array([1.0, 2.0, 3.0, 4.0])
        fake_mesh = types.SimpleNamespace(
            points=np.zeros((4, 2)),
            point_data={"von_mises": sampled_vm},
            cell_data={},
        )

        # Keep the real _extract_meshio_array: it's a pure reader, no VTU I/O.
        with unittest.mock.patch.object(
            _p, "_export_and_read_vtu", return_value=(fake_mesh, None)
        ), unittest.mock.patch.object(
            _p, "_reconstruct_sampled_cauchy_stress", return_value=(None, None)
        ):
            r = apply_sampled_vtu_fallback(
                result,
                solver=_FakeSolver(),
                native=native,
                full_json=None,
                runtime=RuntimeOptions(fallback_mode="always"),
            )

        self.assertIn("von_mises", r._sampled_data)
        np.testing.assert_array_equal(r._sampled_data["von_mises"], sampled_vm)
        self.assertNotIn("von_mises", r._point_data)
        self.assertEqual(r.meta.get("von_mises_source"), "temp_vtu")

    def test_fallback_keeps_stress_reachable_via_result_stress(self):
        """After fallback, ``result.stress`` / ``result.field('stress')`` must
        still return the sampled value (via the new priority-fall-through).
        Existing consumers keep working without knowing it's sampled."""
        result, native = _native_result_and_outputs()
        sampled_stress = np.arange(4 * 3, dtype=np.float64).reshape(4, 3)
        fake_mesh = types.SimpleNamespace(
            points=np.zeros((4, 2)), point_data={}, cell_data={}
        )

        with unittest.mock.patch.object(
            _p, "_export_and_read_vtu", return_value=(fake_mesh, None)
        ), unittest.mock.patch.object(
            _p, "_reconstruct_sampled_cauchy_stress", return_value=(sampled_stress, "point")
        ), unittest.mock.patch.object(
            _p, "_extract_meshio_array", return_value=(None, None)
        ):
            r = apply_sampled_vtu_fallback(
                result,
                solver=_FakeSolver(),
                native=native,
                full_json=None,
                runtime=RuntimeOptions(fallback_mode="always"),
            )

        np.testing.assert_array_equal(r.stress, sampled_stress)
        np.testing.assert_array_equal(r.field("stress"), sampled_stress)

    def test_fallback_noop_when_mode_is_never(self):
        result, native = _native_result_and_outputs()
        r = apply_sampled_vtu_fallback(
            result,
            solver=_FakeSolver(),
            native=native,
            full_json=None,
            runtime=RuntimeOptions(fallback_mode="never"),
        )
        # No sampled data should have been added.
        self.assertEqual(r._sampled_data, {})
        self.assertNotIn("sampled_vtu_fallback", r.meta)

    def test_fallback_does_not_clobber_native_stress_if_present(self):
        """If the native solver already produced stress, the fallback still
        lands in _sampled_data — but ``result.stress`` keeps returning the
        native value (priority: point_data > cell_data > sampled_data)."""
        V = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        C = np.array([[0, 1, 2]], dtype=np.int32)
        native_stress = np.array([[1.0, 2.0, 0.5]] * 3)
        result = Result(
            "numpy",
            V,
            C,
            fields={
                "u": np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]]),
                "stress": native_stress,
            },
        )
        native = NativeOutputs(
            vertices=V,
            cells=C,
            fields={"u": result.u, "stress": native_stress},
            meta={"solver_type": "FakeSolver"},
        )

        sampled_stress = np.arange(4 * 3, dtype=np.float64).reshape(4, 3)
        fake_mesh = types.SimpleNamespace(
            points=np.zeros((4, 2)), point_data={}, cell_data={}
        )
        with unittest.mock.patch.object(
            _p, "_export_and_read_vtu", return_value=(fake_mesh, None)
        ), unittest.mock.patch.object(
            _p, "_reconstruct_sampled_cauchy_stress", return_value=(sampled_stress, "point")
        ), unittest.mock.patch.object(
            _p, "_extract_meshio_array", return_value=(None, None)
        ):
            r = apply_sampled_vtu_fallback(
                result,
                solver=_FakeSolver(),
                native=native,
                full_json=None,
                runtime=RuntimeOptions(fallback_mode="always"),
            )

        np.testing.assert_array_equal(r.stress, native_stress)
        np.testing.assert_array_equal(r._sampled_data["stress"], sampled_stress)


if __name__ == "__main__":
    unittest.main()
