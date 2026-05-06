"""Integration tests for history/exported-VTU sampled-field backfill.

The old temporary-``export_vtu()`` probe path is gone. These tests verify the
new two-source policy:

1. Prefer ``result.history`` (in-memory ``solution_frames``) and project its
   final frame into ``result.sampled_data`` for convenience.
2. If in-memory history is empty, allow the pipeline to adopt history rebuilt
   from user-exported ``impact_step_*.vtu`` files.
"""

from __future__ import annotations

import sys
import unittest
import unittest.mock
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import polyfempy.api._solve_pipeline as _p  # noqa: E402
from polyfempy.api._solve_pipeline import (  # noqa: E402
    NativeOutputs,
    RuntimeOptions,
    apply_sampled_vtu_fallback,
)
from polyfempy.api.result import HistoryView, Result  # noqa: E402


def _make_frame(step: int, *, n_sampled: int = 4, dim: int = 2):
    return {
        "name": f"impact_step_{step}.vtu",
        "points": np.arange(n_sampled * dim, dtype=np.float64).reshape(n_sampled, dim),
        "connectivity": np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32),
        "solution": np.full((n_sampled, dim), float(step), dtype=np.float64),
        "pressure": np.empty((0, 0)),
        "scalar_value": np.full((n_sampled, 1), float(step) + 0.5, dtype=np.float64),
        "scalar_value_avg": np.full((n_sampled, 1), float(step) + 0.25, dtype=np.float64),
        "tensor_value": np.full((n_sampled, dim * dim), float(step) + 9.0, dtype=np.float64),
        "body_ids": np.array([[1], [1], [2], [2]], dtype=np.int32),
    }


def _native_result_and_outputs(*, history=None):
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    cells = np.array([[0, 1, 2]], dtype=np.int32)
    u = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]])
    result = Result("numpy", vertices, cells, fields={"u": u}, history=history)
    native = NativeOutputs(
        vertices=vertices,
        cells=cells,
        fields={"u": u},
        meta={"solver_type": "FakeSolver"},
    )
    return result, native


class HistoryBackfillTests(unittest.TestCase):
    def test_projects_last_history_frame_into_sampled_data(self):
        history = HistoryView(frames=[_make_frame(0), _make_frame(1)], times=[0.0, 0.01])
        result, native = _native_result_and_outputs(history=history)

        r = apply_sampled_vtu_fallback(
            result,
            solver=None,
            native=native,
            full_json=None,
            runtime=RuntimeOptions(),
        )

        np.testing.assert_array_equal(r._sampled_data["stress"], history.stress[-1])
        np.testing.assert_array_equal(r._sampled_data["von_mises"], history.vm[-1])
        np.testing.assert_array_equal(r._sampled_data["von_mises_avg"], history.vm_avg[-1])
        np.testing.assert_array_equal(r._sampled_data["body_ids"], np.array([1, 1, 2, 2]))
        self.assertEqual(r.meta.get("stress_source"), "history:last_frame")
        self.assertEqual(r.meta.get("von_mises_source"), "history:last_frame")

    def test_does_not_clobber_native_stress_when_history_is_projected(self):
        history = HistoryView(frames=[_make_frame(0), _make_frame(1)], times=[0.0, 0.01])
        result, native = _native_result_and_outputs(history=history)
        native_stress = np.array([[1.0, 2.0, 0.5]] * 3)
        result.set_field("stress", native_stress)
        native.fields["stress"] = native_stress

        r = apply_sampled_vtu_fallback(
            result,
            solver=None,
            native=native,
            full_json=None,
            runtime=RuntimeOptions(),
        )

        np.testing.assert_array_equal(r.stress, native_stress)
        np.testing.assert_array_equal(r._sampled_data["stress"], history.stress[-1])

    def test_uses_exported_history_when_in_memory_history_is_empty(self):
        result, native = _native_result_and_outputs(history=None)
        exported_history = HistoryView(
            frames=[_make_frame(0), _make_frame(1), _make_frame(2)],
            times=[0.0, 0.01, 0.02],
        )
        exported_history.source = "exported_vtu_sequence"

        with unittest.mock.patch.object(
            _p,
            "_collect_history_from_exported_vtus",
            return_value=exported_history,
        ):
            r = apply_sampled_vtu_fallback(
                result,
                solver=None,
                native=native,
                full_json={"output": {"directory": "/tmp"}},
                runtime=RuntimeOptions(),
            )

        self.assertTrue(r.history.available)
        self.assertEqual(len(r.history), 3)
        np.testing.assert_array_equal(r._sampled_data["stress"], exported_history.stress[-1])
        np.testing.assert_array_equal(r._sampled_data["von_mises"], exported_history.vm[-1])
        self.assertTrue(r.meta.get("sampled_vtu_fallback"))
        self.assertEqual(r.meta.get("sampled_vtu_fallback_mode"), "exported_files")
        self.assertEqual(r.meta.get("stress_source"), "exported_vtu:last_frame")

    def test_noop_when_neither_history_nor_exported_vtu_is_available(self):
        result, native = _native_result_and_outputs(history=None)

        with unittest.mock.patch.object(
            _p,
            "_collect_history_from_exported_vtus",
            return_value=HistoryView(),
        ):
            r = apply_sampled_vtu_fallback(
                result,
                solver=None,
                native=native,
                full_json=None,
                runtime=RuntimeOptions(),
            )

        self.assertFalse(r.history.available)
        self.assertEqual(r._sampled_data, {})
        self.assertNotIn("sampled_vtu_fallback", r.meta)


if __name__ == "__main__":
    unittest.main()
