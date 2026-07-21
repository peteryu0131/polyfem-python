"""Unit tests for ``Result.history`` / ``HistoryView``.

Round 1 (the current C++ binding change) exposes ``solver.solution_frames`` so
Python can read PolyFEM's in-memory per-timestep data without going through a
VTU file. These tests cover the Python-side wrapper (``HistoryView``) in
isolation — they fabricate frame dicts that mimic what the nanobind binding
produces, so no C++ backend or compiled extension is required.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from polyfempy.runtime.result import HistoryView, Result  # noqa: E402


def _make_frame(step: int, n_sampled: int = 4, dim: int = 2):
    """Produce a fake SolutionFrame-equivalent dict."""
    return {
        "name": f"step_{step}",
        "points": np.arange(n_sampled * dim, dtype=np.float64).reshape(n_sampled, dim),
        "connectivity": np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32),
        "solution": np.full((n_sampled, dim), float(step), dtype=np.float64),
        "pressure": np.empty((0, 0)),
        "scalar_value": np.full((n_sampled, 1), float(step) + 0.5, dtype=np.float64),
        "scalar_value_avg": np.full((n_sampled, 1), float(step) + 0.25, dtype=np.float64),
        "tensor_value": np.full((n_sampled, dim * dim), float(step) + 9.0, dtype=np.float64),
        "body_ids": np.array([[1], [1], [2], [2]], dtype=np.int32),
        "exact": np.empty((0, 0)),
        "error": np.empty((0, 0)),
    }


class HistoryViewBasicTests(unittest.TestCase):
    def test_empty_history_is_falsy_and_reports_unavailable(self):
        h = HistoryView()
        self.assertFalse(h.available)
        self.assertFalse(bool(h))
        self.assertEqual(len(h), 0)
        self.assertEqual(h.u.size, 0)
        self.assertEqual(h.vm.size, 0)

    def test_history_stacks_u_with_time_axis(self):
        frames = [_make_frame(i) for i in range(3)]
        h = HistoryView(frames=frames)
        self.assertTrue(h.available)
        self.assertEqual(len(h), 3)
        # (n_steps, n_sampled, dim)
        self.assertEqual(h.u.shape, (3, 4, 2))
        # Step i has all entries equal to i (from the fake fixture).
        np.testing.assert_array_equal(h.u[0], np.zeros((4, 2)))
        np.testing.assert_array_equal(h.u[1], np.ones((4, 2)))
        np.testing.assert_array_equal(h.u[2], 2 * np.ones((4, 2)))

    def test_vm_trailing_singleton_is_squeezed(self):
        """scalar_value comes back as (n_sampled, 1) from PolyFEM. The view
        must squeeze that trailing axis so the shape becomes
        (n_steps, n_sampled) — matches how we flatten body_ids."""
        frames = [_make_frame(i) for i in range(2)]
        h = HistoryView(frames=frames)
        self.assertEqual(h.vm.shape, (2, 4))
        self.assertEqual(h.vm_avg.shape, (2, 4))

    def test_static_geometry_comes_from_first_frame(self):
        frames = [_make_frame(i) for i in range(2)]
        h = HistoryView(frames=frames)
        np.testing.assert_array_equal(h.points, frames[0]["points"])
        np.testing.assert_array_equal(h.connectivity, frames[0]["connectivity"])

    def test_stress_is_stacked_with_time_axis(self):
        frames = [_make_frame(i) for i in range(3)]
        h = HistoryView(frames=frames)
        self.assertEqual(h.stress.shape, (3, 4, 4))
        np.testing.assert_array_equal(h.stress[0], 9.0 * np.ones((4, 4)))
        np.testing.assert_array_equal(h.stress[2], 11.0 * np.ones((4, 4)))

    def test_body_ids_are_static_and_flattened(self):
        frames = [_make_frame(i) for i in range(2)]
        h = HistoryView(frames=frames)
        self.assertEqual(h.body_ids.shape, (4,))
        np.testing.assert_array_equal(h.body_ids, np.array([1, 1, 2, 2], dtype=np.int32))

    def test_times_default_to_step_indices(self):
        frames = [_make_frame(i) for i in range(4)]
        h = HistoryView(frames=frames)
        np.testing.assert_array_equal(h.times, np.array([0.0, 1.0, 2.0, 3.0]))

    def test_times_can_be_supplied_explicitly(self):
        frames = [_make_frame(i) for i in range(3)]
        h = HistoryView(frames=frames, times=[0.0, 0.01, 0.02])
        np.testing.assert_allclose(h.times, [0.0, 0.01, 0.02])


class HistoryFieldByBodyTests(unittest.TestCase):
    def _hist_of_4(self):
        return HistoryView(frames=[_make_frame(i, n_sampled=4) for i in range(3)])

    def test_splits_vm_across_two_bodies_preserving_time_axis(self):
        h = self._hist_of_4()
        body_ids = np.array([1, 1, 2, 2], dtype=np.int32)
        parts = h.field_by_body("vm", body_ids)
        # (n_steps, N1) and (n_steps, N2), int keys.
        self.assertEqual(sorted(parts.keys()), [1, 2])
        self.assertEqual(parts[1].shape, (3, 2))
        self.assertEqual(parts[2].shape, (3, 2))

    def test_splits_u_preserving_trailing_dim(self):
        """u is (n_steps, n_sampled, dim); the slicing must only touch the
        sampled axis and keep dim intact."""
        h = self._hist_of_4()
        body_ids = np.array([1, 1, 2, 2], dtype=np.int32)
        parts = h.field_by_body("u", body_ids)
        self.assertEqual(parts[1].shape, (3, 2, 2))
        self.assertEqual(parts[2].shape, (3, 2, 2))

    def test_splits_stress_preserving_tensor_axis(self):
        h = self._hist_of_4()
        body_ids = np.array([1, 1, 2, 2], dtype=np.int32)
        parts = h.field_by_body("stress", body_ids)
        self.assertEqual(parts[1].shape, (3, 2, 4))
        self.assertEqual(parts[2].shape, (3, 2, 4))

    def test_raises_when_body_ids_length_mismatches_sampled_axis(self):
        h = self._hist_of_4()
        wrong_ids = np.array([1, 1, 2])  # only 3, but n_sampled=4
        with self.assertRaisesRegex(ValueError, "can't split"):
            h.field_by_body("vm", wrong_ids)

    def test_raises_on_missing_field_name(self):
        h = self._hist_of_4()
        with self.assertRaises(KeyError):
            h.field_by_body("not_a_field", np.array([1, 1, 2, 2]))


class ResultIntegrationTests(unittest.TestCase):
    """``Result`` accepts either a HistoryView directly or a raw list of
    frame dicts. Either way the user sees a consistent ``result.history``."""

    def _native_result(self, history):
        V = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        C = np.array([[0, 1, 2]], dtype=np.int32)
        return Result("numpy", V, C, fields={"u": V.copy()}, history=history)

    def test_result_accepts_history_view(self):
        h = HistoryView(frames=[_make_frame(i) for i in range(2)])
        r = self._native_result(history=h)
        self.assertIs(r.history, h)
        self.assertTrue(r.history.available)

    def test_result_accepts_raw_frames_list(self):
        frames = [_make_frame(i) for i in range(2)]
        r = self._native_result(history=frames)
        self.assertIsInstance(r.history, HistoryView)
        self.assertEqual(len(r.history), 2)

    def test_result_history_is_empty_by_default(self):
        V = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        C = np.array([[0, 1, 2]], dtype=np.int32)
        r = Result("numpy", V, C)
        self.assertFalse(r.history.available)
        self.assertEqual(len(r.history), 0)

    def test_body_ids_property_delegates_to_field_lookup(self):
        V = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        C = np.array([[0, 1, 2]], dtype=np.int32)
        r = Result("numpy", V, C)
        r.set_sampled_field("body_ids", np.array([1, 1, 2, 2], dtype=np.int32))
        np.testing.assert_array_equal(r.body_ids, [1, 1, 2, 2])

    def test_body_ids_property_falls_back_to_history(self):
        V = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        C = np.array([[0, 1, 2]], dtype=np.int32)
        r = Result("numpy", V, C, history=[_make_frame(0), _make_frame(1)])
        np.testing.assert_array_equal(r.body_ids, [1, 1, 2, 2])


if __name__ == "__main__":
    unittest.main()
