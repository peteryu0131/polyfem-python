"""Unit tests for the small internal helpers in `_solve_pipeline`.

Covers edges that the other pipeline tests don't exercise directly:
    - _promote_materials_to_list:  dict->list promotion, optional type-inference
    - _extract_additional_fields:  exception swallowing, probe ordering,
                                   stats non-dict ignore, velocity/pressure/strain

These tests never touch the C++ backend. They only exercise pure Python helpers.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from polyfempy.api._solve_pipeline import (  # noqa: E402
    _dedupe_history_frames,
    _extract_history_step_index,
    _extract_additional_fields,
    _infer_history_times,
    _promote_materials_to_list,
)


class PromoteMaterialsToListTests(unittest.TestCase):
    def test_promotes_dict_to_singleton_list(self):
        payload = {"materials": {"E": 20, "nu": 0.3}}
        _promote_materials_to_list(payload)
        self.assertEqual(payload["materials"], [{"E": 20, "nu": 0.3}])

    def test_leaves_list_materials_untouched(self):
        payload = {"materials": [{"E": 1}, {"E": 2}]}
        original = payload["materials"]
        _promote_materials_to_list(payload)
        self.assertIs(payload["materials"], original)

    def test_noop_when_materials_missing(self):
        payload = {"pde": "LinearElasticity"}
        _promote_materials_to_list(payload)
        self.assertNotIn("materials", payload)

    def test_infer_type_fills_missing_type_for_linear_elasticity(self):
        payload = {"pde": "LinearElasticity", "materials": {"E": 20, "nu": 0.3}}
        _promote_materials_to_list(payload, infer_type_from_pde=True)
        self.assertEqual(payload["materials"][0]["type"], "LinearElasticity")

    def test_infer_type_uses_laplacian_for_poisson(self):
        payload = {"pde": "Poisson", "materials": {"k": 1.0}}
        _promote_materials_to_list(payload, infer_type_from_pde=True)
        self.assertEqual(payload["materials"][0]["type"], "Laplacian")

    def test_infer_type_does_not_override_existing_type(self):
        payload = {
            "pde": "Poisson",
            "materials": {"type": "NeoHookean", "E": 20, "nu": 0.3},
        }
        _promote_materials_to_list(payload, infer_type_from_pde=True)
        self.assertEqual(payload["materials"][0]["type"], "NeoHookean")

    def test_without_infer_type_flag_no_type_is_added(self):
        payload = {"pde": "LinearElasticity", "materials": {"E": 20, "nu": 0.3}}
        _promote_materials_to_list(payload, infer_type_from_pde=False)
        self.assertNotIn("type", payload["materials"][0])


class ExtractAdditionalFieldsTests(unittest.TestCase):
    def test_stats_dict_is_merged_into_meta(self):
        class FakeSolver:
            def get_stats(self):
                return {"iters": 7, "residual": 1e-8}

        fields, meta = {}, {}
        _extract_additional_fields(FakeSolver(), fields, meta)
        self.assertEqual(meta["iters"], 7)
        self.assertAlmostEqual(meta["residual"], 1e-8)

    def test_stats_non_dict_is_silently_ignored_but_still_stops_probe(self):
        class FakeSolver:
            def get_stats(self):
                return "not-a-dict"

            def stats(self):  # next probe in the same group
                return {"fallback_used": True}

        fields, meta = {}, {}
        _extract_additional_fields(FakeSolver(), fields, meta)
        # get_stats returned a non-dict but non-None value, so the probe loop
        # stops there (matching legacy behavior) and `.stats()` is NOT tried.
        self.assertNotIn("fallback_used", meta)

    def test_probe_falls_through_to_next_name_on_exception(self):
        class FakeSolver:
            def get_stress(self):
                raise RuntimeError("primary probe broken")

            def get_cauchy_stress(self):
                return np.array([[1.0, 2.0, 3.0]])

        fields, meta = {}, {}
        _extract_additional_fields(FakeSolver(), fields, meta)
        # The raising probe was skipped and the secondary probe won.
        self.assertIn("stress", fields)
        self.assertEqual(fields["stress"].shape, (1, 3))

    def test_probe_falls_through_to_next_name_on_none(self):
        class FakeSolver:
            def get_stress(self):
                return None  # first probe declines

            def stress(self):
                return np.array([[7.0, 8.0, 9.0]])

        fields, meta = {}, {}
        _extract_additional_fields(FakeSolver(), fields, meta)
        self.assertIn("stress", fields)
        np.testing.assert_array_equal(fields["stress"], np.array([[7.0, 8.0, 9.0]]))

    def test_pressure_and_velocity_land_under_expected_keys(self):
        class FakeSolver:
            def get_pressure(self):
                return np.array([0.1, 0.2, 0.3])

            def get_velocity(self):
                return np.array([[1.0, 0.0], [0.0, 1.0]])

        fields, meta = {}, {}
        _extract_additional_fields(FakeSolver(), fields, meta)
        self.assertIn("p", fields)
        self.assertIn("v", fields)
        self.assertEqual(fields["v"].shape, (2, 2))

    def test_energy_scalar_goes_to_meta_array_goes_to_fields(self):
        class FakeSolverScalar:
            def get_energy(self):
                return 2.5

        class FakeSolverArray:
            def get_energy(self):
                return np.array([1.0, 2.0, 3.0])

        fields_a, meta_a = {}, {}
        _extract_additional_fields(FakeSolverScalar(), fields_a, meta_a)
        self.assertEqual(meta_a["energy"], 2.5)
        self.assertNotIn("energy", fields_a)

        fields_b, meta_b = {}, {}
        _extract_additional_fields(FakeSolverArray(), fields_b, meta_b)
        self.assertNotIn("energy", meta_b)
        self.assertEqual(fields_b["energy"].shape, (3,))

    def test_strain_getter_populates_strain_field(self):
        class FakeSolver:
            def get_strain(self):
                return np.array([[0.01, 0.02, 0.03]])

        fields, meta = {}, {}
        _extract_additional_fields(FakeSolver(), fields, meta)
        self.assertIn("strain", fields)

    def test_empty_solver_leaves_fields_and_meta_unchanged(self):
        class FakeSolver:
            pass

        fields, meta = {}, {}
        _extract_additional_fields(FakeSolver(), fields, meta)
        self.assertEqual(fields, {})
        self.assertEqual(meta, {})


class InferHistoryTimesTests(unittest.TestCase):
    def test_extract_history_step_index_only_accepts_step_suffix(self):
        self.assertEqual(_extract_history_step_index("impact_step_12.vtu"), 12)
        self.assertEqual(_extract_history_step_index("/tmp/run/impact_step_3.vtu"), 3)
        self.assertIsNone(_extract_history_step_index("solve_12.vtu"))
        self.assertIsNone(_extract_history_step_index("initial"))

    def test_dedupes_consecutive_duplicate_steps_keeping_last_frame(self):
        frames = [
            {"name": "impact_step_0.vtu", "marker": "first"},
            {"name": "impact_step_0.vtu", "marker": "second"},
            {"name": "impact_step_1.vtu", "marker": "third"},
            {"name": "impact_step_2.vtu", "marker": "fourth"},
        ]
        deduped = _dedupe_history_frames(frames)
        self.assertEqual([f["marker"] for f in deduped], ["second", "third", "fourth"])

    def test_leaves_frames_unchanged_when_names_are_not_step_based(self):
        frames = [{"name": "solve_0.vtu"}, {"name": "solve_0.vtu"}]
        deduped = _dedupe_history_frames(frames)
        self.assertEqual(deduped, frames)

    def test_uses_step_indices_parsed_from_frame_names(self):
        frames = [
            {"name": "/tmp/run_177/impact_step_0.vtu"},
            {"name": "/tmp/run_177/impact_step_0.vtu"},
            {"name": "/tmp/run_177/impact_step_1.vtu"},
            {"name": "/tmp/run_177/impact_step_2.vtu"},
        ]
        full_json = {"time": {"t0": 0.0, "dt": 0.01, "tend": 0.02}}
        self.assertEqual(
            _infer_history_times(frames, full_json),
            [0.0, 0.0, 0.01, 0.02],
        )

    def test_respects_skipped_saved_steps(self):
        frames = [
            {"name": "impact_step_0.vtu"},
            {"name": "impact_step_2.vtu"},
            {"name": "impact_step_4.vtu"},
        ]
        full_json = {"time": {"t0": 1.5, "dt": 0.25}}
        self.assertEqual(
            _infer_history_times(frames, full_json),
            [1.5, 2.0, 2.5],
        )

    def test_falls_back_to_frame_order_when_names_are_not_parseable(self):
        frames = [{"name": "initial"}, {"name": "after_contact"}, {"name": "final"}]
        full_json = {"time": {"t0": 0.0, "dt": 0.1}}
        self.assertEqual(
            _infer_history_times(frames, full_json),
            [0.0, 0.1, 0.2],
        )

    def test_returns_none_without_positive_dt(self):
        frames = [{"name": "impact_step_0.vtu"}]
        self.assertIsNone(_infer_history_times(frames, {"time": {"t0": 0.0, "dt": 0.0}}))


if __name__ == "__main__":
    unittest.main()
