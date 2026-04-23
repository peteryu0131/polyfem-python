"""Tests for the public result-reporting helpers."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from polyfempy.api.report import (  # noqa: E402
    format_history_bundle_txt,
    format_result_summary,
    summarize_history_bundle,
    summarize_result,
    write_history_bundle_txt,
)
from polyfempy.api.result import HistoryView, Result  # noqa: E402


def _make_frame(step: int, n_sampled: int = 4, dim: int = 2):
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


def _make_result() -> Result:
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    cells = np.array([[0, 1, 2]], dtype=np.int32)
    history = HistoryView(
        frames=[_make_frame(0), _make_frame(1)],
        times=[0.0, 0.01],
    )
    result = Result("numpy", vertices, cells, fields={"u": vertices.copy()}, history=history)
    result.set_sampled_field("stress", np.full((4, 4), 11.0))
    result.set_sampled_field("von_mises", np.full((4, 1), 1.5))
    result.set_sampled_field("body_ids", np.array([1, 1, 2, 2], dtype=np.int32))
    result.meta["stress_source"] = "solver.solution_frames"
    result.meta["von_mises_source"] = "solver.solution_frames"
    return result


class ResultReportTests(unittest.TestCase):
    def test_summarize_result_collects_field_and_history_info(self):
        result = _make_result()
        summary = summarize_result(result)

        self.assertEqual([field["name"] for field in summary["fields"]], ["u", "stress", "von_mises"])
        self.assertEqual(summary["fields"][0]["shape"], (3, 2))
        self.assertEqual(summary["fields"][1]["note"], "source=solver.solution_frames")

        history = summary["history"]
        self.assertTrue(history["available"])
        self.assertEqual(history["frame_count"], 2)
        self.assertEqual(history["times"], [0.0, 0.01])
        self.assertEqual(history["vm_max_by_body"][1], [0.5, 1.5])
        self.assertEqual(history["final_stress_by_body"][2]["n_points"], 2)

    def test_result_format_summary_returns_human_readable_text(self):
        result = _make_result()
        text = result.format_summary(elapsed=1.23)

        self.assertIn("solve() took 1.23s", text)
        self.assertIn("u         : (3, 2)  (native)", text)
        self.assertIn("stress    : (4, 4)  (source=solver.solution_frames)", text)
        self.assertIn("history: 2 frames", text)
        self.assertIn("history times: [0.000, 0.010]", text)
        self.assertIn("body_id=1: [5.000e-01, 1.500e+00]", text)

    def test_module_helper_matches_result_method(self):
        result = _make_result()
        self.assertEqual(
            format_result_summary(result, elapsed=2.0),
            result.format_summary(elapsed=2.0),
        )

    def test_history_bundle_contains_per_step_and_per_body_rows(self):
        result = _make_result()
        cfg = {
            "geometry": [
                {"mesh": "lattice.msh", "volume_selection": 1},
                {"mesh": "block.msh", "volume_selection": 2},
            ]
        }
        bundle = summarize_history_bundle(result, cfg=cfg)

        self.assertTrue(bundle["available"])
        self.assertEqual(bundle["history_source"], "unknown")
        self.assertEqual(bundle["body_legend"][1]["mesh_stem"], "lattice")
        self.assertEqual(bundle["body_legend"][2]["mesh_stem"], "block")
        self.assertEqual(len(bundle["steps"]), 2)
        self.assertEqual(len(bundle["steps_by_body"]), 4)
        self.assertEqual(bundle["steps"][1]["step"], 1)
        self.assertEqual(bundle["steps_by_body"][0]["body_id"], 1)

    def test_history_bundle_text_and_writer_include_body_legend(self):
        result = _make_result()
        cfg = {
            "geometry": [
                {"mesh": "lattice.msh", "volume_selection": 1},
                {"mesh": "block.msh", "volume_selection": 2},
            ]
        }
        text = format_history_bundle_txt(result, cfg=cfg)

        self.assertIn("# PolyFEM History Bundle", text)
        self.assertIn("[body_legend]", text)
        self.assertIn("body_id\tgeometry_index\tvolume_selection\tmesh_stem", text)
        self.assertIn("lattice", text)
        self.assertIn("[steps_by_body]", text)

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "history_bundle.txt"
            written = write_history_bundle_txt(result, out_path, cfg=cfg)
            self.assertEqual(written, out_path.resolve())
            self.assertEqual(out_path.read_text(encoding="utf-8"), text)


if __name__ == "__main__":
    unittest.main()
