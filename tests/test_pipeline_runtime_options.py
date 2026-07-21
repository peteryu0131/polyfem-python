"""Unit tests for runtime-option resolution from backend-shaped payloads."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from polyfempy.runtime._solve_pipeline import RuntimeOptions, resolve_runtime_options  # noqa: E402


def _make_cfg(output: dict | None = None) -> dict:
    cfg = {
        "pde": "LinearElasticity",
        "discr_order": 1,
        "materials": [{"type": "LinearElasticity", "E": 20, "nu": 0.3}],
        "boundary_conditions": {},
        "geometry": [{"mesh": "beam.msh"}],
    }
    if output is not None:
        cfg["output"] = output
    return cfg


class ResolveRuntimeOptionsTests(unittest.TestCase):
    def test_defaults_when_nothing_is_configured(self):
        opts = resolve_runtime_options(_make_cfg(output={"directory": "out"}), None, None)
        self.assertIsInstance(opts, RuntimeOptions)
        self.assertIsNone(opts.requested_fields)
        self.assertFalse(opts.strict)
        self.assertEqual(opts.fallback_mode, "never")
        self.assertEqual(opts.temp_storage, "ram")
        self.assertFalse(opts.keep_temp_files)

    def test_reads_from_cfg_output_dict(self):
        cfg = _make_cfg(
            output={
                "directory": "out",
                "result": {"fields": ["u", "stress"], "strict": True},
                "fallback": {
                    "sampled_vtu": "auto",
                    "temp_storage": "disk",
                    "keep_temp_files": True,
                },
            }
        )
        opts = resolve_runtime_options(cfg, None, None)
        self.assertEqual(opts.requested_fields, ["u", "stress"])
        self.assertTrue(opts.strict)
        self.assertEqual(opts.fallback_mode, "auto")
        self.assertEqual(opts.temp_storage, "disk")
        self.assertTrue(opts.keep_temp_files)

    def test_falls_back_to_full_json_when_cfg_output_empty(self):
        cfg = _make_cfg(output={"directory": "out"})
        full_json = {
            "output": {
                "result": {"fields": ["von_mises"], "strict": False},
                "fallback": {"sampled_vtu": "always"},
            }
        }
        opts = resolve_runtime_options(cfg, full_json, None)
        self.assertEqual(opts.requested_fields, ["von_mises"])
        self.assertEqual(opts.fallback_mode, "always")

    def test_sampled_vtu_fallback_true_forces_always(self):
        cfg = _make_cfg(output={"directory": "out", "fallback": {"sampled_vtu": "never"}})
        opts = resolve_runtime_options(cfg, None, True)
        self.assertEqual(opts.fallback_mode, "always")

    def test_sampled_vtu_fallback_false_forces_never(self):
        cfg = _make_cfg(output={"directory": "out", "fallback": {"sampled_vtu": "always"}})
        opts = resolve_runtime_options(cfg, None, False)
        self.assertEqual(opts.fallback_mode, "never")

    def test_sampled_vtu_fallback_none_leaves_mode_untouched(self):
        cfg = _make_cfg(output={"directory": "out", "fallback": {"sampled_vtu": "auto"}})
        opts = resolve_runtime_options(cfg, None, None)
        self.assertEqual(opts.fallback_mode, "auto")

    def test_requested_fields_coerced_to_strings(self):
        cfg = _make_cfg(output={"directory": "out", "result": {"fields": ["u", 123]}})
        opts = resolve_runtime_options(cfg, None, None)
        self.assertEqual(opts.requested_fields, ["u", "123"])


if __name__ == "__main__":
    unittest.main()
