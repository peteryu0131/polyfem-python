"""Unit tests for runtime-option resolution.

Covers:
    - resolve_runtime_options: cfg.output / full_json.output / sampled_vtu_fallback override

These tests never touch the C++ backend. They only exercise pure Python stages.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from polyfempy.api._solve_pipeline import (  # noqa: E402
    RuntimeOptions,
    resolve_runtime_options,
)
from polyfempy.api.config import Output, ParaviewOutput, SimulationConfig  # noqa: E402


def _make_cfg(output: dict | None = None) -> SimulationConfig:
    d = {
        "pde": "LinearElasticity",
        "discr_order": 1,
        "materials": [{"type": "LinearElasticity", "E": 20, "nu": 0.3}],
        "boundary_conditions": {},
        "geometry": [{"mesh": "beam.msh"}],
    }
    if output is not None:
        d["output"] = output
    return SimulationConfig.from_json_dict(d)


class ResolveRuntimeOptionsTests(unittest.TestCase):
    def test_defaults_when_nothing_is_configured(self):
        cfg = _make_cfg(output={"directory": "out"})
        opts = resolve_runtime_options(cfg, None, None)
        self.assertIsInstance(opts, RuntimeOptions)
        self.assertIsNone(opts.requested_fields)
        self.assertFalse(opts.strict)
        self.assertEqual(opts.fallback_mode, "never")
        self.assertEqual(opts.temp_storage, "ram")
        self.assertFalse(opts.keep_temp_files)

    def test_reads_from_cfg_output(self):
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
        cfg = _make_cfg(output={"directory": "out"})  # no result/fallback on cfg
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
        cfg = _make_cfg(
            output={
                "directory": "out",
                "fallback": {"sampled_vtu": "never"},
            }
        )
        opts = resolve_runtime_options(cfg, None, True)
        self.assertEqual(opts.fallback_mode, "always")

    def test_sampled_vtu_fallback_false_forces_never(self):
        cfg = _make_cfg(
            output={
                "directory": "out",
                "fallback": {"sampled_vtu": "always"},
            }
        )
        opts = resolve_runtime_options(cfg, None, False)
        self.assertEqual(opts.fallback_mode, "never")

    def test_sampled_vtu_fallback_none_leaves_mode_untouched(self):
        cfg = _make_cfg(
            output={
                "directory": "out",
                "fallback": {"sampled_vtu": "auto"},
            }
        )
        opts = resolve_runtime_options(cfg, None, None)
        self.assertEqual(opts.fallback_mode, "auto")

    def test_requested_fields_coerced_to_strings(self):
        cfg = _make_cfg(
            output={
                "directory": "out",
                "result": {"fields": ["u", 123]},  # stray non-string
            }
        )
        opts = resolve_runtime_options(cfg, None, None)
        self.assertEqual(opts.requested_fields, ["u", "123"])


class OutputVtuSwitchTests(unittest.TestCase):
    def test_resolve_relative_paths_rewrites_output_targets(self):
        output = Output(
            directory="out",
            json="impact_stats.json",
            log={"path": "polyfem.log"},
            paraview=ParaviewOutput(file_name="impact.pvd"),
        )
        output.resolve_relative_paths("/tmp/polyfem-run")
        self.assertEqual(output.json, "/tmp/polyfem-run/impact_stats.json")
        self.assertEqual(output.log["path"], "/tmp/polyfem-run/polyfem.log")
        self.assertEqual(output.paraview.file_name, "/tmp/polyfem-run/impact.pvd")

    def test_save_vtu_false_clears_file_name_but_preserves_time_sequence(self):
        output = Output(
            directory="out",
            paraview=ParaviewOutput(file_name="impact.pvd"),
            advanced={"save_time_sequence": True},
            save_vtu=False,
        )
        d = output.to_dict()
        self.assertIn("paraview", d)
        self.assertEqual(d["paraview"].get("file_name"), "")
        self.assertIn("advanced", d)
        self.assertTrue(d["advanced"].get("save_time_sequence"))

    def test_save_paraview_false_still_disables_time_sequence(self):
        output = Output(
            directory="out",
            paraview=ParaviewOutput(file_name="impact.pvd"),
            advanced={"save_time_sequence": True},
            save_paraview=False,
        )
        d = output.to_dict()
        self.assertEqual(d["paraview"].get("file_name"), "")
        self.assertFalse(d["advanced"].get("save_time_sequence"))


if __name__ == "__main__":
    unittest.main()
