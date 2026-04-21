"""Unit tests for runtime-option resolution and fallback decision logic.

Covers:
    - resolve_runtime_options: cfg.output / full_json.output / sampled_vtu_fallback override
    - _should_run_fallback:     decision matrix for never / auto / always + requested fields

These tests never touch the C++ backend. They only exercise pure Python stages.
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
    RuntimeOptions,
    _should_run_fallback,
    resolve_runtime_options,
)
from polyfempy.api.config import SimulationConfig  # noqa: E402
from polyfempy.api.result import Result  # noqa: E402


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


def _empty_result() -> Result:
    V = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    C = np.array([[0, 1, 2]], dtype=np.int32)
    return Result("numpy", V, C)


def _result_with_stress() -> Result:
    r = _empty_result()
    # 3 vertices × 3 Voigt components in 2D → von_mises will be computable.
    r.set_field("stress", np.array([[1.0, 0.5, 0.2]] * 3))
    return r


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


class ShouldRunFallbackTests(unittest.TestCase):
    def _opts(self, **kwargs) -> RuntimeOptions:
        return RuntimeOptions(**kwargs)

    def test_never_mode_returns_false(self):
        self.assertFalse(
            _should_run_fallback(_empty_result(), self._opts(fallback_mode="never"))
        )

    def test_always_mode_returns_true_regardless_of_fields(self):
        self.assertTrue(
            _should_run_fallback(_empty_result(), self._opts(fallback_mode="always"))
        )
        self.assertTrue(
            _should_run_fallback(
                _result_with_stress(),
                self._opts(fallback_mode="always", requested_fields=["stress"]),
            )
        )

    def test_auto_mode_without_requested_fields_returns_false(self):
        self.assertFalse(
            _should_run_fallback(_empty_result(), self._opts(fallback_mode="auto"))
        )

    def test_auto_mode_with_non_sampled_fields_returns_false(self):
        # Only "u" is requested; "u" is never a sampled-candidate.
        self.assertFalse(
            _should_run_fallback(
                _empty_result(),
                self._opts(fallback_mode="auto", requested_fields=["u"]),
            )
        )

    def test_auto_mode_triggers_when_sampled_field_is_missing(self):
        self.assertTrue(
            _should_run_fallback(
                _empty_result(),
                self._opts(fallback_mode="auto", requested_fields=["stress"]),
            )
        )

    def test_auto_mode_skips_when_all_sampled_fields_available(self):
        r = _result_with_stress()  # stress → von_mises computable
        self.assertFalse(
            _should_run_fallback(
                r,
                self._opts(
                    fallback_mode="auto", requested_fields=["stress", "von_mises"]
                ),
            )
        )

    def test_unknown_mode_behaves_as_never(self):
        self.assertFalse(
            _should_run_fallback(
                _empty_result(),
                self._opts(fallback_mode="bogus", requested_fields=["stress"]),
            )
        )


if __name__ == "__main__":
    unittest.main()
