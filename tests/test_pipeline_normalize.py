"""Unit tests for the cfg / mesh normalization stages of `_solve_pipeline`.

Covers:
    - normalize_cfg:               dict / str path / SimulationConfig / None / bad types
    - build_full_json:             extras['_full_json_config'] path + cfg.to_dict() path
    - merge_user_cfg_over_full_json: Python-side overrides win over original full JSON
    - normalize_mesh_inputs:       array mode / JSON mode / neither

These tests never touch the C++ backend. They only exercise pure Python stages.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from polyfempy.api._solve_pipeline import (  # noqa: E402
    NormalizedInputs,
    build_full_json,
    merge_user_cfg_over_full_json,
    normalize_cfg,
    normalize_mesh_inputs,
)
from polyfempy.api.config import SimulationConfig  # noqa: E402


_MINIMAL_FULL = {
    "pde": "LinearElasticity",
    "discr_order": 1,
    "materials": [{"type": "LinearElasticity", "E": 20, "nu": 0.3}],
    "boundary_conditions": {},
    "geometry": [{"mesh": "beam.msh"}],
}


class NormalizeCfgTests(unittest.TestCase):
    def test_accepts_plain_dict(self):
        cfg = normalize_cfg(dict(_MINIMAL_FULL))
        self.assertIsInstance(cfg, SimulationConfig)
        self.assertEqual(cfg.pde, "LinearElasticity")

    def test_accepts_simulationconfig_passthrough(self):
        original = SimulationConfig.from_json_dict(_MINIMAL_FULL)
        returned = normalize_cfg(original)
        self.assertIs(returned, original)

    def test_accepts_json_file_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(_MINIMAL_FULL), encoding="utf-8")
            cfg = normalize_cfg(str(path))
        self.assertIsInstance(cfg, SimulationConfig)

    def test_none_raises_value_error(self):
        with self.assertRaises(ValueError):
            normalize_cfg(None)

    def test_bad_type_raises_type_error(self):
        for bad in (123, 1.5, ["a"], object()):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(TypeError):
                    normalize_cfg(bad)


class BuildFullJsonTests(unittest.TestCase):
    def test_returns_merged_full_json_when_loaded_from_json(self):
        cfg = SimulationConfig.from_json_dict(_MINIMAL_FULL)
        self.assertIn("_full_json_config", cfg.extras)
        fj = build_full_json(cfg)
        self.assertIsInstance(fj, dict)
        self.assertEqual(fj["pde"], "LinearElasticity")
        self.assertEqual(fj["geometry"], [{"mesh": "beam.msh"}])

    def test_python_override_wins_over_original_full_json(self):
        cfg = SimulationConfig.from_json_dict(_MINIMAL_FULL)
        cfg.discr_order = 2  # Python-side override after loading from JSON
        fj = build_full_json(cfg)
        self.assertEqual(fj["discr_order"], 2)

    def test_returns_none_when_no_geometry_anywhere(self):
        cfg = SimulationConfig.linear_elasticity(2100, 0.3)
        # No geometry in extras or to_dict() → JSON mode is infeasible.
        fj = build_full_json(cfg)
        self.assertIsNone(fj)

    def test_returns_none_when_cfg_to_dict_raises(self):
        class BrokenCfg:
            extras: dict = {}

            def to_dict(self):
                raise RuntimeError("boom")

        self.assertIsNone(build_full_json(BrokenCfg()))


class MergeUserCfgOverFullJsonTests(unittest.TestCase):
    def test_python_side_keys_overlay_original_full_json(self):
        full_json = dict(_MINIMAL_FULL)
        cfg = SimulationConfig.from_json_dict(full_json)
        cfg.discr_order = 3
        merged = merge_user_cfg_over_full_json(cfg, full_json)
        self.assertEqual(merged["discr_order"], 3)
        self.assertEqual(merged["geometry"], full_json["geometry"])

    def test_root_path_preserved_from_extras_when_present(self):
        full_json = dict(_MINIMAL_FULL)
        full_json["root_path"] = "/tmp/legacy"
        cfg = SimulationConfig.from_json_dict(full_json)
        cfg.extras["_root_path"] = "/tmp/new"
        merged = merge_user_cfg_over_full_json(cfg, full_json)
        self.assertEqual(merged["root_path"], "/tmp/new")

    def test_root_path_falls_back_to_full_json_when_extras_absent(self):
        full_json = dict(_MINIMAL_FULL)
        full_json["root_path"] = "/tmp/legacy"
        cfg = SimulationConfig.from_json_dict(full_json)
        cfg.extras.pop("_root_path", None)
        merged = merge_user_cfg_over_full_json(cfg, full_json)
        self.assertEqual(merged["root_path"], "/tmp/legacy")

    def test_returns_full_json_when_cfg_to_dict_fails(self):
        class BrokenCfg:
            def to_dict(self):
                raise RuntimeError("boom")

        full_json = {"a": 1}
        self.assertEqual(
            merge_user_cfg_over_full_json(BrokenCfg(), full_json),
            full_json,
        )


class NormalizeMeshInputsTests(unittest.TestCase):
    def test_array_mode_returns_numpy_arrays(self):
        V = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        C = np.array([[0, 1, 2]], dtype=np.int64)  # not int32 on purpose
        inputs = normalize_mesh_inputs(V, C, None, dtype=None)
        self.assertIsInstance(inputs, NormalizedInputs)
        self.assertFalse(inputs.use_json_mode)
        self.assertEqual(inputs.V_np.shape, (3, 2))
        self.assertEqual(inputs.C_np.dtype, np.int32)  # cast to int32
        self.assertEqual(inputs.v_backend, "numpy")

    def test_json_mode_when_full_json_has_geometry_and_no_arrays(self):
        full_json = {"geometry": [{"mesh": "beam.msh"}]}
        inputs = normalize_mesh_inputs(None, None, full_json, dtype=None)
        self.assertTrue(inputs.use_json_mode)
        self.assertIsNone(inputs.V_np)
        self.assertIsNone(inputs.C_np)

    def test_neither_arrays_nor_geometry_raises(self):
        with self.assertRaises(ValueError):
            normalize_mesh_inputs(None, None, None, dtype=None)
        with self.assertRaises(ValueError):
            normalize_mesh_inputs(None, None, {"pde": "LinearElasticity"}, dtype=None)

    def test_arrays_override_json_mode_when_both_provided(self):
        V = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        C = np.array([[0, 1, 2]], dtype=np.int32)
        full_json = {"geometry": [{"mesh": "beam.msh"}]}
        inputs = normalize_mesh_inputs(V, C, full_json, dtype=None)
        # Arrays are present, so the pipeline prefers array mode.
        self.assertFalse(inputs.use_json_mode)
        self.assertIsNotNone(inputs.V_np)


if __name__ == "__main__":
    unittest.main()
