"""Unit tests for converting the standard backend result bundle to Result."""

from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from polyfempy.runtime import _solve_backend as backend_module  # noqa: E402
from polyfempy.runtime import _solve_contract as contract_module  # noqa: E402


def _run_pipeline_from_backend_return(ret):
    solve_module = importlib.import_module("polyfempy.runtime.solve")
    canonical = SimpleNamespace(
        config={},
        full_json=None,
        mesh_source=SimpleNamespace(
            mode="json",
            vertices=None,
            cells=None,
            body_ids=None,
            boundary_ids=None,
            v_backend="numpy",
        ),
        backend_settings={},
    )

    with (
        patch.object(contract_module, "prepare_canonical_solve_input", return_value=canonical),
        patch.object(backend_module, "build_solver", return_value=object()),
        patch.object(backend_module, "configure_solver", return_value=None),
        patch.object(backend_module, "apply_sidesets", return_value=None),
        patch.object(backend_module, "run_solver_stage", return_value=ret),
    ):
        return solve_module.run_pipeline(cfg={})


class BackendBundleTests(unittest.TestCase):
    def test_pipeline_extracts_current_backend_bundle_fields(self):
        ret = {
            "_result_bundle": True,
            "vertices": np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64),
            "cells": np.array([[0, 1, 2]], dtype=np.int64),
            "u": np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]]),
            "p": np.array([0.5, 0.5, 0.5]),
            "stress": np.ones((3, 3)),
            "energy": 2.5,
            "meta": {"from_bundle": True},
        }

        result = _run_pipeline_from_backend_return(ret)

        np.testing.assert_array_equal(result.vertices, ret["vertices"])
        np.testing.assert_array_equal(result.cells, ret["cells"].astype(np.int32))
        self.assertEqual(result.cells.dtype, np.int32)
        np.testing.assert_array_equal(result.u, ret["u"])
        np.testing.assert_array_equal(result.p, ret["p"])
        self.assertEqual(result.meta, {"from_bundle": True})
        self.assertFalse(hasattr(result, "fields"))

    def test_pipeline_drops_empty_pressure(self):
        ret = {
            "_result_bundle": True,
            "vertices": np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64),
            "cells": np.array([[0, 1, 2]], dtype=np.int32),
            "u": np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]]),
            "p": np.array([]),
        }

        result = _run_pipeline_from_backend_return(ret)

        np.testing.assert_array_equal(result.u, ret["u"])
        self.assertIsNone(result.p)

    def test_pipeline_rejects_non_bundle_backend_result(self):
        with self.assertRaisesRegex(RuntimeError, "standard result bundle"):
            _run_pipeline_from_backend_return((np.array([1.0]),))

    def test_pipeline_rejects_bundle_missing_required_keys(self):
        ret = {
            "_result_bundle": True,
            "vertices": np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64),
            "u": np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]]),
        }

        with self.assertRaisesRegex(RuntimeError, "missing required keys: cells"):
            _run_pipeline_from_backend_return(ret)


if __name__ == "__main__":
    unittest.main()
