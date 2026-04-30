"""Unit tests for the native-result extraction stage of `_solve_pipeline`.

Covers `extract_native_outputs` across the four strategies it dispatches between:
    1. ``ret`` is a ``_result_bundle`` dict        -> _outputs_from_bundle
    2. ``ret`` is a (sol, pressure) tuple/list     -> _outputs_from_tuple
    3. ``solver.get_sampled_solution()`` available -> _outputs_from_sampled_getter
    4. ``solver.get_solution()`` / friends         -> _outputs_from_direct_getters
    5. none of the above                           -> RuntimeError

Additional checks:
    - Bundle: pressure of size 0 is dropped; energy scalar lands in meta.
    - Tuple:  additional solver getters (stress/strain/energy/...) feed fields/meta.
    - Failure: bare FakeSolver with none of the getters raises a RuntimeError.

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
    NativeOutputs,
    NormalizedInputs,
    extract_native_outputs,
)


def _array_mode_inputs() -> NormalizedInputs:
    V = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    C = np.array([[0, 1, 2]], dtype=np.int32)
    return NormalizedInputs(
        V_np=V,
        C_np=C,
        body_ids_np=None,
        boundary_ids_np=None,
        v_backend="numpy",
        use_json_mode=False,
    )


def _json_mode_inputs() -> NormalizedInputs:
    # JSON mode: no vertices/cells supplied by the caller. Extraction must
    # rely entirely on what the solver can report back.
    return NormalizedInputs(
        V_np=None,
        C_np=None,
        body_ids_np=None,
        boundary_ids_np=None,
        v_backend="numpy",
        use_json_mode=True,
    )


class BundleStrategyTests(unittest.TestCase):
    def test_bundle_extracts_mesh_and_fields_directly(self):
        ret = {
            "_result_bundle": True,
            "vertices": np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64),
            "cells": np.array([[0, 1, 2]], dtype=np.int32),
            "u": np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]]),
            "p": np.array([0.5, 0.5, 0.5]),
            "stress": np.tile(np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]), (3, 1)),
            "energy": 2.5,
            "meta": {"iters": 7},
        }

        class FakeSolver:
            pass

        native = extract_native_outputs(ret, FakeSolver(), _array_mode_inputs())

        self.assertIsInstance(native, NativeOutputs)
        self.assertEqual(native.vertices.shape, (3, 2))
        self.assertEqual(native.cells.shape, (1, 3))
        self.assertEqual(native.cells.dtype, np.int32)
        self.assertIn("u", native.fields)
        self.assertIn("p", native.fields)
        self.assertIn("stress", native.fields)
        self.assertEqual(native.meta["energy"], 2.5)
        self.assertEqual(native.meta["iters"], 7)
        self.assertEqual(native.meta["solver_type"], "FakeSolver")

    def test_bundle_drops_empty_pressure_but_keeps_displacement(self):
        ret = {
            "_result_bundle": True,
            "vertices": np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64),
            "cells": np.array([[0, 1, 2]], dtype=np.int32),
            "u": np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]]),
            "p": np.array([]),  # size 0 -> must not land in fields
        }

        class FakeSolver:
            pass

        native = extract_native_outputs(ret, FakeSolver(), _array_mode_inputs())
        self.assertIn("u", native.fields)
        self.assertNotIn("p", native.fields)

    def test_bundle_with_array_energy_lands_in_fields_not_meta(self):
        ret = {
            "_result_bundle": True,
            "vertices": np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64),
            "cells": np.array([[0, 1, 2]], dtype=np.int32),
            "u": np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]]),
            "energy": np.array([1.0, 2.0, 3.0]),
        }

        class FakeSolver:
            pass

        native = extract_native_outputs(ret, FakeSolver(), _array_mode_inputs())
        self.assertIn("energy", native.fields)
        self.assertNotIn("energy", native.meta)


class TupleStrategyTests(unittest.TestCase):
    def test_tuple_of_length_one_produces_u_only(self):
        u = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]])

        class FakeSolver:
            def get_vertices(self):
                return np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64)

            def get_elements(self):
                return np.array([[0, 1, 2]], dtype=np.int32)

        native = extract_native_outputs((u,), FakeSolver(), _array_mode_inputs())
        self.assertIn("u", native.fields)
        self.assertNotIn("p", native.fields)
        np.testing.assert_array_equal(native.fields["u"], u)

    def test_tuple_of_length_two_produces_u_and_p(self):
        u = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]])
        p = np.array([0.1, 0.2, 0.3])

        class FakeSolver:
            def get_vertices(self):
                return np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64)

            def get_elements(self):
                return np.array([[0, 1, 2]], dtype=np.int32)

        native = extract_native_outputs([u, p], FakeSolver(), _array_mode_inputs())
        np.testing.assert_array_equal(native.fields["u"], u)
        np.testing.assert_array_equal(native.fields["p"], p)

    def test_tuple_falls_back_to_input_mesh_when_solver_has_no_getters(self):
        u = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]])

        class FakeSolver:  # no mesh getters
            pass

        inputs = _array_mode_inputs()
        native = extract_native_outputs((u,), FakeSolver(), inputs)
        # Falls back to the vertices/cells supplied by the caller.
        np.testing.assert_array_equal(native.vertices, inputs.V_np)
        np.testing.assert_array_equal(native.cells, inputs.C_np)

    def test_tuple_picks_up_additional_fields_from_solver_getters(self):
        u = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]])
        stress = np.tile(np.array([[1.0, 2.0, 3.0]]), (3, 1))

        class FakeSolver:
            def get_vertices(self):
                return np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64)

            def get_elements(self):
                return np.array([[0, 1, 2]], dtype=np.int32)

            def get_stress(self):
                return stress

            def get_energy(self):
                return 3.14

        native = extract_native_outputs((u,), FakeSolver(), _array_mode_inputs())
        self.assertIn("stress", native.fields)
        np.testing.assert_array_equal(native.fields["stress"], stress)
        self.assertAlmostEqual(native.meta["energy"], 3.14)


class SampledGetterStrategyTests(unittest.TestCase):
    def test_sampled_getter_prefers_sampled_mesh_and_displacement(self):
        sampled_V = np.array(
            [[0, 0], [1, 0], [0, 1], [0.5, 0.5]], dtype=np.float64
        )
        sampled_u = np.array(
            [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.05, 0.05]]
        )

        class FakeSolver:
            def get_sampled_solution(self):
                # (vertices, ?, ?, ?, solution) -- only indices 0 and 4 are read.
                return (sampled_V, None, None, None, sampled_u)

            def get_elements(self):
                return np.array([[0, 1, 2]], dtype=np.int32)

        native = extract_native_outputs(None, FakeSolver(), _json_mode_inputs())
        np.testing.assert_array_equal(native.vertices, sampled_V)
        np.testing.assert_array_equal(native.fields["u"], sampled_u)

    def test_sampled_getter_ignored_when_result_is_short_tuple(self):
        # If get_sampled_solution returns fewer than 5 elements, the strategy
        # must fall through to the direct-getter branch.
        class FakeSolver:
            def get_sampled_solution(self):
                return (np.array([[0, 0]]), None)  # too short

            def get_solution(self):
                return np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]])

            def get_vertices(self):
                return np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64)

            def get_elements(self):
                return np.array([[0, 1, 2]], dtype=np.int32)

        native = extract_native_outputs(None, FakeSolver(), _array_mode_inputs())
        self.assertEqual(native.fields["u"].shape, (3, 2))
        self.assertEqual(native.vertices.shape, (3, 2))


class DirectGetterStrategyTests(unittest.TestCase):
    def test_direct_getters_used_when_ret_is_none_and_no_sampled(self):
        u = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]])

        class FakeSolver:
            def get_solution(self):
                return u

            def get_vertices(self):
                return np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64)

            def get_elements(self):
                return np.array([[0, 1, 2]], dtype=np.int32)

        native = extract_native_outputs(None, FakeSolver(), _array_mode_inputs())
        np.testing.assert_array_equal(native.fields["u"], u)
        self.assertEqual(native.vertices.shape, (3, 2))
        self.assertEqual(native.cells.shape, (1, 3))

    def test_direct_getter_falls_back_to_input_mesh(self):
        u = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]])

        class FakeSolver:
            def get_u(self):
                return u
            # No vertex/element getters.

        inputs = _array_mode_inputs()
        native = extract_native_outputs(None, FakeSolver(), inputs)
        np.testing.assert_array_equal(native.vertices, inputs.V_np)
        np.testing.assert_array_equal(native.cells, inputs.C_np)

    def test_raises_when_no_extraction_path_is_available(self):
        class FakeSolver:
            pass

        with self.assertRaises(RuntimeError):
            extract_native_outputs(None, FakeSolver(), _json_mode_inputs())


if __name__ == "__main__":
    unittest.main()
