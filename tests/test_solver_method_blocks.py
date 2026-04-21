"""Regression tests: solver.nonlinear method-specific blocks must round-trip.

PolyFEM's JSON schema lets users tune the active nonlinear solver via a
sub-dict keyed by the solver name, e.g.::

    {
      "solver": {
        "nonlinear": {
          "solver": "Newton",
          "Newton": {"residual_tolerance": 100}
        }
      }
    }

Before this fix, ``Solver.from_dict`` silently dropped every such block
(``Newton`` / ``ADAM`` / ``L-BFGS`` / ``L-BFGS-B`` / ``StochasticADAM`` /
``StochasticGradientDescent``) to sidestep JSON-schema validation errors.
The side effect was that ``cfg.to_dict()`` / ``solve()`` emitted a nonlinear
block without the user's per-method overrides, so the C++ solver used its
default ``residual_tolerance=1e-7`` no matter what the user had asked for.

These tests pin down the preserved round-trip.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
CONFIG_PATH = _REPO / "polyfempy" / "api" / "config.py"

SPEC = importlib.util.spec_from_file_location(
    "polyfempy_api_config_for_solver_tests", CONFIG_PATH
)
CONFIG_MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(CONFIG_MODULE)
SimulationConfig = CONFIG_MODULE.SimulationConfig
Solver = CONFIG_MODULE.Solver
NonlinearSolver = CONFIG_MODULE.NonlinearSolver

if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _cfg_with_solver(solver_block):
    return SimulationConfig.from_json_dict(
        {
            "pde": "LinearElasticity",
            "discr_order": 1,
            "materials": [{"type": "LinearElasticity", "E": 20, "nu": 0.3}],
            "boundary_conditions": {},
            "geometry": [{"mesh": "beam.msh"}],
            "solver": solver_block,
        }
    )


class NewtonBlockRoundTripTests(unittest.TestCase):
    """The specific regression: Newton.residual_tolerance must survive."""

    def test_newton_residual_tolerance_survives_from_dict(self):
        s = Solver.from_dict(
            {
                "nonlinear": {
                    "solver": "Newton",
                    "Newton": {"residual_tolerance": 100},
                }
            }
        )
        self.assertIsNotNone(s.nonlinear)
        self.assertIsNotNone(s.nonlinear.method_blocks)
        self.assertEqual(
            s.nonlinear.method_blocks["Newton"]["residual_tolerance"], 100
        )

    def test_newton_block_is_reemitted_in_to_dict(self):
        s = Solver.from_dict(
            {
                "nonlinear": {
                    "solver": "Newton",
                    "Newton": {"residual_tolerance": 100},
                }
            }
        )
        d = s.to_dict()
        self.assertIn("Newton", d["nonlinear"])
        self.assertEqual(d["nonlinear"]["Newton"]["residual_tolerance"], 100)

    def test_other_method_blocks_round_trip_too(self):
        """ADAM / L-BFGS / L-BFGS-B / StochasticADAM /
        StochasticGradientDescent must all be preserved, not only Newton."""
        d_in = {
            "nonlinear": {
                "solver": "Newton",
                "ADAM": {"alpha": 0.01},
                "L-BFGS": {"history_size": 6},
                "L-BFGS-B": {"history_size": 6},
                "StochasticADAM": {"alpha": 0.1},
                "StochasticGradientDescent": {"alpha": 0.1},
            }
        }
        s = Solver.from_dict(d_in)
        out = s.to_dict()
        for key in (
            "ADAM",
            "L-BFGS",
            "L-BFGS-B",
            "StochasticADAM",
            "StochasticGradientDescent",
        ):
            with self.subTest(block=key):
                self.assertIn(key, out["nonlinear"])
                self.assertEqual(out["nonlinear"][key], d_in["nonlinear"][key])


class FullConfigRoundTripTests(unittest.TestCase):
    """End-to-end: the same JSON a user would call ``solve()`` with must
    produce a ``to_dict()`` that still has the method block, regardless of
    whether ``_full_json_config`` or the synthesized solver dataclass wins the
    overlay."""

    def test_newton_residual_tolerance_survives_simulation_config_to_dict(self):
        cfg = _cfg_with_solver(
            {
                "linear": {"solver": "Eigen::PardisoLDLT"},
                "nonlinear": {
                    "solver": "Newton",
                    "grad_norm": 0.002,
                    "max_iterations": 800,
                    "Newton": {"residual_tolerance": 100},
                },
                "contact": {"barrier_stiffness": "adaptive"},
            }
        )
        full = cfg.to_dict()
        self.assertIn("Newton", full["solver"]["nonlinear"])
        self.assertEqual(
            full["solver"]["nonlinear"]["Newton"]["residual_tolerance"], 100
        )
        # The non-method keys must also survive.
        self.assertEqual(full["solver"]["nonlinear"]["grad_norm"], 0.002)
        self.assertEqual(full["solver"]["nonlinear"]["max_iterations"], 800)


class NonlinearSolverDirectConstructTests(unittest.TestCase):
    """Users who construct NonlinearSolver directly in Python must be able
    to set method blocks without going through JSON."""

    def test_direct_construction_with_method_blocks(self):
        nl = NonlinearSolver(
            solver_type="Newton",
            method_blocks={"Newton": {"residual_tolerance": 50}},
        )
        out = nl.to_dict()
        self.assertEqual(out["solver"], "Newton")
        self.assertEqual(out["Newton"]["residual_tolerance"], 50)

    def test_default_construction_emits_no_method_blocks(self):
        """Users who don't set method_blocks must see a clean to_dict, no
        stray empty dict that would confuse the schema."""
        nl = NonlinearSolver(solver_type="newton")
        out = nl.to_dict()
        for key in (
            "ADAM",
            "L-BFGS",
            "L-BFGS-B",
            "Newton",
            "StochasticADAM",
            "StochasticGradientDescent",
        ):
            self.assertNotIn(key, out)


if __name__ == "__main__":
    unittest.main()
