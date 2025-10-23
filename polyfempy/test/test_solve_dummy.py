# test/test_solve_dummy.py
# -----------------------------------------------------------------------------
# Goal: verify that in Dummy mode, solve() will immediately refuse to run
# a real solve.
# Details:
#   - By setting POLYFEMPY_ALLOW_DUMMY_SETTINGS=1, make cfg.to_settings()
#     return a DummySettings object.
#   - solve() should raise RuntimeError with a message containing "DummySettings".
# -----------------------------------------------------------------------------

import numpy as np
import pytest

from polyfempy.api import SimulationConfig
from polyfempy.api.solve import solve


class _DummySettings:
    def __init__(self):
        self._is_dummy = True


def _force_dummy(monkeypatch):
    # The env var can be used by config.py later; the key is to force
    # to_settings() to return DummySettings for this test.
    monkeypatch.setenv("POLYFEMPY_ALLOW_DUMMY_SETTINGS", "1")
    monkeypatch.setattr(SimulationConfig, "to_settings", lambda self: _DummySettings(), raising=True)


def test_solve_rejects_in_dummy(monkeypatch):
    _force_dummy(monkeypatch)

    V = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    C = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

    cfg = SimulationConfig.linear_elasticity(E=1000.0, nu=0.3, order=1)
    cfg.boundary_conditions = {
        "dirichlet_boundary": [{"id": 4, "value": [0.0, 0.0]}],
        "rhs": [0.0, -0.1],
    }

    with pytest.raises(RuntimeError, match="DummySettings"):
        _ = solve(V, C, cfg)
