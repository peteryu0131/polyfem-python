# test/test_batch.py
# -----------------------------------------------------------------------------
# Purpose
# -------
# 1) Verify that in Dummy mode, batch_solve() fails immediately, propagating
#    the same guard error that inner solve() raises.
# 2) Without requiring C++ bindings, use a monkeypatched fake_solve() to check
#    the two key batch contracts:
#      * order preservation (outputs match input order)
#      * kwargs forwarding (per-job kwargs are passed through to solve()).
#
# Design
# ------
# - Scenario 1 (failure path):
#   Force SimulationConfig.to_settings() to return an object with _is_dummy=True.
#   This guarantees that solve() raises RuntimeError (including "DummySettings")
#   before any numerical solve, and batch_solve() must not swallow that error.
#
# - Scenario 2 (success path contract):
#   Monkeypatch the *symbol that batch_solve uses internally* (batch_mod.solve)
#   with a fake_solve(V, C, cfg, **kwargs) that:
#     - normalizes inputs (as real solve would: as_numpy + cells→int32),
#     - fabricates a placeholder Result,
#     - records call order + received kwargs into Result.meta.
#   This lets us assert order preservation and kwargs forwarding without the
#   compiled polyfempy bindings present.
#
# Notes
# -----
# - No dependency on the C++ bindings (.pyd); suitable for CI.
# - If batch_solve becomes concurrent later, this test still holds as long as
#   the external contract (order preservation / kwargs forwarding) remains the
#   same. If the contract changes, update the assertions accordingly.
# -----------------------------------------------------------------------------

import numpy as np
import pytest

from polyfempy.api import SimulationConfig, Result
from polyfempy.api import batch_solve
import polyfempy.api.batch as batch_mod  # Important: we patch the symbol that batch_solve calls

# ---- Dummy sentinel ----------------------------------------------------------
class _DummySettings:
    """Minimal DummySettings: only the _is_dummy flag to trigger solve()'s guard."""
    _is_dummy = True


def _force_dummy(monkeypatch):
    """
    Force cfg.to_settings() to return DummySettings so the Dummy branch is hit
    deterministically. Also set the env var for future compatibility with
    config.py if it reads it.
    """
    monkeypatch.setenv("POLYFEMPY_ALLOW_DUMMY_SETTINGS", "1")
    # Replace SimulationConfig.to_settings with a function returning DummySettings
    monkeypatch.setattr(SimulationConfig, "to_settings", lambda self: _DummySettings(), raising=True)


# -------------------- Scenario 1: batch_solve should raise in Dummy mode --------------------
def test_batch_raises_in_dummy(monkeypatch):
    """
    Expected behavior:
    - The first inner solve() call raises RuntimeError due to the Dummy guard.
    - batch_solve(jobs) must not swallow the exception; it should propagate up
      with a message containing "DummySettings".
    """
    _force_dummy(monkeypatch)

    # Simple 2D mesh: unit square split into two triangles
    V = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    C = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    cfg = SimulationConfig.linear_elasticity(E=1000.0, nu=0.3, order=1)

    # Two identical jobs; the first already triggers the error
    jobs = [(V, C, cfg), (V, C, cfg)]

    # Assert the error message contains "DummySettings"
    with pytest.raises(RuntimeError, match="DummySettings"):
        _ = batch_solve(jobs)


# -------- Scenario 2: no C++ bindings; mock solve to verify order & kwargs forwarding --------
def test_batch_order_and_kwargs_via_monkeypatch(monkeypatch):
    """
    Replace the batch module's internal solve symbol with fake_solve:
    - Each call records its index (idx = 0,1,2,...).
    - Inputs are normalized (as real solve would do): as_numpy + int32 cells.
    - Returns a minimal Result: fields['u'] filled with idx; meta stores idx and
      the list of kwargs keys it received.
    Assertions:
    - Output length matches input jobs.
    - meta.idx equals [0,1,2] strictly => order-preserving.
    - kwargs are correctly forwarded to each solve call.
    """
    calls = []  # record fake_solve invocations to double-check counts/order

    def fake_solve(V, C, cfg, **kwargs):
        """
        Lightweight “fake solver”:
        - Records invocation and kwargs.
        - Normalizes inputs to NumPy (ensures cells=int32).
        - Produces a placeholder displacement field u (all entries = idx),
          and stores kwargs keys in meta.
        """
        import numpy as _np
        from polyfempy.api import tensor as T

        # Call index: 0,1,2,...
        idx = len(calls)
        calls.append((V, C, cfg, kwargs))

        # Backend unification: mirror the real solve’s preprocessing
        V_np, backend = T.as_numpy(V)
        C_np, _ = T.as_numpy(C, dtype=_np.int32)
        if C_np.dtype != _np.int32:
            C_np = C_np.astype(_np.int32, copy=False)

        # Placeholder displacement: shape (N, dim), filled with idx
        dim = V_np.shape[1] if V_np.ndim == 2 else 1
        u = _np.full((V_np.shape[0], dim), float(idx), dtype=float)

        return Result(
            backend=backend,
            vertices=V_np,
            cells=C_np,
            fields={"u": u},
            meta={
                "idx": idx,                            # for order-preservation check
                "kwargs": sorted(list(kwargs.keys())), # for kwargs-forwarding check
            },
        )

    # Key point: patch the *batch module’s* reference to solve
    # If batch.py did `from .solve import solve`, we must patch batch_mod.solve
    # to affect the target actually called inside batch_solve.
    monkeypatch.setattr(batch_mod, "solve", fake_solve, raising=True)

    # Three jobs; the 2nd and 3rd pass different kwargs to test forwarding
    V = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    C = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    cfg = SimulationConfig.linear_elasticity(E=2000.0, nu=0.25, order=1)

    jobs = [
        (V, C, cfg),                                   # no kwargs
        (V, C, cfg, {"sidesets_func": (lambda p: 4)}), # has sidesets_func
        (V, C, cfg, {"dtype": np.float64}),            # has dtype
    ]

    # Run the batch
    out = batch_solve(jobs)

    # --- Assertion 1: length & order-preserving ---
    assert len(out) == 3
    assert [r.meta["idx"] for r in out] == [0, 1, 2]

    # --- Assertion 2: kwargs forwarding ---
    assert out[0].meta["kwargs"] == []                 # job 1 had no kwargs
    assert "sidesets_func" in out[1].meta["kwargs"]    # job 2 forwarded sidesets_func
    assert "dtype"        in out[2].meta["kwargs"]     # job 3 forwarded dtype

    # --- Assertion 3: internal call count (one per job) ---
    assert len(calls) == 3
