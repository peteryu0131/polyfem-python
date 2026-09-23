from __future__ import annotations

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


class _CompleteSession:
    def set_settings(self, settings):
        return None

    def set_objective(self, objective):
        return None

    def set_shape_parameter(self, value, *, selection=None):
        return None

    def solve(self):
        return None

    def backward_shape(self, grad_u):
        return None


class _CompleteBackend:
    DifferentiableSession = _CompleteSession


def _spec_variable_to_simulation_options() -> tuple[str, ...]:
    spec_path = ROOT / "polyfem" / "json-specs" / "opt-input-spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    for entry in spec:
        if entry.get("pointer") == "/variable_to_simulation/*/type":
            return tuple(entry["options"])
    raise AssertionError("opt-input-spec.json is missing /variable_to_simulation/*/type")


def test_backend_contract_is_shape_mvp_only():
    from polyfempy.differentiable_api import _backend

    assert _backend.REQUIRED_BACKEND_SYMBOLS == ("DifferentiableSession",)
    assert _backend.REQUIRED_SESSION_METHODS == (
        "set_settings",
        "set_objective",
        "set_shape_parameter",
        "solve",
        "backward_shape",
    )
    assert _backend.REQUIRED_SHAPE_SOLVE_SESSION_METHODS == (
        "set_settings",
        "set_objective",
        "set_shape_vertices",
        "solve",
        "backward_shape",
    )
    assert _backend.SUPPORTED_PARAMETER_KINDS == ("shape",)
    assert _backend.UNSUPPORTED_OPT_PARAMETER_KINDS == (
        "periodic-shape",
        "elastic",
        "friction",
        "damping",
        "initial",
        "dirichlet-boundary",
        "dirichlet-nodes",
        "pressure",
    )


def test_backend_parameter_policy_tracks_opt_input_spec():
    from polyfempy.differentiable_api import _backend

    spec_options = _spec_variable_to_simulation_options()

    assert _backend.declared_opt_parameter_kinds() == spec_options
    assert set(_backend.SUPPORTED_PARAMETER_KINDS).isdisjoint(
        _backend.UNSUPPORTED_OPT_PARAMETER_KINDS
    )
    assert set(_backend.SUPPORTED_PARAMETER_KINDS) | set(
        _backend.UNSUPPORTED_OPT_PARAMETER_KINDS
    ) == set(spec_options)


def test_backend_contract_accepts_complete_shape_session():
    from polyfempy.differentiable_api import _backend

    session_type = _backend.require_shape_mvp_backend(_CompleteBackend())

    assert session_type is _CompleteSession


def test_compiled_backend_exports_differentiable_session_skeleton():
    backend = pytest.importorskip("polyfempy.polyfempy")
    from polyfempy.differentiable_api import _backend

    session_type = _backend.require_shape_solve_backend(backend)
    session = session_type()

    for method in _backend.REQUIRED_SHAPE_SOLVE_SESSION_METHODS:
        assert callable(getattr(session, method))


def test_backend_contract_rejects_missing_session_type():
    from polyfempy.differentiable_api import _backend

    with pytest.raises(_backend.BackendContractError, match="DifferentiableSession"):
        _backend.require_shape_mvp_backend(object())


def test_backend_contract_rejects_missing_session_method():
    from polyfempy.differentiable_api import _backend

    class IncompleteSession:
        def set_settings(self, settings):
            return None

    class IncompleteBackend:
        DifferentiableSession = IncompleteSession

    with pytest.raises(_backend.BackendContractError, match="backward_shape"):
        _backend.require_shape_mvp_backend(IncompleteBackend())
