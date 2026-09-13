from __future__ import annotations

import pytest


class _CompleteSession:
    def set_settings(self, settings):
        return None

    def set_shape_parameter(self, value, *, selection=None):
        return None

    def solve(self):
        return None

    def backward_shape(self, grad_u):
        return None


class _CompleteBackend:
    DifferentiableSession = _CompleteSession


def test_backend_contract_is_shape_mvp_only():
    from polyfempy.differentiable_api import _backend

    assert _backend.REQUIRED_BACKEND_SYMBOLS == ("DifferentiableSession",)
    assert _backend.REQUIRED_SESSION_METHODS == (
        "set_settings",
        "set_shape_parameter",
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


def test_backend_contract_accepts_complete_shape_session():
    from polyfempy.differentiable_api import _backend

    session_type = _backend.require_shape_mvp_backend(_CompleteBackend())

    assert session_type is _CompleteSession


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

