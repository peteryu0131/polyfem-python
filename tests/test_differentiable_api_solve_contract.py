from __future__ import annotations

import importlib
import sys

import pytest


class _FakeSession:
    calls: list[tuple] = []

    def __init__(self):
        self.calls.append(("init",))

    def set_settings(self, settings):
        self.calls.append(("set_settings", settings))

    def set_shape_parameter(self, value, *, selection=None):
        self.calls.append(("set_shape_parameter", value, selection))

    def solve(self):
        self.calls.append(("solve",))
        return "fake-solution"

    def backward_shape(self, grad_u):
        self.calls.append(("backward_shape", grad_u))
        return "fake-gradient"


class _FakeBackend:
    DifferentiableSession = _FakeSession


def test_solve_runs_shape_mvp_call_sequence_with_fake_backend():
    from polyfempy import differentiable_api as D

    _FakeSession.calls = []
    payload = {"geometry": [{"mesh": "beam.msh"}]}
    tensor = object()
    state = D.state(payload)
    shape = D.parameter.shape(state=state, selection="all", tensor=tensor)

    result = D.solve(state=state, parameters=[shape], backend=_FakeBackend)

    assert result.u == "fake-solution"
    assert result.parameters == (shape,)
    assert _FakeSession.calls == [
        ("init",),
        ("set_settings", payload),
        ("set_shape_parameter", tensor, "all"),
        ("solve",),
    ]


def test_solve_requires_differentiable_state():
    from polyfempy import differentiable_api as D

    with pytest.raises(TypeError, match="state must be a polyfempy.differentiable_api.State"):
        D.solve(state={"geometry": []}, parameters=[], backend=_FakeBackend)


def test_solve_requires_at_least_one_parameter():
    from polyfempy import differentiable_api as D

    state = D.state({"geometry": [{"mesh": "beam.msh"}]})

    with pytest.raises(ValueError, match="at least one differentiable parameter"):
        D.solve(state=state, parameters=[], backend=_FakeBackend)


def test_solve_rejects_parameter_from_another_state():
    from polyfempy import differentiable_api as D

    state = D.state({"geometry": [{"mesh": "beam.msh"}]})
    other_state = D.state({"geometry": [{"mesh": "other.msh"}]})
    shape = D.parameter.shape(state=other_state, selection="all")

    with pytest.raises(ValueError, match="same state"):
        D.solve(state=state, parameters=[shape], backend=_FakeBackend)


def test_solve_rejects_unsupported_parameter_kind():
    from polyfempy import differentiable_api as D

    class ElasticParameter:
        kind = "elastic"

        def __init__(self, state):
            self.state = state

    state = D.state({"geometry": [{"mesh": "beam.msh"}]})

    with pytest.raises(ValueError, match="unsupported differentiable parameter kind"):
        D.solve(state=state, parameters=[ElasticParameter(state)], backend=_FakeBackend)


def test_solve_requires_shape_parameter_object():
    from polyfempy import differentiable_api as D

    class ShapeLikeParameter:
        kind = "shape"
        selection = "all"
        tensor = None

        def __init__(self, state):
            self.state = state

    state = D.state({"geometry": [{"mesh": "beam.msh"}]})

    with pytest.raises(TypeError, match="D.parameter.shape"):
        D.solve(state=state, parameters=[ShapeLikeParameter(state)], backend=_FakeBackend)


def test_solve_reports_missing_backend_contract():
    from polyfempy import differentiable_api as D
    from polyfempy.differentiable_api import _backend

    state = D.state({"geometry": [{"mesh": "beam.msh"}]})
    shape = D.parameter.shape(state=state, selection="all")

    with pytest.raises(_backend.BackendContractError, match="DifferentiableSession"):
        D.solve(state=state, parameters=[shape], backend=object())


def test_solve_import_does_not_load_torch_or_old_reference_package():
    torch_was_loaded = "torch" in sys.modules
    sys.modules.pop("polyfempy.differentiable", None)

    importlib.import_module("polyfempy.differentiable_api.solve")

    assert "polyfempy.differentiable" not in sys.modules
    if not torch_was_loaded:
        assert "torch" not in sys.modules
