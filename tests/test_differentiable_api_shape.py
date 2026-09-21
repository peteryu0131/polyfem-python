from __future__ import annotations

import importlib
import sys

import pytest


class _ShapeSession:
    calls: list[tuple] = []

    def __init__(self):
        self.calls.append(("init",))

    def set_settings(self, settings):
        self.calls.append(("set_settings", settings))

    def set_shape_vertices(self, vertices, *, selection=None):
        self.calls.append(("set_shape_vertices", vertices, selection))

    def solve(self):
        self.calls.append(("solve",))
        return "fake-solution"

    def backward_shape(self, grad_solution):
        self.calls.append(("backward_shape", grad_solution))
        return "fake-gradient"


class _ShapeBackend:
    DifferentiableSession = _ShapeSession


def test_shape_solve_runs_single_model_shape_call_sequence_with_fake_backend():
    from polyfempy import differentiable_api as D

    _ShapeSession.calls = []
    payload = {"geometry": [{"mesh": "beam.msh"}]}
    vertices = object()
    diff_model = D.model([payload])

    solution = D.shape_solve(
        model=diff_model,
        selection="all",
        tensor=vertices,
        backend=_ShapeBackend,
    )

    assert solution == "fake-solution"
    assert _ShapeSession.calls == [
        ("init",),
        ("set_settings", payload),
        ("set_shape_vertices", vertices, "all"),
        ("solve",),
    ]


def test_differentiable_model_shape_delegates_to_shape_solve():
    from polyfempy import differentiable_api as D

    _ShapeSession.calls = []
    payload = {"geometry": [{"mesh": "beam.msh"}]}
    vertices = object()
    diff_model = D.model([payload])

    solution = diff_model.shape(
        selection="all",
        tensor=vertices,
        backend=_ShapeBackend,
    )

    assert solution == "fake-solution"
    assert _ShapeSession.calls == [
        ("init",),
        ("set_settings", payload),
        ("set_shape_vertices", vertices, "all"),
        ("solve",),
    ]


def test_shape_solve_rejects_multi_model_for_mvp():
    from polyfempy import differentiable_api as D

    diff_model = D.model([
        {"geometry": [{"mesh": "beam-a.msh"}]},
        {"geometry": [{"mesh": "beam-b.msh"}]},
    ])

    with pytest.raises(ValueError, match="exactly one model"):
        D.shape_solve(
            model=diff_model,
            selection="all",
            tensor=object(),
            backend=_ShapeBackend,
        )


def test_shape_solve_rejects_non_differentiable_model():
    from polyfempy import differentiable_api as D

    with pytest.raises(TypeError, match="model must be a polyfempy.differentiable_api.DifferentiableModel"):
        D.shape_solve(
            model={"geometry": []},
            selection="all",
            tensor=object(),
            backend=_ShapeBackend,
        )


def test_shape_solve_requires_selection_and_tensor():
    from polyfempy import differentiable_api as D

    diff_model = D.model([{"geometry": [{"mesh": "beam.msh"}]}])

    with pytest.raises(ValueError, match="selection must not be None"):
        D.shape_solve(
            model=diff_model,
            selection=None,
            tensor=object(),
            backend=_ShapeBackend,
        )

    with pytest.raises(ValueError, match="tensor must not be None"):
        D.shape_solve(
            model=diff_model,
            selection="all",
            tensor=None,
            backend=_ShapeBackend,
        )


def test_shape_solve_reports_missing_shape_backend_contract():
    from polyfempy import differentiable_api as D
    from polyfempy.differentiable_api import _backend

    class IncompleteSession:
        def set_settings(self, settings):
            return None

        def solve(self):
            return None

    class IncompleteBackend:
        DifferentiableSession = IncompleteSession

    diff_model = D.model([{"geometry": [{"mesh": "beam.msh"}]}])

    with pytest.raises(_backend.BackendContractError, match="set_shape_vertices"):
        D.shape_solve(
            model=diff_model,
            selection="all",
            tensor=object(),
            backend=IncompleteBackend,
        )


def test_shape_module_import_does_not_load_torch_or_old_reference_package():
    torch_was_loaded = "torch" in sys.modules
    sys.modules.pop("polyfempy.differentiable", None)

    importlib.import_module("polyfempy.differentiable_api.shape")

    assert "polyfempy.differentiable" not in sys.modules
    if not torch_was_loaded:
        assert "torch" not in sys.modules
