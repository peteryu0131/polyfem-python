from __future__ import annotations

import importlib
import sys
import types

import pytest


class _ShapeSession:
    calls: list[tuple] = []

    def __init__(self):
        self.calls.append(("init",))

    def set_settings(self, settings):
        self.calls.append(("set_settings", settings))

    def set_objective(self, objective):
        self.calls.append(("set_objective", objective))

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


class _FakeVertices:
    def __init__(self):
        self.grad = None
        self.value = 0


class _FakeGrad:
    def norm(self):
        return 3.5


class _FakeLoss:
    def __init__(self, value, vertices):
        self.value = value
        self.vertices = vertices

    def backward(self):
        self.vertices.grad = _FakeGrad()

    def detach(self):
        return self

    def __float__(self):
        return float(self.value)


class _FakeShapeOpt:
    calls: list[tuple] = []
    vertex_values: list[int] = []

    @classmethod
    def apply(cls, *args, **kwargs):
        cls.calls.append((args, kwargs))
        cls.vertex_values.append(args[2].value)
        return _FakeLoss(len(cls.calls), args[2])


class _FakeOptimizer:
    def __init__(self, parameters, **kwargs):
        self.parameters = parameters
        self.kwargs = kwargs
        self.calls: list[str] = []

    def zero_grad(self):
        self.calls.append("zero_grad")

    def step(self):
        self.calls.append("step")
        self.parameters[0].value += 1


def _install_fake_shapeopt(monkeypatch):
    module = types.ModuleType("polyfempy.differentiable_api.torch_ops")
    module.ShapeOpt = _FakeShapeOpt
    _FakeShapeOpt.calls = []
    _FakeShapeOpt.vertex_values = []
    monkeypatch.setitem(
        sys.modules,
        "polyfempy.differentiable_api.torch_ops",
        module,
    )


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


def test_shape_opt_builds_callable_problem_without_running_backward(monkeypatch):
    _install_fake_shapeopt(monkeypatch)

    from polyfempy import differentiable_api as D

    payload = {"geometry": [{"mesh": "beam.msh"}]}
    vertices = _FakeVertices()
    diff_model = D.model([payload])

    shape_opt = D.shape_opt(
        diff_model,
        vertices,
        objective=D.Objective.STRESS_NORM,
    )

    assert shape_opt.model is diff_model
    assert shape_opt.vertices is vertices
    assert shape_opt.objective is D.Objective.STRESS_NORM
    assert shape_opt.selection == "all"

    loss = shape_opt()

    assert isinstance(loss, _FakeLoss)
    assert vertices.grad is None
    assert _FakeShapeOpt.calls == [
        (
            (diff_model, "all", vertices),
            {"objective": D.Objective.STRESS_NORM},
        ),
    ]


def test_shape_opt_optimizes_with_chained_optimizer_and_steps(monkeypatch):
    _install_fake_shapeopt(monkeypatch)

    from polyfempy import differentiable_api as D

    vertices = _FakeVertices()
    diff_model = D.model([{"geometry": [{"mesh": "beam.msh"}]}])

    shape_opt = D.shape_opt(
        diff_model,
        vertices,
        objective=D.Objective.STRESS_NORM,
    )
    result = (
        shape_opt
        .optimizer(_FakeOptimizer, lr=1e-7)
        .steps(2)
        .optimize()
    )

    assert result.vertices is vertices
    assert result.loss.value == 2
    assert result.history == (
        {"step": 0, "loss": 1.0, "gradient_norm": 3.5},
        {"step": 1, "loss": 2.0, "gradient_norm": 3.5},
    )
    assert _FakeShapeOpt.vertex_values == [0, 1]
    assert shape_opt.configured_optimizer.parameters == [vertices]
    assert shape_opt.configured_optimizer.kwargs == {"lr": 1e-7}
    assert shape_opt.configured_optimizer.calls == [
        "zero_grad",
        "step",
        "zero_grad",
        "step",
    ]


def test_shape_opt_accepts_existing_optimizer_instance(monkeypatch):
    _install_fake_shapeopt(monkeypatch)

    from polyfempy import differentiable_api as D

    vertices = _FakeVertices()
    optimizer = _FakeOptimizer([vertices], lr=1e-7)
    diff_model = D.model([{"geometry": [{"mesh": "beam.msh"}]}])

    result = (
        D.shape_opt(diff_model, vertices, objective=D.Objective.STRESS_NORM)
        .optimizer(optimizer)
        .steps(1)
        .optimize()
    )

    assert result.vertices is vertices
    assert result.loss.value == 1
    assert optimizer.calls == ["zero_grad", "step"]


def test_shape_opt_requires_optimizer_and_steps_before_optimize(monkeypatch):
    _install_fake_shapeopt(monkeypatch)

    from polyfempy import differentiable_api as D

    shape_opt = D.shape_opt(
        D.model([{"geometry": [{"mesh": "beam.msh"}]}]),
        _FakeVertices(),
        objective=D.Objective.STRESS_NORM,
    )

    with pytest.raises(ValueError, match="optimizer"):
        shape_opt.optimize()

    with pytest.raises(ValueError, match="steps"):
        shape_opt.optimizer(_FakeOptimizer, lr=1e-7).optimize()


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

        def set_objective(self, objective):
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
