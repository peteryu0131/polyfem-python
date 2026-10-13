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


@pytest.mark.parametrize("optimizer_type", ["SGD", "Adam"])
def test_shape_opt_uses_shared_loop_and_matches_manual_updates(monkeypatch, optimizer_type):
    torch = pytest.importorskip("torch")
    from polyfempy import differentiable_api as D

    vertices = torch.ones((2, 3), dtype=torch.float64, requires_grad=True)
    manual = vertices.detach().clone().requires_grad_(True)
    monkeypatch.setattr(D.ShapeOptimization, "__call__", lambda self: self.vertices.square().sum())
    problem = D.shape_opt(D.model({}), vertices, objective=D.Objective.STRESS_NORM)
    calls = []
    original = D.Optimization.optimize

    def run_shared(self, **kwargs):
        calls.append(self)
        return original(self, **kwargs)

    monkeypatch.setattr(D.Optimization, "optimize", run_shared)
    optimizer_class = getattr(torch.optim, optimizer_type)
    result = (problem.optimizer(optimizer_class, lr=0.1).steps(3)
              .optimize(bounds=[(0.75, 1.0)], record=lambda step, value: {"iteration": step}))
    assert len(calls) == 1
    assert isinstance(result, D.OptimizationResult)
    assert isinstance(result, D.ShapeOptimizationResult)
    assert result.vertices is result.parameters[0] is vertices
    assert problem.configured_steps == 3
    assert problem.configured_optimizer.param_groups[0]["params"][0] is vertices
    optimizer = optimizer_class([manual], lr=0.1)
    for step, row in enumerate(result.history):
        optimizer.zero_grad()
        loss = manual.square().sum()
        loss.backward()
        assert row["iteration"] == row["step"] == step
        assert row["loss"] == pytest.approx(float(loss.detach()))
        assert row["gradient_norm"] == pytest.approx(float(manual.grad.norm()))
        optimizer.step()
        with torch.no_grad():
            manual.clamp_(0.75, 1.0)
    torch.testing.assert_close(vertices, manual)
    torch.testing.assert_close(result.loss.detach(), loss.detach())


def test_shape_opt_preserves_keyword_call_and_legacy_result_constructor(monkeypatch):
    torch = pytest.importorskip("torch")
    from polyfempy import differentiable_api as D

    vertices = torch.ones(2, requires_grad=True)
    monkeypatch.setattr(D.ShapeOptimization, "__call__", lambda self: self.vertices.square().sum())
    problem = D.shape_opt(D.model({}), vertices, objective=D.Objective.STRESS_NORM)
    optimizer = torch.optim.SGD([vertices], lr=0.1)
    result = problem.optimize(optimizer=optimizer, steps=2)
    assert problem.configured_optimizer is optimizer
    torch.testing.assert_close(vertices, torch.full_like(vertices, 0.64))
    for legacy in (D.ShapeOptimizationResult(result.history, vertices, result.loss),
                   D.ShapeOptimizationResult(history=result.history, vertices=vertices, loss=result.loss)):
        assert isinstance(legacy, D.OptimizationResult)
        assert legacy.vertices is legacy.parameters[0] is vertices
        assert legacy.history is result.history
        assert legacy.loss is result.loss


def test_shape_opt_requires_configuration_and_shared_parameter_validation():
    torch = pytest.importorskip("torch")
    from polyfempy import differentiable_api as D

    vertices = torch.ones(2, requires_grad=True)
    problem = D.shape_opt(D.model({}), vertices, objective=D.Objective.STRESS_NORM)
    with pytest.raises(ValueError, match="optimizer"):
        problem.optimize()
    with pytest.raises(ValueError, match="steps"):
        problem.optimizer(torch.optim.SGD, lr=0.1).optimize()
    with pytest.raises(TypeError, match="integer"):
        problem.steps(True)
    with pytest.raises(ValueError, match="leaf"):
        D.shape_opt(D.model({}), vertices * 2, objective=D.Objective.STRESS_NORM).optimizer(torch.optim.SGD, lr=0.1)


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
