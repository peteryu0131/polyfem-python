from __future__ import annotations

import importlib
import sys
import types

import pytest


class _FakeFunction:
    @classmethod
    def apply(cls, *args):
        ctx = types.SimpleNamespace()
        cls._last_ctx = ctx
        return cls.forward(ctx, *args)


class _FakeTensor:
    def __init__(self, name: str, shape: tuple[int, ...] = (3,)):
        self.name = name
        self.shape = shape

    def __eq__(self, other):
        return (
            isinstance(other, _FakeTensor)
            and self.name == other.name
            and self.shape == other.shape
        )

    def __repr__(self):
        return f"_FakeTensor(name={self.name!r}, shape={self.shape!r})"


class _TorchShapeSession:
    calls: list[tuple] = []

    def __init__(self):
        self.vertices = None
        self.calls.append(("init",))

    def set_settings(self, settings):
        self.calls.append(("set_settings", settings))

    def set_shape_vertices(self, vertices, *, selection=None):
        self.vertices = vertices
        self.calls.append(("set_shape_vertices", vertices, selection))

    def solve(self):
        self.calls.append(("solve",))
        return _FakeTensor("solution", self.vertices.shape)

    def backward_shape(self, grad_solution):
        self.calls.append(("backward_shape", grad_solution))
        return _FakeTensor("gradient", self.vertices.shape)


class _TorchShapeBackend:
    DifferentiableSession = _TorchShapeSession


def _install_fake_torch(monkeypatch):
    torch_module = types.ModuleType("torch")
    autograd_module = types.ModuleType("torch.autograd")
    autograd_module.Function = _FakeFunction
    autograd_module.function = types.SimpleNamespace(
        once_differentiable=lambda fn: fn,
    )
    torch_module.autograd = autograd_module

    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "torch.autograd", autograd_module)

    sys.modules.pop("polyfempy.differentiable_api.torch_ops", None)
    import polyfempy.differentiable_api as D

    D.__dict__.pop("ShapeOpt", None)
    return torch_module


def test_shapeopt_forward_backward_uses_shape_backend_contract(monkeypatch):
    _install_fake_torch(monkeypatch)

    from polyfempy import differentiable_api as D

    _TorchShapeSession.calls = []
    payload = {"geometry": [{"mesh": "beam.msh"}]}
    diff_model = D.model([payload])
    vertices = _FakeTensor("vertices")

    solution = D.ShapeOpt.apply(diff_model, "all", vertices, _TorchShapeBackend)

    assert solution == _FakeTensor("solution")

    grad_solution = _FakeTensor("grad_solution")
    grads = D.ShapeOpt.backward(D.ShapeOpt._last_ctx, grad_solution)

    assert grads == (None, None, _FakeTensor("gradient"), None)
    assert _TorchShapeSession.calls == [
        ("init",),
        ("set_settings", payload),
        ("set_shape_vertices", vertices, "all"),
        ("solve",),
        ("backward_shape", grad_solution),
    ]


def test_shapeopt_rejects_multi_model_for_mvp(monkeypatch):
    _install_fake_torch(monkeypatch)

    from polyfempy import differentiable_api as D

    diff_model = D.model([
        {"geometry": [{"mesh": "beam-a.msh"}]},
        {"geometry": [{"mesh": "beam-b.msh"}]},
    ])

    with pytest.raises(ValueError, match="exactly one model"):
        D.ShapeOpt.apply(diff_model, "all", _FakeTensor("vertices"), _TorchShapeBackend)


def test_shapeopt_rejects_missing_shape_backend_contract(monkeypatch):
    _install_fake_torch(monkeypatch)

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
        D.ShapeOpt.apply(diff_model, "all", _FakeTensor("vertices"), IncompleteBackend)


def test_shapeopt_lazy_export_loads_torch_without_loading_old_reference_package(monkeypatch):
    _install_fake_torch(monkeypatch)
    sys.modules.pop("polyfempy.differentiable", None)

    from polyfempy import differentiable_api as D

    assert D.ShapeOpt.__name__ == "ShapeOpt"
    assert "torch" in sys.modules
    assert "polyfempy.differentiable" not in sys.modules


def test_torch_ops_module_import_does_not_load_old_reference_package(monkeypatch):
    _install_fake_torch(monkeypatch)
    sys.modules.pop("polyfempy.differentiable", None)

    importlib.import_module("polyfempy.differentiable_api.torch_ops")

    assert "polyfempy.differentiable" not in sys.modules

