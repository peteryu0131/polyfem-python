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


class _FakeTorchTensor:
    def __init__(
        self,
        name: str,
        array,
        *,
        dtype: str = "float64",
        device: str = "cuda:0",
    ):
        self.name = name
        self.array = array
        self.dtype = dtype
        self.device = device
        self.calls: list[str] = []

    def detach(self):
        self.calls.append("detach")
        return self

    def cpu(self):
        self.calls.append("cpu")
        return self

    def numpy(self):
        self.calls.append("numpy")
        return self.array

    def __eq__(self, other):
        return (
            isinstance(other, _FakeTorchTensor)
            and self.name == other.name
            and self.array == other.array
            and self.dtype == other.dtype
            and self.device == other.device
        )


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

    def set_objective(self, objective):
        self.calls.append(("set_objective", objective))

    def solve(self):
        self.calls.append(("solve",))
        return _FakeTensor("solution", self.vertices.shape)

    def backward_shape(self, grad_solution):
        self.calls.append(("backward_shape", grad_solution))
        return _FakeTensor("gradient", self.vertices.shape)


class _TorchShapeBackend:
    DifferentiableSession = _TorchShapeSession


@pytest.fixture(autouse=True)
def _clear_lazy_shapeopt_export_after_test():
    yield
    sys.modules.pop("polyfempy.differentiable_api.torch_ops", None)
    module = sys.modules.get("polyfempy.differentiable_api")
    if module is not None:
        module.__dict__.pop("ShapeOpt", None)


def _install_fake_torch(monkeypatch, *, tensor_type=None):
    torch_module = types.ModuleType("torch")
    autograd_module = types.ModuleType("torch.autograd")
    autograd_module.Function = _FakeFunction
    autograd_module.function = types.SimpleNamespace(
        once_differentiable=lambda fn: fn,
    )
    torch_module.autograd = autograd_module
    torch_module._as_tensor_calls = []
    if tensor_type is not None:
        torch_module.Tensor = tensor_type

        def as_tensor(value, *, dtype=None, device=None):
            torch_module._as_tensor_calls.append((value, dtype, device))
            return tensor_type(
                "as_tensor",
                value,
                dtype=dtype,
                device=device,
            )

        torch_module.as_tensor = as_tensor

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
    assert D.ShapeOpt._last_ctx.session is None
    assert D.ShapeOpt._last_ctx.input_tensor is None
    assert _TorchShapeSession.calls == [
        ("init",),
        ("set_settings", payload),
        ("set_shape_vertices", vertices, "all"),
        ("solve",),
        ("backward_shape", grad_solution),
    ]


def test_shapeopt_accepts_objective_aware_keyword_api(monkeypatch):
    _install_fake_torch(monkeypatch)

    from polyfempy import differentiable_api as D

    _TorchShapeSession.calls = []
    payload = {"geometry": [{"mesh": "beam.msh"}]}
    diff_model = D.model([payload])
    vertices = _FakeTensor("vertices")
    objective = D.MaxStress(selection=7)

    solution = D.ShapeOpt.apply(
        model=diff_model,
        selection="all",
        tensor=vertices,
        objective=objective,
        backend=_TorchShapeBackend,
    )

    assert solution == _FakeTensor("solution")

    grad_solution = _FakeTensor("grad_solution")
    grads = D.ShapeOpt.backward(D.ShapeOpt._last_ctx, grad_solution)

    assert grads == (None, None, _FakeTensor("gradient"), None, None)
    assert _TorchShapeSession.calls == [
        ("init",),
        ("set_settings", payload),
        ("set_objective", {
            "type": "max_stress",
            "state": "last",
            "volume_selection": [7],
        }),
        ("set_shape_vertices", vertices, "all"),
        ("solve",),
        ("backward_shape", grad_solution),
    ]


def test_shapeopt_converts_torch_tensors_at_backend_boundary(monkeypatch):
    torch_module = _install_fake_torch(monkeypatch, tensor_type=_FakeTorchTensor)

    from polyfempy import differentiable_api as D

    class ArraySession:
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
            return "backend-solution"

        def backward_shape(self, grad_solution):
            self.calls.append(("backward_shape", grad_solution))
            return "backend-gradient"

    class ArrayBackend:
        DifferentiableSession = ArraySession

    ArraySession.calls = []
    payload = {"geometry": [{"mesh": "beam.msh"}]}
    diff_model = D.model([payload])
    vertices = _FakeTorchTensor("vertices", "backend-vertices")

    solution = D.ShapeOpt.apply(diff_model, "all", vertices, ArrayBackend)

    assert vertices.calls == ["detach", "cpu", "numpy"]
    assert solution == _FakeTorchTensor("as_tensor", "backend-solution")
    assert torch_module._as_tensor_calls == [
        ("backend-solution", "float64", "cuda:0"),
    ]
    assert ArraySession.calls == [
        ("init",),
        ("set_settings", payload),
        ("set_shape_vertices", "backend-vertices", "all"),
        ("solve",),
    ]

    grad_solution = _FakeTorchTensor("grad_solution", "backend-grad-solution")
    grads = D.ShapeOpt.backward(D.ShapeOpt._last_ctx, grad_solution)

    assert grad_solution.calls == ["detach", "cpu", "numpy"]
    assert grads == (
        None,
        None,
        _FakeTorchTensor("as_tensor", "backend-gradient"),
        None,
    )
    assert D.ShapeOpt._last_ctx.session is None
    assert D.ShapeOpt._last_ctx.input_tensor is None
    assert torch_module._as_tensor_calls == [
        ("backend-solution", "float64", "cuda:0"),
        ("backend-gradient", "float64", "cuda:0"),
    ]
    assert ArraySession.calls == [
        ("init",),
        ("set_settings", payload),
        ("set_shape_vertices", "backend-vertices", "all"),
        ("solve",),
        ("backward_shape", "backend-grad-solution"),
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

        def set_objective(self, objective):
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
