from __future__ import annotations

import importlib
import sys

import pytest


def test_shape_parameter_records_direct_shape_metadata_without_tensor():
    from polyfempy import differentiable_api as D
    from polyfempy.differentiable_api.parameter import ShapeParameter

    state = D.state({"geometry": [{"mesh": "beam.msh"}]})

    shape = D.parameter.shape(state=state, selection="all")

    assert isinstance(shape, ShapeParameter)
    assert shape.kind == "shape"
    assert shape.state is state
    assert shape.selection == "all"
    assert shape.parametrization == "direct"
    assert shape.tensor is None
    assert shape.tensors() == ()
    assert "geometry" not in shape.__dict__


def test_shape_parameter_records_optional_user_tensor_without_importing_torch():
    from polyfempy import differentiable_api as D

    state = D.state({"geometry": [{"mesh": "beam.msh"}]})
    tensor = object()

    shape = D.parameter.shape(state=state, selection="all", tensor=tensor)

    assert shape.tensor is tensor
    assert shape.tensors() == (tensor,)


def test_shape_parameter_requires_differentiable_state():
    from polyfempy import differentiable_api as D

    with pytest.raises(TypeError, match="state must be a polyfempy.differentiable_api.State"):
        D.parameter.shape(state={"geometry": []}, selection="all")


def test_shape_parameter_requires_selection():
    from polyfempy import differentiable_api as D

    state = D.state({"geometry": [{"mesh": "beam.msh"}]})

    with pytest.raises(ValueError, match="selection must not be None"):
        D.parameter.shape(state=state, selection=None)


def test_shape_parameter_only_supports_direct_parametrization_in_d3():
    from polyfempy import differentiable_api as D

    state = D.state({"geometry": [{"mesh": "beam.msh"}]})

    with pytest.raises(ValueError, match="only direct shape parametrization is supported"):
        D.parameter.shape(state=state, selection="all", parametrization="bbw")


def test_parameter_module_import_does_not_load_torch_or_old_reference_package():
    torch_was_loaded = "torch" in sys.modules
    sys.modules.pop("polyfempy.differentiable", None)

    importlib.import_module("polyfempy.differentiable_api.parameter")

    assert "polyfempy.differentiable" not in sys.modules
    if not torch_was_loaded:
        assert "torch" not in sys.modules

