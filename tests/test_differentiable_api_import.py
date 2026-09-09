from __future__ import annotations

import importlib
import sys


def test_differentiable_api_import_surface_is_small():
    from polyfempy import differentiable_api as D

    assert D.__all__ == [
        "DifferentiableResult",
        "State",
        "parameter",
        "solve",
        "state",
    ]
    assert callable(D.state)
    assert hasattr(D.parameter, "shape")
    assert callable(D.parameter.shape)
    assert callable(D.solve)


def test_differentiable_api_import_does_not_load_torch_or_old_reference_package():
    torch_was_loaded = "torch" in sys.modules
    sys.modules.pop("polyfempy.differentiable", None)

    importlib.import_module("polyfempy.differentiable_api")

    assert "polyfempy.differentiable" not in sys.modules
    if not torch_was_loaded:
        assert "torch" not in sys.modules

