from __future__ import annotations

import importlib
import sys


def test_public_imports_reuse_existing_api_objects():
    import polyfem
    import polyfem.diff as diff
    from polyfempy import differentiable_api
    from polyfempy.generated_api import generated_api

    assert polyfem.model is generated_api.model
    assert polyfem.config is generated_api.config
    assert diff.material is differentiable_api.material
    assert diff.Objective is differentiable_api.Objective


def test_differentiable_api_import_surface_is_small():
    from polyfempy import differentiable_api as D

    assert D.__all__ == [
        "DifferentiableModel",
        "ModelMetadata",
        "DifferentiableResult",
        "InitialConditionOpt",
        "InitialConditionOptimization",
        "MaxStress",
        "MaterialOpt",
        "MaterialOptimization",
        "Objective",
        "ObjectiveSpec",
        "Optimization",
        "OptimizationResult",
        "optimization",
        "ShapeOpt",
        "ShapeOptimization",
        "ShapeOptimizationResult",
        "State",
        "StressNorm",
        "initial_condition_opt",
        "model",
        "object",
        "material",
        "shape",
        "initial",
        "damping",
        "friction",
        "dirichlet",
        "pressure",
        "periodic_shape",
        "material_opt",
        "objectives",
        "parameter",
        "shape_opt",
        "shape_solve",
        "solve",
        "state",
    ]
    assert callable(D.model)
    assert callable(D.periodic_shape)
    assert D.Objective.STRESS_NORM.value == "stress_norm"
    assert callable(D.MaxStress)
    assert callable(D.StressNorm)
    assert callable(D.initial_condition_opt)
    assert callable(D.shape_opt)
    assert callable(D.shape_solve)
    assert callable(D.state)
    assert callable(D.objectives.stress_norm)
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


def test_short_factories_survive_implementation_module_imports():
    import subprocess
    code = """
import importlib
from polyfempy import differentiable_api as diff
for name in ('material', 'shape', 'initial', 'damping', 'friction', 'dirichlet', 'pressure'):
    assert callable(getattr(diff, name))
    assert getattr(diff, name) is getattr(diff.object, name)
for name in ('material', 'shape', 'control', 'initial_condition'):
    importlib.import_module('polyfempy.differentiable_api.' + name)
from polyfempy.differentiable_api.material import MaterialOptimization
from polyfempy.differentiable_api.shape import ShapeOptimization
assert callable(diff.material) and callable(diff.shape)
assert diff.material is diff.object.material
assert diff.shape is diff.object.shape
"""
    subprocess.run([sys.executable, '-c', code], check=True)
