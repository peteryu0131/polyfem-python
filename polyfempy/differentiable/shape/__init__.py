"""Shape-specific differentiable optimization helpers."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_MODULES = {
    "LossBuilder": ".problem",
    "LossOutput": ".problem",
    "ParameterizedShapeOptimizationProblem": ".problem",
    "ShapeOptimizationProblem": ".problem",
    "ShapeOptimizationStep": ".problem",
    "body_vertex_mask": ".mask",
    "shape_gradient_for_body": ".mask",
    "relative_scale": ".geometry_maps",
    "scale_selected_vertices": ".geometry_maps",
    "scale_selected_vertices_about_axis_center": ".geometry_maps",
    "scale_selected_vertices_about_x_center": ".geometry_maps",
    "selected_axis_center": ".geometry_maps",
    "selected_x_center": ".geometry_maps",
    "tan_half_angle_scale": ".geometry_maps",
    "vertices_axis_le": ".geometry_maps",
    "vertices_y_le": ".geometry_maps",
    "format_shape_optimization_history_summary": ".optimization",
    "format_shape_optimization_step": ".optimization",
    "make_shape_optimizer": ".optimization",
    "make_von_mises_shape_loss": ".optimization",
    "prepare_parameterized_shape_differentiable_simulation": ".optimization",
    "prepare_parameterized_shape_optimization_problem": ".optimization",
    "prepare_parameterized_shape_problem": ".optimization",
    "prepare_shape_differentiable_simulation": ".optimization",
    "prepare_shape_optimization_problem": ".optimization",
    "print_shape_optimization_step": ".optimization",
    "run_shape_optimization": ".optimization",
}

__all__ = sorted(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, package=__name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *__all__])
