"""Differentiable simulation support for PolyFEM.

Recommended user scripts should stay on the small public surface below. Lower
level diagnostics, finite-difference checks, and legacy torch-bridge probes live
under ``polyfempy.differentiable.advanced``. Compatibility names are still
available by explicit import when PyTorch is installed, but they are excluded
from ``__all__`` so the recommended API is easy to inspect.
"""

# pyright: reportMissingImports=false
# torch is an optional dependency, so type checkers may not find it.

from __future__ import annotations

import os
import sys
from typing import Callable

from ._exports import (
    ADVANCED_API,
    ADVANCED_COMPAT_API,
    COMPATIBILITY_API,
    PUBLIC_API,
)


# Fix OpenMP library conflicts on Windows before importing torch.
if sys.platform == "win32" and "KMP_DUPLICATE_LIB_OK" not in os.environ:
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    import torch  # pyright: ignore[reportMissingImports]  # noqa: F401

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

__all__ = list(PUBLIC_API)


def _missing_torch_stub(name: str) -> Callable[..., None]:
    def _stub(*args, **kwargs):
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )

    _stub.__name__ = name
    _stub.__qualname__ = name
    _stub.__doc__ = "Unavailable placeholder used when PyTorch is not installed."
    return _stub


if _TORCH_AVAILABLE:
    from .solve_diff import (
        build_lame_from_youngs,
        build_solver_from_settings,
        prepare_differentiable_simulation,
        solve_differentiable,
        solve_differentiable_material,
        solve_differentiable_material_from_youngs,
        solver_body_ids_for_assembly,
        solver_body_slot_mask,
        youngs_to_lame,
        youngs_value_to_internal,
    )
    from .result_diff import DifferentiableMaterialResult, DifferentiableResult
    from .design import (
        ParameterizedVertexDesign,
        make_bounds_projector,
        make_named_parameter_map,
        make_parameter,
    )
    from .geometry_maps import (
        relative_scale,
        scale_selected_vertices,
        scale_selected_vertices_about_axis_center,
        scale_selected_vertices_about_x_center,
        selected_axis_center,
        selected_x_center,
        tan_half_angle_scale,
        vertices_axis_le,
        vertices_y_le,
    )
    from .objective_bridge import (
        ObjectiveLossResult,
        SmoothTimeAggregationName,
        TimeAggregation,
        TimeAggregationName,
        create_polyfem_objective,
        get_direct_von_mises_monitor,
        make_material_von_mises_loss,
        make_stress_norm_loss,
        make_von_mises_loss,
        material_design_vector,
    )
    from .material_optimization import (
        ScalarMaterialOptimizationProblem,
        ScalarMaterialOptimizationStep,
        format_scalar_material_optimization_history_summary,
        format_scalar_material_optimization_step,
        make_material_optimizer,
        prepare_material_differentiable_simulation,
        prepare_material_optimization_problem,
        prepare_scalar_youngs_material_problem,
        run_scalar_material_optimization,
    )
    from .material_diagnostics import (
        apply_finite_difference_gradient_fallback,
        collect_material_chain_diagnostics,
        evaluate_material_loss_value_for_E,
        finite_difference_material_E_gradient,
        format_optional,
        objective_solution_rhs_diagnostics,
        print_material_chain_diagnostics,
        print_objective_rhs_diagnostics,
        usable_scalar_gradient,
    )
    from .shape_optimization import (
        ParameterizedShapeOptimizationProblem,
        ShapeOptimizationProblem,
        ShapeOptimizationStep,
        format_shape_optimization_history_summary,
        format_shape_optimization_step,
        make_shape_optimizer,
        make_von_mises_shape_loss,
        prepare_parameterized_shape_differentiable_simulation,
        prepare_parameterized_shape_optimization_problem,
        prepare_parameterized_shape_problem,
        prepare_shape_differentiable_simulation,
        prepare_shape_optimization_problem,
        print_shape_optimization_step,
        run_shape_optimization,
    )
    from .optimization_problem import (
        OptimizationKind,
        OptimizationProblem,
        OptimizationRunResult,
        make_optimizer,
        prepare_optimization_baseline_simulation,
        prepare_optimization_problem,
        report_optimization_baseline,
        run_optimization,
    )
    from .summary import (
        gradient_norm,
        print_loss_summary,
        print_parameterized_shape_summary,
        print_scalar_material_summary,
    )
    from .shape_mask import body_vertex_mask, shape_gradient_for_body
    from .training_data import save_training_sample
else:
    for _name in dict.fromkeys(PUBLIC_API + ADVANCED_COMPAT_API):
        globals()[_name] = _missing_torch_stub(_name)
    del _name
