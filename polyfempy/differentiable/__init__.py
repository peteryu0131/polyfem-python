"""Differentiable simulation support for PolyFEM.

This public module intentionally exposes a small user-facing surface:

- ``prepare_differentiable_simulation(...)`` for one-off loss/gradient runs
- ``solve_differentiable(...)`` as the lower-level backward-compatible alias
- ``solve_differentiable_material_from_youngs(...)`` for ``E, nu`` material solves
- ``make_von_mises_loss(...)`` / ``make_stress_norm_loss(...)`` for common losses
- ``prepare_optimization_problem(...)`` as a unified shape/material optimizer entry point
- ``make_parameter(...)`` / ``prepare_parameterized_shape_problem(...)`` for named shape parameters
- ``ParameterizedVertexDesign`` for user-defined ``params -> vertices`` shape maps
- ``prepare_shape_optimization_problem(...)`` for small vertex-optimization loops
- ``get_direct_von_mises_monitor(...)`` for monitor-only solver output
- ``print_loss_summary(...)`` / ``gradient_norm(...)`` for tiny demo summaries

Lower-level bridge/operator helpers stay in submodules such as
``objective_bridge`` and ``torch_bridge``.
"""

# pyright: reportMissingImports=false
# torch is an optional dependency, so type checker may not find it

import os
import sys

# Fix OpenMP library conflicts on Windows (set before importing torch)
if sys.platform == 'win32' and 'KMP_DUPLICATE_LIB_OK' not in os.environ:
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

try:
    import torch  # pyright: ignore[reportMissingImports]
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

if _TORCH_AVAILABLE:
    from .solve_diff import (
        build_solver_from_settings,
        solver_body_ids_for_assembly,
        solver_body_slot_mask,
        prepare_differentiable_simulation,
        solve_differentiable,
        solve_differentiable_material,
        solve_differentiable_material_from_youngs,
        youngs_value_to_internal,
        youngs_to_lame,
        build_lame_from_youngs,
    )
    from .result_diff import DifferentiableResult, DifferentiableMaterialResult
    from .design import (
        ParameterizedVertexDesign,
        make_bounds_projector,
        make_named_parameter_map,
        make_parameter,
    )
    from .objective_bridge import (
        ObjectiveLossResult,
        SmoothTimeAggregationName,
        TimeAggregation,
        TimeAggregationName,
        create_polyfem_objective,
        make_material_von_mises_loss,
        make_von_mises_loss,
        make_stress_norm_loss,
        material_design_vector,
        get_direct_von_mises_monitor,
    )
    from .material_optimization import (
        ScalarMaterialOptimizationProblem,
        ScalarMaterialOptimizationStep,
        apply_finite_difference_gradient_fallback,
        collect_material_chain_diagnostics,
        evaluate_material_loss_value_for_E,
        finite_difference_material_E_gradient,
        format_optional,
        format_scalar_material_optimization_history_summary,
        format_scalar_material_optimization_step,
        make_material_optimizer,
        objective_solution_rhs_diagnostics,
        print_material_chain_diagnostics,
        print_objective_rhs_diagnostics,
        prepare_material_differentiable_simulation,
        prepare_material_optimization_problem,
        run_scalar_material_optimization,
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
        prepare_parameterized_shape_problem,
        prepare_parameterized_shape_optimization_problem,
        prepare_shape_differentiable_simulation,
        prepare_shape_optimization_problem,
        print_shape_optimization_step,
        run_shape_optimization,
    )
    from .optimization_problem import (
        make_optimizer,
        OptimizationKind,
        OptimizationProblem,
        prepare_optimization_baseline_simulation,
        prepare_optimization_problem,
        report_optimization_baseline,
        run_optimization,
    )
    from .summary import gradient_norm, print_loss_summary
    from .shape_mask import body_vertex_mask, shape_gradient_for_body
    from .training_data import save_training_sample
    __all__ = [
        "build_solver_from_settings",
        "solver_body_ids_for_assembly",
        "solver_body_slot_mask",
        "prepare_differentiable_simulation",
        "solve_differentiable",
        "solve_differentiable_material",
        "solve_differentiable_material_from_youngs",
        "youngs_value_to_internal",
        "youngs_to_lame",
        "build_lame_from_youngs",
        "DifferentiableResult",
        "DifferentiableMaterialResult",
        "ParameterizedVertexDesign",
        "make_bounds_projector",
        "make_named_parameter_map",
        "make_parameter",
        "ObjectiveLossResult",
        "SmoothTimeAggregationName",
        "TimeAggregation",
        "TimeAggregationName",
        "create_polyfem_objective",
        "make_material_von_mises_loss",
        "make_von_mises_loss",
        "make_stress_norm_loss",
        "material_design_vector",
        "make_optimizer",
        "OptimizationKind",
        "OptimizationProblem",
        "prepare_optimization_baseline_simulation",
        "prepare_optimization_problem",
        "report_optimization_baseline",
        "run_optimization",
        "ScalarMaterialOptimizationProblem",
        "ScalarMaterialOptimizationStep",
        "apply_finite_difference_gradient_fallback",
        "collect_material_chain_diagnostics",
        "evaluate_material_loss_value_for_E",
        "finite_difference_material_E_gradient",
        "format_optional",
        "format_scalar_material_optimization_history_summary",
        "format_scalar_material_optimization_step",
        "make_material_optimizer",
        "objective_solution_rhs_diagnostics",
        "print_material_chain_diagnostics",
        "print_objective_rhs_diagnostics",
        "prepare_material_differentiable_simulation",
        "prepare_material_optimization_problem",
        "run_scalar_material_optimization",
        "usable_scalar_gradient",
        "ParameterizedShapeOptimizationProblem",
        "ShapeOptimizationProblem",
        "ShapeOptimizationStep",
        "format_shape_optimization_history_summary",
        "format_shape_optimization_step",
        "make_shape_optimizer",
        "make_von_mises_shape_loss",
        "prepare_parameterized_shape_differentiable_simulation",
        "prepare_parameterized_shape_problem",
        "prepare_parameterized_shape_optimization_problem",
        "prepare_shape_differentiable_simulation",
        "prepare_shape_optimization_problem",
        "print_shape_optimization_step",
        "run_shape_optimization",
        "get_direct_von_mises_monitor",
        "gradient_norm",
        "body_vertex_mask",
        "shape_gradient_for_body",
        "print_loss_summary",
        "save_training_sample",
    ]
else:
    __all__ = []

    def build_solver_from_settings(*args, **kwargs):
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )

    def solver_body_ids_for_assembly(*args, **kwargs):
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )

    def solver_body_slot_mask(*args, **kwargs):
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )

    def prepare_differentiable_simulation(*args, **kwargs):
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )
    
    def solve_differentiable(*args, **kwargs):
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )

    def DifferentiableResult(*args, **kwargs):
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )
    
    def solve_differentiable_material(*args, **kwargs):
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )

    def solve_differentiable_material_from_youngs(*args, **kwargs):
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )

    def youngs_value_to_internal(*args, **kwargs):
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )

    def youngs_to_lame(*args, **kwargs):
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )

    def build_lame_from_youngs(*args, **kwargs):
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )

    def DifferentiableMaterialResult(*args, **kwargs):
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )

    def ParameterizedVertexDesign(*args, **kwargs):
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )

    def make_bounds_projector(*args, **kwargs):
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )

    def make_named_parameter_map(*args, **kwargs):
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )

    def make_parameter(*args, **kwargs):
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )

    def make_von_mises_loss(*args, **kwargs):
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )

    def make_material_von_mises_loss(*args, **kwargs):
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )

    def make_stress_norm_loss(*args, **kwargs):
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )

    def create_polyfem_objective(*args, **kwargs):
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )

    def material_design_vector(*args, **kwargs):
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )

    def apply_finite_difference_gradient_fallback(*args, **kwargs):
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )

    def ScalarMaterialOptimizationProblem(*args, **kwargs):
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )

    def ScalarMaterialOptimizationStep(*args, **kwargs):
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )

    def collect_material_chain_diagnostics(*args, **kwargs):
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )

    def evaluate_material_loss_value_for_E(*args, **kwargs):
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )

    def finite_difference_material_E_gradient(*args, **kwargs):
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )

    def format_optional(*args, **kwargs):
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )

    def format_scalar_material_optimization_history_summary(*args, **kwargs):
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )

    def format_scalar_material_optimization_step(*args, **kwargs):
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )

    def make_material_optimizer(*args, **kwargs):
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )

    def objective_solution_rhs_diagnostics(*args, **kwargs):
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )

    def print_material_chain_diagnostics(*args, **kwargs):
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )

    def print_objective_rhs_diagnostics(*args, **kwargs):
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )

    def prepare_material_optimization_problem(*args, **kwargs):
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )

    def prepare_material_differentiable_simulation(*args, **kwargs):
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )

    def run_scalar_material_optimization(*args, **kwargs):
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )

    def usable_scalar_gradient(*args, **kwargs):
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )

    def ShapeOptimizationProblem(*args, **kwargs):
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )

    def ParameterizedShapeOptimizationProblem(*args, **kwargs):
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )

    def ShapeOptimizationStep(*args, **kwargs):
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )

    def make_shape_optimizer(*args, **kwargs):
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )

    def make_von_mises_shape_loss(*args, **kwargs):
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )

    def prepare_parameterized_shape_differentiable_simulation(*args, **kwargs):
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )

    def prepare_parameterized_shape_problem(*args, **kwargs):
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )

    def prepare_parameterized_shape_optimization_problem(*args, **kwargs):
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )

    def prepare_shape_differentiable_simulation(*args, **kwargs):
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )

    def format_shape_optimization_step(*args, **kwargs):
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )

    def format_shape_optimization_history_summary(*args, **kwargs):
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )

    def prepare_shape_optimization_problem(*args, **kwargs):
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )

    def prepare_optimization_problem(*args, **kwargs):
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )

    def prepare_optimization_baseline_simulation(*args, **kwargs):
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )

    def report_optimization_baseline(*args, **kwargs):
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )

    def make_optimizer(*args, **kwargs):
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )

    def run_optimization(*args, **kwargs):
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )

    def print_shape_optimization_step(*args, **kwargs):
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )

    def run_shape_optimization(*args, **kwargs):
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )

    def save_training_sample(*args, **kwargs):
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )

    def body_vertex_mask(*args, **kwargs):
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )

    def shape_gradient_for_body(*args, **kwargs):
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )

    def get_direct_von_mises_monitor(*args, **kwargs):
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )

    def gradient_norm(*args, **kwargs):
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )

    def print_loss_summary(*args, **kwargs):
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )
