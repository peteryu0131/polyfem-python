"""Unified entry point for optimization problem preparation."""

from __future__ import annotations

from typing import Any, Literal, Union, overload

from ..material.optimization import (
    ScalarMaterialOptimizationProblem,
    prepare_material_optimization_problem,
    prepare_scalar_youngs_material_problem,
)
from .runner import (
    make_optimizer,
    OptimizationRunResult,
    prepare_optimization_baseline_simulation,
    report_optimization_baseline,
    run_optimization,
)
from ..shape.optimization import (
    ParameterizedShapeOptimizationProblem,
    ShapeOptimizationProblem,
    prepare_parameterized_shape_problem,
    prepare_shape_optimization_problem,
)


OptimizationKind = Literal[
    "shape",
    "geometry",
    "parameterized_shape",
    "parametric_shape",
    "material",
    "E",
    "e",
    "youngs",
]
OptimizationProblem = Union[
    ShapeOptimizationProblem,
    ParameterizedShapeOptimizationProblem,
    ScalarMaterialOptimizationProblem,
]


@overload
def prepare_optimization_problem(
    *,
    cfg: Any,
    kind: Literal["shape", "geometry"] = "shape",
    **kwargs: Any,
) -> ShapeOptimizationProblem:
    ...


@overload
def prepare_optimization_problem(
    *,
    cfg: Any,
    kind: Literal["material", "E", "e", "youngs"],
    **kwargs: Any,
) -> ScalarMaterialOptimizationProblem:
    ...


@overload
def prepare_optimization_problem(
    *,
    cfg: Any,
    kind: Literal["parameterized_shape", "parametric_shape"],
    **kwargs: Any,
) -> ParameterizedShapeOptimizationProblem:
    ...


def prepare_optimization_problem(
    *,
    cfg: Any,
    kind: str = "shape",
    **kwargs: Any,
) -> OptimizationProblem:
    """Prepare an optimization problem by user-facing optimization kind.

    ``kind="shape"`` prepares a vertex/geometry optimization problem.
    ``kind="parameterized_shape"`` prepares a user-parameterized vertex-map
    optimization problem.
    ``kind="material"`` prepares a scalar Young's modulus optimization problem
    and requires the material-specific arguments such as ``body_id``. By
    default it keeps the compatibility ``log_E`` parameterization. Pass an
    explicit ``E_parameter`` to optimize a user-facing physical E parameter.
    """
    normalized = str(kind).strip().lower()
    if normalized in {"shape", "geometry"}:
        return prepare_shape_optimization_problem(cfg=cfg, **kwargs)
    if normalized in {
        "parameterized_shape",
        "parameterized-shape",
        "parametric_shape",
        "parametric-shape",
    }:
        return prepare_parameterized_shape_problem(cfg=cfg, **kwargs)
    if normalized in {"material", "e", "youngs"}:
        if any(key in kwargs for key in ("E_parameter", "parameter_name", "bounds")):
            return prepare_scalar_youngs_material_problem(cfg=cfg, **kwargs)
        return prepare_material_optimization_problem(cfg=cfg, **kwargs)
    raise ValueError(
        "unsupported optimization kind "
        f"{kind!r}; expected 'shape', 'parameterized_shape', or 'material'"
    )


__all__ = [
    "make_optimizer",
    "OptimizationKind",
    "OptimizationProblem",
    "OptimizationRunResult",
    "ParameterizedShapeOptimizationProblem",
    "prepare_optimization_baseline_simulation",
    "prepare_optimization_problem",
    "report_optimization_baseline",
    "run_optimization",
]
