"""Unified entry point for optimization problem preparation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Union, overload

from ..api.runtime import report_history_bundle
from .material_optimization import (
    ScalarMaterialOptimizationProblem,
    make_material_optimizer,
    prepare_material_differentiable_simulation,
    prepare_material_optimization_problem,
    run_scalar_material_optimization,
)
from .shape_optimization import (
    ParameterizedShapeOptimizationProblem,
    ShapeOptimizationProblem,
    make_shape_optimizer,
    prepare_parameterized_shape_differentiable_simulation,
    prepare_parameterized_shape_optimization_problem,
    prepare_shape_differentiable_simulation,
    prepare_shape_optimization_problem,
    run_shape_optimization,
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
_DEFAULT = object()


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
    and requires the material-specific arguments such as ``body_id``.
    """
    normalized = str(kind).strip().lower()
    if normalized in {"shape", "geometry"}:
        return prepare_shape_optimization_problem(cfg=cfg, **kwargs)
    if normalized in {"parameterized_shape", "parameterized-shape", "parametric_shape", "parametric-shape"}:
        return prepare_parameterized_shape_optimization_problem(cfg=cfg, **kwargs)
    if normalized in {"material", "e", "youngs"}:
        return prepare_material_optimization_problem(cfg=cfg, **kwargs)
    raise ValueError(
        f"unsupported optimization kind {kind!r}; expected 'shape', 'parameterized_shape', or 'material'"
    )


def prepare_optimization_baseline_simulation(
    problem: OptimizationProblem,
) -> Any:
    """Return the cached or freshly solved baseline result for an optimization problem."""
    if isinstance(problem, ShapeOptimizationProblem):
        return prepare_shape_differentiable_simulation(problem)
    if isinstance(problem, ParameterizedShapeOptimizationProblem):
        return prepare_parameterized_shape_differentiable_simulation(problem)
    if isinstance(problem, ScalarMaterialOptimizationProblem):
        return prepare_material_differentiable_simulation(problem)
    raise TypeError(
        "expected ShapeOptimizationProblem, ParameterizedShapeOptimizationProblem, "
        "or ScalarMaterialOptimizationProblem; "
        f"got {type(problem).__name__}"
    )


def report_optimization_baseline(
    *,
    problem: OptimizationProblem,
    workspace,
    cfg,
    release_solver: bool = True,
    **report_kwargs: Any,
):
    """Report an optimization baseline result and release its solver reference.

    The optimization problem keeps its reusable solver. Only the temporary result
    returned for baseline reporting is released by default.
    """
    result = prepare_optimization_baseline_simulation(problem)
    try:
        return report_history_bundle(
            result=result,
            workspace=workspace,
            cfg=cfg,
            **report_kwargs,
        )
    finally:
        if release_solver and hasattr(result, "release_solver"):
            result.release_solver()


def make_optimizer(problem: OptimizationProblem, **kwargs: Any):
    """Create the default optimizer for a shape or material optimization problem."""
    if isinstance(problem, (ShapeOptimizationProblem, ParameterizedShapeOptimizationProblem)):
        return make_shape_optimizer(problem, **kwargs)
    if isinstance(problem, ScalarMaterialOptimizationProblem):
        return make_material_optimizer(problem, **kwargs)
    raise TypeError(
        "expected ShapeOptimizationProblem, ParameterizedShapeOptimizationProblem, "
        "or ScalarMaterialOptimizationProblem; "
        f"got {type(problem).__name__}"
    )


def run_optimization(
    problem: OptimizationProblem,
    *,
    steps: int,
    optimizer: Any,
    loss_fn: Any,
    workspace: Any = None,
    summary_path: Any = _DEFAULT,
    history_summary_path: Any = _DEFAULT,
    gradient_dir: Any = _DEFAULT,
    max_vertex_step: Any = _DEFAULT,
    **kwargs: Any,
):
    """Run shape or material optimization with type-specific output defaults."""
    workspace_path = Path(workspace) if workspace is not None else None

    if isinstance(problem, (ShapeOptimizationProblem, ParameterizedShapeOptimizationProblem)):
        if summary_path is _DEFAULT:
            summary_path = workspace_path / "shape_optimization_summary.txt" if workspace_path else None
        if history_summary_path is _DEFAULT:
            history_summary_path = workspace_path / "history_summary.txt" if workspace_path else None
        if gradient_dir is _DEFAULT:
            gradient_dir = workspace_path / "shape_gradients" if workspace_path else None
        if max_vertex_step is _DEFAULT:
            max_vertex_step = None if isinstance(problem, ParameterizedShapeOptimizationProblem) else 1e-4
        return run_shape_optimization(
            problem,
            steps=steps,
            optimizer=optimizer,
            loss_fn=loss_fn,
            summary_path=summary_path,
            history_summary_path=history_summary_path,
            gradient_dir=gradient_dir,
            max_vertex_step=max_vertex_step,
            **kwargs,
        )

    if isinstance(problem, ScalarMaterialOptimizationProblem):
        if gradient_dir is not _DEFAULT and gradient_dir is not None:
            raise ValueError("gradient_dir is only supported for shape optimization")
        if max_vertex_step is not _DEFAULT and max_vertex_step is not None:
            raise ValueError("max_vertex_step is only supported for shape optimization")
        if summary_path is _DEFAULT:
            summary_path = workspace_path / "material_optimization_summary.txt" if workspace_path else None
        if history_summary_path is _DEFAULT:
            history_summary_path = workspace_path / "history_summary.txt" if workspace_path else None
        return run_scalar_material_optimization(
            problem,
            steps=steps,
            optimizer=optimizer,
            loss_fn=loss_fn,
            summary_path=summary_path,
            history_summary_path=history_summary_path,
            **kwargs,
        )

    raise TypeError(
        "expected ShapeOptimizationProblem, ParameterizedShapeOptimizationProblem, "
        "or ScalarMaterialOptimizationProblem; "
        f"got {type(problem).__name__}"
    )


__all__ = [
    "make_optimizer",
    "OptimizationKind",
    "OptimizationProblem",
    "ParameterizedShapeOptimizationProblem",
    "prepare_optimization_baseline_simulation",
    "prepare_optimization_problem",
    "report_optimization_baseline",
    "run_optimization",
]
