"""Runtime helpers for prepared optimization problems."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Union, overload

from ..api.runtime import report_history_bundle
from ._optimization_result import (
    OptimizationRunResult,
    _completion_status,
    _path_or_none,
)
from .material_optimization import (
    ScalarMaterialOptimizationProblem,
    make_material_optimizer,
    prepare_material_differentiable_simulation,
    run_scalar_material_optimization,
)
from .shape_optimization import (
    ParameterizedShapeOptimizationProblem,
    ShapeOptimizationProblem,
    make_shape_optimizer,
    prepare_parameterized_shape_differentiable_simulation,
    prepare_shape_differentiable_simulation,
    run_shape_optimization,
)


OptimizationProblem = Union[
    ShapeOptimizationProblem,
    ParameterizedShapeOptimizationProblem,
    ScalarMaterialOptimizationProblem,
]
_DEFAULT = object()


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
    raise TypeError(_problem_type_error(problem))


def report_optimization_baseline(
    *,
    problem: OptimizationProblem,
    workspace,
    cfg,
    release_solver: bool = True,
    **report_kwargs: Any,
):
    """Report an optimization baseline result and release its solver reference."""
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
    """Create the default optimizer for a prepared optimization problem."""
    if isinstance(problem, (ShapeOptimizationProblem, ParameterizedShapeOptimizationProblem)):
        return make_shape_optimizer(problem, **kwargs)
    if isinstance(problem, ScalarMaterialOptimizationProblem):
        return make_material_optimizer(problem, **kwargs)
    raise TypeError(_problem_type_error(problem))


@overload
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
    return_result: Literal[False] = False,
    **kwargs: Any,
) -> list[Any]:
    ...


@overload
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
    return_result: Literal[True],
    **kwargs: Any,
) -> OptimizationRunResult:
    ...


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
    return_result: bool = False,
    **kwargs: Any,
):
    """Run shape or material optimization with type-specific output defaults.

    By default this returns the legacy list of completed step objects. Pass
    ``return_result=True`` to get a stable ``OptimizationRunResult`` wrapper
    with ``.steps``, ``.final_step``, ``.final_loss``, and ``.summary()``.
    """
    if isinstance(problem, (ShapeOptimizationProblem, ParameterizedShapeOptimizationProblem)):
        return _run_shape_problem(
            problem,
            steps=steps,
            optimizer=optimizer,
            loss_fn=loss_fn,
            workspace=workspace,
            summary_path=summary_path,
            history_summary_path=history_summary_path,
            gradient_dir=gradient_dir,
            max_vertex_step=max_vertex_step,
            return_result=return_result,
            **kwargs,
        )

    if isinstance(problem, ScalarMaterialOptimizationProblem):
        return _run_material_problem(
            problem,
            steps=steps,
            optimizer=optimizer,
            loss_fn=loss_fn,
            workspace=workspace,
            summary_path=summary_path,
            history_summary_path=history_summary_path,
            gradient_dir=gradient_dir,
            max_vertex_step=max_vertex_step,
            return_result=return_result,
            **kwargs,
        )

    raise TypeError(_problem_type_error(problem))


def _run_shape_problem(
    problem: Union[ShapeOptimizationProblem, ParameterizedShapeOptimizationProblem],
    *,
    steps: int,
    optimizer: Any,
    loss_fn: Any,
    workspace: Any,
    summary_path: Any,
    history_summary_path: Any,
    gradient_dir: Any,
    max_vertex_step: Any,
    return_result: bool,
    **kwargs: Any,
):
    workspace_path = Path(workspace) if workspace is not None else None
    if summary_path is _DEFAULT:
        summary_path = workspace_path / "shape_optimization_summary.txt" if workspace_path else None
    if history_summary_path is _DEFAULT:
        history_summary_path = workspace_path / "history_summary.txt" if workspace_path else None
    if gradient_dir is _DEFAULT:
        gradient_dir = workspace_path / "shape_gradients" if workspace_path else None
    if max_vertex_step is _DEFAULT:
        max_vertex_step = None if isinstance(problem, ParameterizedShapeOptimizationProblem) else 1e-4
    completed_steps = run_shape_optimization(
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
    if not return_result:
        return completed_steps
    success, message = _completion_status(completed_steps, steps)
    return OptimizationRunResult(
        problem=problem,
        steps=completed_steps,
        workspace=workspace_path,
        summary_path=_path_or_none(summary_path, _DEFAULT),
        history_summary_path=_path_or_none(history_summary_path, _DEFAULT),
        gradient_dir=_path_or_none(gradient_dir, _DEFAULT),
        success=success,
        message=message,
    )


def _run_material_problem(
    problem: ScalarMaterialOptimizationProblem,
    *,
    steps: int,
    optimizer: Any,
    loss_fn: Any,
    workspace: Any,
    summary_path: Any,
    history_summary_path: Any,
    gradient_dir: Any,
    max_vertex_step: Any,
    return_result: bool,
    **kwargs: Any,
):
    if gradient_dir is not _DEFAULT and gradient_dir is not None:
        raise ValueError("gradient_dir is only supported for shape optimization")
    if max_vertex_step is not _DEFAULT and max_vertex_step is not None:
        raise ValueError("max_vertex_step is only supported for shape optimization")

    workspace_path = Path(workspace) if workspace is not None else None
    if summary_path is _DEFAULT:
        summary_path = workspace_path / "material_optimization_summary.txt" if workspace_path else None
    if history_summary_path is _DEFAULT:
        history_summary_path = workspace_path / "history_summary.txt" if workspace_path else None
    completed_steps = run_scalar_material_optimization(
        problem,
        steps=steps,
        optimizer=optimizer,
        loss_fn=loss_fn,
        summary_path=summary_path,
        history_summary_path=history_summary_path,
        **kwargs,
    )
    if not return_result:
        return completed_steps
    success, message = _completion_status(completed_steps, steps)
    return OptimizationRunResult(
        problem=problem,
        steps=completed_steps,
        workspace=workspace_path,
        summary_path=_path_or_none(summary_path, _DEFAULT),
        history_summary_path=_path_or_none(history_summary_path, _DEFAULT),
        gradient_dir=None,
        success=success,
        message=message,
    )


def _problem_type_error(problem: Any) -> str:
    return (
        "expected ShapeOptimizationProblem, ParameterizedShapeOptimizationProblem, "
        "or ScalarMaterialOptimizationProblem; "
        f"got {type(problem).__name__}"
    )


__all__ = [
    "OptimizationProblem",
    "OptimizationRunResult",
    "make_optimizer",
    "prepare_optimization_baseline_simulation",
    "report_optimization_baseline",
    "run_optimization",
]
