"""Runtime helpers for prepared optimization problems."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Union, overload

from ..api.runtime import report_history_bundle
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


@dataclass
class OptimizationRunResult:
    """Stable result object returned by ``run_optimization(..., return_result=True)``."""

    problem: OptimizationProblem
    steps: list[Any]
    workspace: Optional[Path] = None
    summary_path: Optional[Path] = None
    history_summary_path: Optional[Path] = None
    gradient_dir: Optional[Path] = None

    @property
    def iterations(self) -> int:
        """Number of completed optimization steps."""
        return len(self.steps)

    @property
    def final_step(self) -> Any | None:
        """Last completed step, or ``None`` when no step ran."""
        return self.steps[-1] if self.steps else None

    @property
    def final_loss(self) -> float | None:
        """Final loss as a Python float, or ``None`` when no step ran."""
        step = self.final_step
        if step is None or not hasattr(step, "loss"):
            return None
        loss = step.loss
        if hasattr(loss, "detach"):
            return float(loss.detach().cpu().item())
        return float(loss)

    @property
    def final_gradient(self) -> Any | None:
        """Gradient stored on the last completed step, if the step type has one."""
        step = self.final_step
        return None if step is None else getattr(step, "gradient", None)

    def summary(self) -> dict[str, Any]:
        """Return a JSON-friendly summary for scripts and notebooks."""
        step = self.final_step
        out: dict[str, Any] = {
            "problem_type": type(self.problem).__name__,
            "optimization_steps": self.iterations,
            "final_iteration": None if step is None else getattr(step, "iteration", None),
            "final_loss": self.final_loss,
            "workspace": None if self.workspace is None else str(self.workspace),
            "summary_path": None if self.summary_path is None else str(self.summary_path),
            "history_summary_path": None
            if self.history_summary_path is None
            else str(self.history_summary_path),
            "gradient_dir": None if self.gradient_dir is None else str(self.gradient_dir),
        }
        if step is not None:
            if hasattr(step, "step_norm"):
                out["final_step_norm"] = float(step.step_norm)
            if hasattr(step, "max_vertex_update"):
                out["final_max_vertex_update"] = float(step.max_vertex_update)
            if getattr(step, "gradient_path", None) is not None:
                out["final_gradient_path"] = str(step.gradient_path)
            if hasattr(step, "E_value"):
                out["final_E_value"] = float(step.E_value)
                out["final_E_unit"] = str(step.E_unit)
        return out


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
    return OptimizationRunResult(
        problem=problem,
        steps=completed_steps,
        workspace=workspace_path,
        summary_path=_path_or_none(summary_path),
        history_summary_path=_path_or_none(history_summary_path),
        gradient_dir=_path_or_none(gradient_dir),
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
    return OptimizationRunResult(
        problem=problem,
        steps=completed_steps,
        workspace=workspace_path,
        summary_path=_path_or_none(summary_path),
        history_summary_path=_path_or_none(history_summary_path),
        gradient_dir=None,
    )


def _path_or_none(value: Any) -> Optional[Path]:
    if value is None or value is _DEFAULT:
        return None
    return Path(value)


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
