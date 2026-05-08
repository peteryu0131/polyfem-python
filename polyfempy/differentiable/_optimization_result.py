"""Stable result object for generic differentiable optimization runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class OptimizationRunResult:
    """Stable result object returned by ``run_optimization(..., return_result=True)``."""

    problem: Any
    steps: list[Any]
    workspace: Optional[Path] = None
    summary_path: Optional[Path] = None
    history_summary_path: Optional[Path] = None
    gradient_dir: Optional[Path] = None
    success: bool = True
    message: str = "completed"

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
        return _step_loss_float(self.final_step)

    @property
    def best_step(self) -> Any | None:
        """Completed step with the smallest loss, or ``None`` when unavailable."""
        candidates = [
            (loss, step)
            for step in self.steps
            if (loss := _step_loss_float(step)) is not None
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda item: item[0])[1]

    @property
    def best_loss(self) -> float | None:
        """Best loss across completed steps, or ``None`` when no loss exists."""
        return _step_loss_float(self.best_step)

    @property
    def best_iteration(self) -> int | None:
        """Iteration index for ``best_step``, if available."""
        step = self.best_step
        if step is None:
            return None
        iteration = getattr(step, "iteration", None)
        return None if iteration is None else int(iteration)

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
            "success": bool(self.success),
            "message": str(self.message),
            "optimization_steps": self.iterations,
            "final_iteration": None if step is None else getattr(step, "iteration", None),
            "final_loss": self.final_loss,
            "best_iteration": self.best_iteration,
            "best_loss": self.best_loss,
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


def _step_loss_float(step: Any) -> float | None:
    if step is None or not hasattr(step, "loss"):
        return None
    loss = step.loss
    if hasattr(loss, "detach"):
        return float(loss.detach().cpu().item())
    return float(loss)


def _completion_status(completed_steps: list[Any], requested_steps: int) -> tuple[bool, str]:
    requested = int(requested_steps)
    completed = len(completed_steps)
    if completed == requested:
        return True, f"completed {completed} optimization {_step_word(completed)}"
    return False, (
        f"completed {completed} of {requested} requested optimization "
        f"{_step_word(requested)}"
    )


def _step_word(count: int) -> str:
    return "step" if int(count) == 1 else "steps"


def _path_or_none(value: Any, default_sentinel: Any) -> Optional[Path]:
    if value is None or value is default_sentinel:
        return None
    return Path(value)


__all__ = ["OptimizationRunResult"]
