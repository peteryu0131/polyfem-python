"""Small helpers for PyTorch-style shape optimization loops."""

# pyright: reportMissingImports=false

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Tuple, Union, overload

import numpy as np
import torch

from .objective_bridge import (
    SmoothTimeAggregationName,
    TimeAggregation,
    TimeAggregationName,
    make_von_mises_loss,
)
from .result_diff import DifferentiableResult
from .solve_diff import _console_log_level_from_settings, prepare_differentiable_simulation
from .summary import gradient_norm
from .torch_integration import PolyFEMFunction
from ..api.report import summarize_history_bundle
from ..api.runtime import format_history_summary


BaselineCallback = Callable[[DifferentiableResult], None]
LossOutput = Union[torch.Tensor, Tuple[torch.Tensor, Any]]
LossBuilder = Callable[[DifferentiableResult], LossOutput]


@dataclass
class ShapeOptimizationStep:
    """Compact record for one completed shape optimization step."""

    iteration: int
    loss: torch.Tensor
    state_col: Optional[int]
    objective_info: Any
    gradient: Optional[torch.Tensor]
    step_norm: float
    max_vertex_update: float
    step_scale: float = 1.0
    gradient_path: Optional[str] = None
    history_bundle: Optional[dict[str, Any]] = None


@dataclass
class ShapeOptimizationProblem:
    """Reusable array-mode payload for optimizing mesh vertices."""

    vertices: torch.nn.Parameter
    cells: torch.Tensor
    cfg: dict[str, Any]
    report_cfg: Any = None
    body_ids: Optional[np.ndarray] = None
    boundary_ids: Optional[np.ndarray] = None
    derivative_type: str = "shape"
    solver: Any = None
    solve_log_level: int = 2
    baseline_result: Optional[DifferentiableResult] = None

    def solve(self) -> DifferentiableResult:
        """Run one differentiable solve at the current design vertices."""
        if self.solver is not None:
            solutions = PolyFEMFunction.apply(
                self.solver,
                self.vertices,
                self.derivative_type,
                int(self.solve_log_level),
            )
            return DifferentiableResult(
                u=solutions,
                solver=self.solver,
                derivative_type=self.derivative_type,
                differentiable_params=["geometry"],
                vertices=self.vertices,
                meta={"_solve_settings": self.cfg},
            )

        return prepare_differentiable_simulation(
            V=self.vertices,
            C=self.cells,
            cfg=self.cfg,
            body_ids=self.body_ids,
            boundary_ids=self.boundary_ids,
            derivative_type=self.derivative_type,
        )

    def make_optimizer(self, *, name: str = "sgd", lr: float = 1e-6) -> torch.optim.Optimizer:
        """Create a PyTorch optimizer for the design vertices."""
        opt_name = str(name).strip().lower()
        if opt_name == "sgd":
            return torch.optim.SGD([self.vertices], lr=float(lr))
        if opt_name == "adam":
            return torch.optim.Adam([self.vertices], lr=float(lr))
        raise ValueError(f"unsupported optimizer {name!r}")

    def release_solver(self) -> None:
        """Drop the reusable C++ solver reference held by this problem."""
        if self.baseline_result is not None:
            self.baseline_result.release_solver()
            self.baseline_result = None
        self.solver = None

    def optimize(
        self,
        *,
        steps: int,
        optimizer: torch.optim.Optimizer,
        loss_fn: LossBuilder,
        max_vertex_step: Optional[float] = None,
        collect_history: bool = False,
    ) -> Iterator[ShapeOptimizationStep]:
        """Yield completed optimization steps for a user-supplied loss."""
        max_vertex_step_value = None if max_vertex_step is None else float(max_vertex_step)
        if max_vertex_step_value is not None and max_vertex_step_value <= 0:
            raise ValueError(f"max_vertex_step must be positive; got {max_vertex_step!r}")

        for iteration in range(int(steps)):
            optimizer.zero_grad(set_to_none=True)
            result = self.solve()
            try:
                loss_out = loss_fn(result)
                if isinstance(loss_out, tuple):
                    loss, objective_info = loss_out
                    if isinstance(objective_info, dict):
                        state_col_raw = objective_info.get("selected_state_col")
                    else:
                        state_col_raw = objective_info
                    state_col = int(state_col_raw) if state_col_raw is not None else None
                else:
                    loss = loss_out
                    objective_info = None
                    state_col = None

                loss.backward()
                gradient = (
                    None
                    if result.shape_gradient is None
                    else result.shape_gradient.detach().clone()
                )
                history_bundle = (
                    summarize_history_bundle(
                        result,
                        cfg=self.report_cfg if self.report_cfg is not None else self.cfg,
                    )
                    if collect_history
                    else None
                )

                vertices_before = self.vertices.detach().clone()
                optimizer.step()
                step_scale = 1.0
                with torch.no_grad():
                    update = self.vertices.detach() - vertices_before
                    per_vertex_update = torch.linalg.norm(update.reshape(update.shape[0], -1), dim=1)
                    max_vertex_update = float(per_vertex_update.max().cpu().item()) if per_vertex_update.numel() else 0.0
                    if (
                        max_vertex_step_value is not None
                        and max_vertex_update > max_vertex_step_value
                    ):
                        step_scale = max_vertex_step_value / max_vertex_update
                        self.vertices.copy_(vertices_before + update * step_scale)
                        update = self.vertices.detach() - vertices_before
                        per_vertex_update = torch.linalg.norm(update.reshape(update.shape[0], -1), dim=1)
                        max_vertex_update = (
                            float(per_vertex_update.max().cpu().item())
                            if per_vertex_update.numel()
                            else 0.0
                        )

                step_norm = float(
                    torch.linalg.norm(update)
                    .cpu()
                    .item()
                )

                yield ShapeOptimizationStep(
                    iteration=iteration,
                    loss=loss.detach(),
                    state_col=state_col,
                    objective_info=objective_info,
                    gradient=gradient,
                    step_norm=step_norm,
                    max_vertex_update=max_vertex_update,
                    step_scale=step_scale,
                    history_bundle=history_bundle,
                )
            finally:
                result.release_solver()


def _config_to_array_mode_dict(cfg: Any) -> dict[str, Any]:
    if hasattr(cfg, "to_dict"):
        out = cfg.to_dict()
    elif isinstance(cfg, dict):
        out = dict(cfg)
    else:
        raise ValueError("cfg must be a dict-like config or expose to_dict()")
    out.pop("geometry", None)
    out.pop("root_path", None)
    return out


def _extract_body_ids(mesh: Any) -> Optional[np.ndarray]:
    if not hasattr(mesh, "get_body_ids"):
        return None
    ids = np.asarray(mesh.get_body_ids(), dtype=np.int32).reshape(-1)
    return ids if ids.size else None


def _extract_boundary_ids(mesh: Any) -> Optional[np.ndarray]:
    if not (hasattr(mesh, "get_boundary_id") and hasattr(mesh, "n_boundary_elements")):
        return None
    n_boundary = int(mesh.n_boundary_elements())
    if n_boundary <= 0:
        return None
    return np.asarray(
        [mesh.get_boundary_id(i) for i in range(n_boundary)],
        dtype=np.int32,
    )


def make_shape_optimizer(
    problem: ShapeOptimizationProblem,
    *,
    name: str = "sgd",
    lr: float = 1e-6,
) -> torch.optim.Optimizer:
    """Create a PyTorch optimizer for a shape optimization problem."""
    return problem.make_optimizer(name=name, lr=lr)


def prepare_shape_differentiable_simulation(
    problem: ShapeOptimizationProblem,
) -> DifferentiableResult:
    """Run or return one baseline differentiable shape simulation.

    If ``prepare_shape_optimization_problem(...)`` cached the initial result,
    this returns that result without another solve. Otherwise it solves from
    the problem's current vertices.
    """
    if problem.baseline_result is not None:
        result = problem.baseline_result
        problem.baseline_result = None
        return result
    return problem.solve()


@overload
def make_von_mises_shape_loss(
    *,
    volume_selection: int = 1,
    time_aggregation: SmoothTimeAggregationName,
    smooth_max_sharpness: Optional[float] = None,
    power: int = 2,
    state: Optional[Union[str, int]] = "last",
) -> LossBuilder:
    ...


@overload
def make_von_mises_shape_loss(
    *,
    volume_selection: int = 1,
    time_aggregation: TimeAggregationName = "last",
    power: int = 2,
    state: Optional[Union[str, int]] = "last",
) -> LossBuilder:
    ...


@overload
def make_von_mises_shape_loss(
    *,
    power: int = 2,
    volume_selection: int = 1,
    state: Optional[Union[str, int]] = "last",
    time_aggregation: Optional[TimeAggregationName] = None,
    smooth_max_sharpness: Optional[float] = None,
    time_reduction: Optional[TimeAggregationName] = None,
    smooth_max_beta: Optional[float] = None,
) -> LossBuilder:
    ...


def make_von_mises_shape_loss(
    *,
    power: int = 2,
    volume_selection: int = 1,
    state: Optional[Union[str, int]] = "last",
    time_aggregation: Optional[TimeAggregation] = None,
    smooth_max_sharpness: Optional[float] = None,
    time_reduction: Optional[TimeAggregation] = None,
    smooth_max_beta: Optional[float] = None,
) -> LossBuilder:
    """Compatibility wrapper for a reusable von Mises loss builder.

    Prefer calling ``make_von_mises_loss(...)`` directly. This wrapper is kept
    so older shape-optimization examples continue to run.

    Common use:
    - ``time_aggregation="smooth_max"`` for a smooth worst-time-step objective
    - ``time_aggregation="max"`` for a hard worst-time-step objective
    - ``time_aggregation="mean"`` to average all time steps

    Args:
        power: Objective power. Defaults to ``2``.
        volume_selection: Body/volume id used by PolyFEM.
        state: Single state to use when ``time_aggregation`` is omitted.
        time_aggregation: How to combine per-time-step losses. Allowed values:
            ``"last"``, ``"final"``, ``"first"``, ``"max"``,
            ``"smooth_max"``, ``"mean"``, and ``"sum"``. Use ``"max"`` or
            ``"smooth_max"`` for impact-style worst-case objectives.
        smooth_max_sharpness: Only used with ``time_aggregation="smooth_max"``.
            Larger values behave more like hard max; omit it to auto-scale from
            the current loss magnitude.
    """
    def loss_fn(result: DifferentiableResult) -> LossOutput:
        return make_von_mises_loss(
            result=result,
            power=int(power),
            volume_selection=int(volume_selection),
            state=state,
            time_aggregation=time_aggregation,
            smooth_max_sharpness=smooth_max_sharpness,
            time_reduction=time_reduction,
            smooth_max_beta=smooth_max_beta,
        )

    return loss_fn


def format_shape_optimization_step(step: ShapeOptimizationStep) -> str:
    """Format one completed optimization step for console or text reports."""
    loss_value = float(step.loss.detach().cpu().item())
    objective_info = step.objective_info if isinstance(step.objective_info, dict) else {}
    lines = [
        f"iter {step.iteration}: state_col={step.state_col}",
        f"loss: {loss_value:.6e}",
        f"grad_norm: {gradient_norm(step.gradient):.6e}",
        f"step_norm: {step.step_norm:.6e}",
        f"max_vertex_update: {step.max_vertex_update:.6e}",
    ]
    if step.step_scale < 1.0:
        lines.append(f"step_scale: {step.step_scale:.6e}")
    if objective_info.get("time_aggregation") is not None:
        lines.append(f"time_aggregation: {objective_info['time_aggregation']}")
    if objective_info.get("selected_state_col") is not None:
        lines.append(f"selected_state_col: {objective_info['selected_state_col']}")
    if step.gradient_path is not None:
        lines.append(f"gradient_path: {step.gradient_path}")
    return "\n".join(lines)


def print_shape_optimization_step(step: ShapeOptimizationStep) -> None:
    """Print a compact summary for one completed optimization step."""
    print(format_shape_optimization_step(step))


def format_shape_optimization_history_summary(steps: list[ShapeOptimizationStep]) -> str:
    """Format per-iteration PolyFEM history summaries for shape optimization."""
    sections: list[str] = ["# Shape Optimization History Summary"]
    if not steps:
        sections.append("iterations: 0")
        return "\n".join(sections) + "\n"

    for step in steps:
        sections.extend(
            [
                "",
                f"## iter {step.iteration}",
                format_shape_optimization_step(step),
                "",
            ]
        )
        if step.history_bundle is None:
            sections.append("history_available: false")
            sections.append("note: history collection was not enabled")
        else:
            sections.append(format_history_summary(step.history_bundle).rstrip())
    return "\n".join(sections) + "\n"


def run_shape_optimization(
    problem: ShapeOptimizationProblem,
    *,
    steps: int,
    optimizer: torch.optim.Optimizer,
    loss_fn: LossBuilder,
    print_steps: bool = True,
    summary_path: Optional[Union[str, Path]] = None,
    gradient_dir: Optional[Union[str, Path]] = None,
    history_summary_path: Optional[Union[str, Path]] = None,
    max_vertex_step: Optional[float] = None,
    release_solver: bool = True,
) -> list[ShapeOptimizationStep]:
    """Run a complete shape optimization loop and return completed steps."""
    completed_steps: list[ShapeOptimizationStep] = []
    report_lines: list[str] = []
    grad_dir = Path(gradient_dir) if gradient_dir is not None else None
    if grad_dir is not None:
        grad_dir.mkdir(parents=True, exist_ok=True)

    def write_reports() -> None:
        if summary_path is not None:
            path = Path(summary_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            text = "\n\n".join(report_lines)
            path.write_text(text + ("\n" if text else ""), encoding="utf-8")

        if history_summary_path is not None:
            path = Path(history_summary_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                format_shape_optimization_history_summary(completed_steps),
                encoding="utf-8",
            )

    try:
        for step in problem.optimize(
            steps=steps,
            optimizer=optimizer,
            loss_fn=loss_fn,
            max_vertex_step=max_vertex_step,
            collect_history=history_summary_path is not None,
        ):
            if grad_dir is not None and step.gradient is not None:
                grad_path = grad_dir / f"shape_gradient_iter_{step.iteration:04d}.npy"
                np.save(grad_path, step.gradient.detach().cpu().numpy())
                step.gradient_path = str(grad_path)

            completed_steps.append(step)
            step_text = format_shape_optimization_step(step)
            report_lines.append(step_text)
            if print_steps:
                print(step_text)
            write_reports()
    finally:
        if completed_steps:
            write_reports()
        if release_solver:
            problem.release_solver()

    return completed_steps


def prepare_shape_optimization_problem(
    *,
    cfg: Any,
    derivative_type: str = "shape",
    initial_result: Optional[DifferentiableResult] = None,
    baseline_result: Optional[DifferentiableResult] = None,
    on_baseline_result: Optional[BaselineCallback] = None,
    keep_baseline_result: bool = True,
) -> ShapeOptimizationProblem:
    """Prepare a config-backed mesh for repeated vertex optimization solves.

    If ``initial_result`` is provided, it is reused instead of running another
    initial simulation. This lets callers report or inspect the initial result
    outside this helper, then pass the same result here without duplicate work.

    The initial result is kept by default so a problem-first flow can fetch it
    with ``prepare_shape_differentiable_simulation(problem)`` without repeating
    the first solve. Set ``keep_baseline_result=False`` when the caller has
    already consumed the initial result and wants to release that graph earlier.
    """
    if initial_result is not None and baseline_result is not None:
        raise ValueError("Use either initial_result or baseline_result, not both.")
    result = initial_result if initial_result is not None else baseline_result
    if result is None:
        result = prepare_differentiable_simulation(cfg=cfg, derivative_type=derivative_type)
    baseline_result_retained = False
    try:
        if on_baseline_result is not None:
            on_baseline_result(result)

        if result.solver is None:
            raise ValueError("initial_result must keep its solver until prepare_shape_optimization_problem() returns")

        solver = result.solver
        mesh = result.solver.mesh()
        settings = result.meta.get("_solve_settings")
        settings_dict = settings if isinstance(settings, dict) else _config_to_array_mode_dict(cfg)
        cells = torch.as_tensor(
            np.asarray(mesh.elements(), dtype=np.int32),
            dtype=torch.int32,
        )
        vertices = torch.nn.Parameter(result.vertices.detach().clone())
        problem = ShapeOptimizationProblem(
            vertices=vertices,
            cells=cells,
            cfg=_config_to_array_mode_dict(cfg),
            report_cfg=cfg,
            body_ids=_extract_body_ids(mesh),
            boundary_ids=_extract_boundary_ids(mesh),
            derivative_type=str(derivative_type),
            solver=solver,
            solve_log_level=_console_log_level_from_settings(settings_dict),
            baseline_result=result if keep_baseline_result else None,
        )
        baseline_result_retained = bool(keep_baseline_result)
        return problem
    finally:
        if not baseline_result_retained:
            result.solver = None
            result.u = result.u.detach()


__all__ = [
    "ShapeOptimizationProblem",
    "ShapeOptimizationStep",
    "make_shape_optimizer",
    "prepare_shape_differentiable_simulation",
    "make_von_mises_shape_loss",
    "format_shape_optimization_step",
    "format_shape_optimization_history_summary",
    "print_shape_optimization_step",
    "run_shape_optimization",
    "prepare_shape_optimization_problem",
]
