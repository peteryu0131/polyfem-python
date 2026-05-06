"""Small helpers for PyTorch-style shape optimization loops."""

# pyright: reportMissingImports=false

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Union, overload

import numpy as np
import torch

from .cpp_ext import get_cpp_polyfempy
from .design import (
    ParameterizedVertexDesign,
    make_bounds_projector,
    make_named_parameter_map,
    normalize_design_parameters,
)
from .objective_bridge import (
    SmoothTimeAggregationName,
    TimeAggregation,
    TimeAggregationName,
    make_von_mises_loss,
)
from .shape_problem import (
    LossBuilder,
    LossOutput,
    ParameterizedShapeOptimizationProblem,
    ShapeOptimizationProblem,
    ShapeOptimizationStep,
)
from .result_diff import DifferentiableResult
from .solve_diff import (
    _console_log_level_from_settings,
    _differentiable_config_and_settings,
    _geometry_uses_only_absolute_mesh_paths,
    _solver_set_log_level_off,
    prepare_differentiable_simulation,
)
from .summary import gradient_norm
from ..api.runtime import format_history_summary


BaselineCallback = Callable[[DifferentiableResult], None]


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


def _load_parameterized_shape_mesh_from_cfg(
    *,
    cfg: Any,
    quiet_polyfem_setup: bool = True,
) -> tuple[Any, dict[str, Any], torch.Tensor, torch.Tensor, Optional[np.ndarray], Optional[np.ndarray], int]:
    """Load the fixed-topology mesh for parameterized shape without running a solve."""
    config, _, settings, _ = _differentiable_config_and_settings(cfg)
    if not settings.get("geometry"):
        raise ValueError(
            "parameterized_shape currently expects a config-backed mesh. "
            "Use a config with geometry mesh paths so the fixed topology can be loaded."
        )
    if settings.get("geometry") and not settings.get("root_path") and not _geometry_uses_only_absolute_mesh_paths(settings):
        raise ValueError(
            "parameterized_shape requires root_path for relative mesh paths. "
            "Pass cfg as a JSON path, set cfg.extras['_root_path'], or use absolute mesh paths."
        )

    pf = get_cpp_polyfempy()
    solver = pf.Solver()
    if quiet_polyfem_setup:
        _solver_set_log_level_off(solver)
    solver.set_settings(json.dumps(settings), strict_validation=False)
    solver.load_mesh_from_settings()
    if hasattr(solver, "build_basis"):
        solver.build_basis()
    mesh = solver.mesh()
    vertices = torch.as_tensor(np.asarray(mesh.vertices(), dtype=np.float64), dtype=torch.get_default_dtype())
    cells = torch.as_tensor(np.asarray(mesh.elements(), dtype=np.int32), dtype=torch.int32)
    return (
        solver,
        settings,
        vertices,
        cells,
        _extract_body_ids(mesh),
        _extract_boundary_ids(mesh),
        _console_log_level_from_settings(settings),
    )


def make_shape_optimizer(
    problem: Union[ShapeOptimizationProblem, ParameterizedShapeOptimizationProblem],
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


def prepare_parameterized_shape_differentiable_simulation(
    problem: ParameterizedShapeOptimizationProblem,
) -> DifferentiableResult:
    """Run one differentiable parameterized-shape simulation."""
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
    problem: Union[ShapeOptimizationProblem, ParameterizedShapeOptimizationProblem],
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


def prepare_parameterized_shape_optimization_problem(
    *,
    cfg: Any,
    parameters: Optional[Sequence[torch.nn.Parameter]] = None,
    vertex_map: Optional[Callable[..., torch.Tensor]] = None,
    geometry: Any = None,
    design: Optional[ParameterizedVertexDesign] = None,
    parameter_map: Optional[Callable[[Sequence[torch.nn.Parameter]], Any]] = None,
    project: Optional[Callable[[Sequence[torch.nn.Parameter]], None]] = None,
    context: Any = None,
    differentiable_params: Optional[list[str]] = None,
    derivative_type: str = "shape",
    quiet_polyfem_setup: bool = True,
) -> ParameterizedShapeOptimizationProblem:
    """Advanced builder for a user-parameterized shape problem.

    Most user scripts should call ``prepare_parameterized_shape_problem(...)``
    or ``prepare_optimization_problem(kind="parameterized_shape", ...)``. This
    lower-level helper stays available for callers that already own a
    ``ParameterizedVertexDesign`` or need full control over ``parameter_map``,
    ``project``, or ``geometry``.

    The user supplies either:

    - ``parameters`` + ``vertex_map(design_value, base_vertices)``
    - ``parameters`` + ``vertex_map(design_value, base_vertices, context)``
    - or a ``geometry`` module/callable whose ``forward`` returns vertices.

    The returned vertices must keep the same shape and fixed connectivity as
    the mesh loaded from ``cfg``.
    """
    (
        solver,
        settings,
        base_vertices,
        cells,
        body_ids,
        boundary_ids,
        solve_log_level,
    ) = _load_parameterized_shape_mesh_from_cfg(
        cfg=cfg,
        quiet_polyfem_setup=quiet_polyfem_setup,
    )

    if design is None:
        design = ParameterizedVertexDesign(
            parameters=parameters,
            vertex_map=vertex_map,
            base_vertices=base_vertices,
            parameter_map=parameter_map,
            project=project,
            context=context,
            geometry=geometry,
            differentiable_params=differentiable_params,
        )
    else:
        if design.base_vertices is None:
            design.base_vertices = base_vertices
        if context is not None and design.context is None:
            design.context = context

    return ParameterizedShapeOptimizationProblem(
        design=design,
        cells=cells,
        cfg=settings,
        report_cfg=cfg,
        body_ids=body_ids,
        boundary_ids=boundary_ids,
        derivative_type=str(derivative_type),
        solver=solver,
        solve_log_level=int(solve_log_level),
        reference_vertices=base_vertices,
    )


def prepare_parameterized_shape_problem(
    *,
    cfg: Any,
    parameters: Optional[Sequence[torch.nn.Parameter]] = None,
    vertex_map: Optional[Callable[..., torch.Tensor]] = None,
    parameter_names: Optional[Sequence[str]] = None,
    bounds: Optional[Mapping[str, Sequence[Optional[float]]]] = None,
    context: Any = None,
    parameter_map: Optional[Callable[[Sequence[torch.nn.Parameter]], Any]] = None,
    project: Optional[Callable[[Sequence[torch.nn.Parameter]], None]] = None,
    differentiable_params: Optional[list[str]] = None,
    geometry: Any = None,
    design: Optional[ParameterizedVertexDesign] = None,
    derivative_type: str = "shape",
    quiet_polyfem_setup: bool = True,
) -> ParameterizedShapeOptimizationProblem:
    """Prepare a named-parameter shape problem with a user vertex map.

    This is the user-friendly wrapper around the lower-level
    ``prepare_parameterized_shape_optimization_problem(...)``. It expects
    plain ``torch.nn.Parameter`` objects plus explicit ``parameter_names``.
    The ``vertex_map`` receives a dictionary by default:

    ``vertex_map({"h": h, "theta": theta}, base_vertices)``.

    ``context`` is optional. If the map accepts a third argument, the wrapper
    passes this context so the map may cache non-differentiable helper data such
    as masks. If the map only accepts two arguments, no context is passed.

    Advanced callers may still pass ``parameter_map``, ``project``, ``geometry``,
    or a prebuilt ``ParameterizedVertexDesign``. Those routes are kept here so
    the public dispatcher can stay thin without breaking existing experiments.
    """
    if design is not None or geometry is not None:
        if parameter_names is not None or bounds is not None:
            raise ValueError(
                "parameter_names and bounds are only applied when using "
                "parameters + vertex_map directly"
            )
        return prepare_parameterized_shape_optimization_problem(
            cfg=cfg,
            parameters=parameters,
            vertex_map=vertex_map,
            geometry=geometry,
            design=design,
            parameter_map=parameter_map,
            project=project,
            context=context,
            differentiable_params=differentiable_params,
            derivative_type=derivative_type,
            quiet_polyfem_setup=quiet_polyfem_setup,
        )

    if parameters is None:
        raise ValueError(
            "prepare_parameterized_shape_problem requires parameters when "
            "geometry/design is not provided"
        )
    if vertex_map is None:
        raise ValueError(
            "prepare_parameterized_shape_problem requires vertex_map when "
            "geometry/design is not provided"
        )

    torch_params, names, _ = normalize_design_parameters(
        parameters,
        parameter_names=parameter_names,
        bounds=bounds,
    )
    if parameter_map is None:
        parameter_map = make_named_parameter_map(
            torch_params,
            parameter_names=names,
        )
    if project is None:
        project = make_bounds_projector(
            torch_params,
            parameter_names=names,
            bounds=bounds,
        )
    if differentiable_params is None:
        differentiable_params = names

    return prepare_parameterized_shape_optimization_problem(
        cfg=cfg,
        parameters=torch_params,
        vertex_map=vertex_map,
        parameter_map=parameter_map,
        project=project,
        context={} if context is None else context,
        differentiable_params=differentiable_params,
        derivative_type=derivative_type,
        quiet_polyfem_setup=quiet_polyfem_setup,
    )


__all__ = [
    "ShapeOptimizationProblem",
    "ParameterizedShapeOptimizationProblem",
    "ShapeOptimizationStep",
    "make_shape_optimizer",
    "prepare_shape_differentiable_simulation",
    "prepare_parameterized_shape_differentiable_simulation",
    "make_von_mises_shape_loss",
    "format_shape_optimization_step",
    "format_shape_optimization_history_summary",
    "print_shape_optimization_step",
    "run_shape_optimization",
    "prepare_shape_optimization_problem",
    "prepare_parameterized_shape_problem",
    "prepare_parameterized_shape_optimization_problem",
]
