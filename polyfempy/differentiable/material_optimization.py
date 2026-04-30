"""Reusable helpers for scalar-Young's-modulus material experiments.

These helpers keep experiment scripts focused on their config and optimization
loop while sharing the common scalar-material problem setup and reporting.
Advanced probes and finite-difference checks live in ``material_diagnostics``.
"""

# pyright: reportMissingImports=false

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F

from .design import make_bounds_projector, make_parameter
from .material_config import (
    material_for_body,
    nu_from_material,
    other_material_for_body,
    youngs_from_material,
)
from .summary import gradient_norm
from .solve_diff import (
    _differentiable_config_and_settings,
    build_solver_from_settings,
    solve_differentiable_material_from_youngs,
    solver_body_slot_mask,
    youngs_value_to_internal,
)
from ..api.report import summarize_history_bundle
from ..api.runtime import format_history_summary


@dataclass
class ScalarMaterialOptimizationStep:
    """Compact record for one completed scalar-``E`` optimization step."""

    iteration: int
    E_value: float
    E_unit: str
    loss: torch.Tensor
    objective_info: Any
    gradient: Optional[torch.Tensor]
    history_bundle: Optional[dict[str, Any]] = None


@dataclass
class ScalarMaterialOptimizationProblem:
    """Reusable payload for optimizing one positive Young's modulus scalar."""

    cfg: dict[str, Any]
    solver: Any
    slot_mask: torch.Tensor
    E_unit: str
    nu: float
    other_E_internal: float
    other_nu: float
    report_cfg: Any = None
    solve_log_level: int = 3
    e_floor: float = 1.0
    log_E: Optional[torch.nn.Parameter] = None
    E_parameter: Optional[torch.nn.Parameter] = None
    project: Optional[Callable[[Sequence[torch.nn.Parameter]], None]] = None
    current_E: Optional[torch.Tensor] = None

    def solve(self) -> Any:
        """Run one differentiable solve at the current scalar ``E``."""
        if self.E_parameter is not None:
            self.current_E = self.E_parameter
        else:
            if self.log_E is None:
                raise ValueError("log_E or E_parameter must be initialized on this optimization problem")
            self.current_E = F.softplus(self.log_E) + float(self.e_floor)
        if self.current_E.requires_grad:
            self.current_E.retain_grad()
        solver_E = youngs_value_to_internal(
            self.current_E,
            pressure_unit=self.E_unit,
            solver_settings=self.cfg,
        )
        result = solve_differentiable_material_from_youngs(
            solver=self.solver,
            E=solver_E,
            nu=float(self.nu),
            slot_mask=self.slot_mask,
            other_E=float(self.other_E_internal),
            other_nu=float(self.other_nu),
            solve_log_level=int(self.solve_log_level),
        )
        result.meta.setdefault("_solve_settings", self.cfg)
        return result

    def make_optimizer(self, *, name: str = "adam", lr: float = 1e-2) -> torch.optim.Optimizer:
        """Create a PyTorch optimizer for the scalar design variable."""
        params = self.torch_parameters()
        if not params:
            raise ValueError("no scalar material design parameter is initialized")

        opt_name = str(name).strip().lower()
        if opt_name == "adam":
            return torch.optim.Adam(params, lr=float(lr))
        if opt_name == "sgd":
            return torch.optim.SGD(params, lr=float(lr))
        raise ValueError(f"unsupported optimizer {name!r}")

    def torch_parameters(self) -> list[torch.nn.Parameter]:
        """Return the user-optimized scalar material parameter."""
        if self.E_parameter is not None:
            return [self.E_parameter]
        if self.log_E is not None:
            return [self.log_E]
        return []

    def project_(self) -> None:
        """Apply optional post-step projection such as physical E bounds."""
        if self.project is not None:
            self.project(self.torch_parameters())

    def release_solver(self) -> None:
        """Drop the reusable C++ solver reference held by this problem."""
        self.solver = None

    def optimize(
        self,
        *,
        steps: int,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable[[Any], Any],
        collect_history: bool = False,
    ) -> Iterator[ScalarMaterialOptimizationStep]:
        """Yield completed optimization steps for a user-supplied loss."""
        for iteration in range(int(steps)):
            optimizer.zero_grad(set_to_none=True)
            result = self.solve()
            try:
                loss_out = loss_fn(result)
                if isinstance(loss_out, tuple):
                    loss, objective_info = loss_out
                else:
                    loss, objective_info = loss_out, None

                loss.backward()
                gradient = None
                if self.current_E is not None and self.current_E.grad is not None:
                    gradient = self.current_E.grad.detach().clone()
                E_value = (
                    float(self.current_E.detach().item())
                    if self.current_E is not None
                    else float("nan")
                )
                history_bundle = (
                    summarize_history_bundle(
                        result,
                        cfg=self.report_cfg if self.report_cfg is not None else self.cfg,
                    )
                    if collect_history
                    else None
                )
                optimizer.step()
                self.project_()

                yield ScalarMaterialOptimizationStep(
                    iteration=iteration,
                    E_value=E_value,
                    E_unit=self.E_unit,
                    loss=loss.detach(),
                    objective_info=objective_info,
                    gradient=gradient,
                    history_bundle=history_bundle,
                )
            finally:
                result.release_solver()


def prepare_material_optimization_problem(
    *,
    cfg: Any,
    body_id: int,
    initial_E_value: Optional[float] = None,
    E_unit: Optional[str] = None,
    nu: Optional[float] = None,
    other_body_id: Optional[int] = None,
    other_E_value: Optional[float] = None,
    other_E_unit: Optional[str] = None,
    other_nu: Optional[float] = None,
    solve_log_level: int = 3,
    e_floor: float = 1.0,
    root_path: Optional[str] = None,
) -> ScalarMaterialOptimizationProblem:
    """Prepare a config-backed reusable solver for material ``E`` optimization.

    E/nu values are inferred from ``cfg.materials`` by ``body_id`` when omitted.
    Explicit arguments remain available as overrides for experiments that want
    to start from values different from the config. By default, repeated
    optimization solves use warning-level solver output and keep ``E`` above
    ``e_floor=1.0`` in the selected ``E_unit``.
    """
    _, _, settings, _ = _differentiable_config_and_settings(
        cfg,
        root_path=root_path,
    )
    material = material_for_body(settings, body_id=int(body_id))
    cfg_E_value, cfg_E_unit = youngs_from_material(material)
    cfg_nu = nu_from_material(material)

    initial_E_value = cfg_E_value if initial_E_value is None else float(initial_E_value)
    E_unit = cfg_E_unit if E_unit is None else str(E_unit)
    nu = cfg_nu if nu is None else float(nu)

    other_material = other_material_for_body(
        settings,
        body_id=int(body_id),
        other_body_id=other_body_id,
    )
    if other_E_value is None or other_nu is None:
        if other_material is None:
            other_cfg_E_value, other_cfg_E_unit, other_cfg_nu = (
                initial_E_value,
                E_unit,
                nu,
            )
        else:
            other_cfg_E_value, other_cfg_E_unit = youngs_from_material(other_material)
            other_cfg_nu = nu_from_material(other_material)
        other_E_value = (
            other_cfg_E_value
            if other_E_value is None
            else float(other_E_value)
        )
        other_E_unit = (
            other_cfg_E_unit
            if other_E_unit is None
            else str(other_E_unit)
        )
        other_nu = other_cfg_nu if other_nu is None else float(other_nu)
    elif other_E_unit is None:
        other_E_unit = E_unit

    solver = build_solver_from_settings(settings)
    slot_mask = solver_body_slot_mask(solver, body_id=int(body_id))
    other_E_internal = float(
        youngs_value_to_internal(
            float(other_E_value),
            pressure_unit=other_E_unit,
            solver_settings=settings,
        )
    )
    init_log_E = np.log(np.expm1(max(float(initial_E_value) - float(e_floor), 1e-12)))
    log_E = torch.nn.Parameter(torch.tensor(init_log_E, dtype=torch.get_default_dtype()))
    return ScalarMaterialOptimizationProblem(
        cfg=settings,
        solver=solver,
        slot_mask=slot_mask,
        E_unit=E_unit,
        nu=float(nu),
        other_E_internal=other_E_internal,
        other_nu=float(other_nu),
        report_cfg=cfg,
        solve_log_level=int(solve_log_level),
        e_floor=float(e_floor),
        log_E=log_E,
    )


def prepare_scalar_youngs_material_problem(
    *,
    cfg: Any,
    body_id: int,
    E_parameter: Optional[torch.nn.Parameter] = None,
    parameter_name: str = "E",
    bounds: Optional[Sequence[Optional[float]]] = None,
    initial_E_value: Optional[float] = None,
    E_unit: Optional[str] = None,
    nu: Optional[float] = None,
    other_body_id: Optional[int] = None,
    other_E_value: Optional[float] = None,
    other_E_unit: Optional[str] = None,
    other_nu: Optional[float] = None,
    solve_log_level: int = 3,
    e_floor: float = 1.0,
    root_path: Optional[str] = None,
) -> ScalarMaterialOptimizationProblem:
    """Prepare scalar Young's-modulus optimization with a physical ``E`` parameter.

    This is the material-side counterpart to ``make_parameter(...)`` +
    ``prepare_parameterized_shape_problem(...)`` for the common scalar-E case.
    The optimized parameter is the user-facing Young's modulus in ``E_unit``;
    the lower-level solve still converts that value to Lamé tensors internally.

    The older ``prepare_material_optimization_problem(...)`` path remains
    available and optimizes an unconstrained ``log_E`` through a softplus.
    """
    problem = prepare_material_optimization_problem(
        cfg=cfg,
        body_id=body_id,
        initial_E_value=initial_E_value,
        E_unit=E_unit,
        nu=nu,
        other_body_id=other_body_id,
        other_E_value=other_E_value,
        other_E_unit=other_E_unit,
        other_nu=other_nu,
        solve_log_level=solve_log_level,
        e_floor=e_floor,
        root_path=root_path,
    )

    if E_parameter is None:
        if problem.log_E is None:
            raise ValueError("internal log_E was not initialized while preparing scalar material problem")
        inferred_E = float((F.softplus(problem.log_E.detach()) + float(problem.e_floor)).cpu().item())
        E_parameter = make_parameter(
            parameter_name,
            inferred_E,
            bounds=bounds if bounds is not None else (float(e_floor), None),
            dtype=torch.get_default_dtype(),
        )
    else:
        if not isinstance(E_parameter, torch.nn.Parameter):
            raise TypeError(
                "E_parameter must be a torch.nn.Parameter; "
                f"got {type(E_parameter).__name__}"
            )
        if getattr(E_parameter, "_polyfem_design_name", None) is None:
            E_parameter._polyfem_design_name = str(parameter_name)  # type: ignore[attr-defined]
        if bounds is not None:
            if len(bounds) != 2:
                raise ValueError(f"bounds must contain exactly two entries, got {len(bounds)}")
            E_parameter._polyfem_bounds = (  # type: ignore[attr-defined]
                None if bounds[0] is None else float(bounds[0]),
                None if bounds[1] is None else float(bounds[1]),
            )

    problem.log_E = None
    problem.E_parameter = E_parameter
    problem.project = make_bounds_projector([E_parameter])
    return problem


def prepare_material_differentiable_simulation(
    problem: ScalarMaterialOptimizationProblem,
) -> Any:
    """Run one differentiable material simulation from a prepared problem.

    This mirrors ``prepare_differentiable_simulation(...)`` at the experiment
    level, but uses the material problem's current scalar E design variable
    instead of treating mesh vertices as the differentiable input.
    """
    return problem.solve()


def make_material_optimizer(
    problem: ScalarMaterialOptimizationProblem,
    *,
    name: str = "adam",
    lr: float = 1e-2,
) -> torch.optim.Optimizer:
    """Create a PyTorch optimizer for a scalar material optimization problem."""
    return problem.make_optimizer(name=name, lr=lr)


def format_scalar_material_optimization_step(step: ScalarMaterialOptimizationStep) -> str:
    """Format one completed scalar material optimization step."""
    loss_value = float(step.loss.detach().cpu().item())
    objective_info = step.objective_info if isinstance(step.objective_info, dict) else {}
    unit_suffix = f"_{step.E_unit}" if step.E_unit else ""
    lines = [
        f"iter {step.iteration}: E{unit_suffix}={step.E_value:.6e}",
        f"loss: {loss_value:.6e}",
        f"grad_norm: {gradient_norm(step.gradient):.6e}",
    ]
    if objective_info.get("resolved_objective_name") is not None:
        lines.append(f"objective: {objective_info['resolved_objective_name']}")
    if objective_info.get("time_aggregation") is not None:
        lines.append(f"time_aggregation: {objective_info['time_aggregation']}")
    if objective_info.get("selected_state_col") is not None:
        lines.append(f"selected_state_col: {objective_info['selected_state_col']}")
    return "\n".join(lines)


def format_scalar_material_optimization_history_summary(
    steps: list[ScalarMaterialOptimizationStep],
) -> str:
    """Format per-iteration PolyFEM history summaries for scalar material optimization."""
    sections: list[str] = ["# Scalar Material Optimization History Summary"]
    if not steps:
        sections.append("iterations: 0")
        return "\n".join(sections) + "\n"

    for step in steps:
        sections.extend(
            [
                "",
                f"## iter {step.iteration}",
                format_scalar_material_optimization_step(step),
                "",
            ]
        )
        if step.history_bundle is None:
            sections.append("history_available: false")
            sections.append("note: history collection was not enabled")
        else:
            sections.append(format_history_summary(step.history_bundle).rstrip())
    return "\n".join(sections) + "\n"


def run_scalar_material_optimization(
    problem: ScalarMaterialOptimizationProblem,
    *,
    steps: int,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[Any], Any],
    print_steps: bool = True,
    preferred_objective_name: Optional[str] = None,
    summary_path: Optional[Union[str, Path]] = None,
    history_summary_path: Optional[Union[str, Path]] = None,
    release_solver: bool = True,
) -> list[ScalarMaterialOptimizationStep]:
    """Run a complete scalar material optimization loop and return completed steps."""
    completed_steps: list[ScalarMaterialOptimizationStep] = []
    report_lines: list[str] = []

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
                format_scalar_material_optimization_history_summary(completed_steps),
                encoding="utf-8",
            )

    try:
        for step in problem.optimize(
            steps=steps,
            optimizer=optimizer,
            loss_fn=loss_fn,
            collect_history=history_summary_path is not None,
        ):
            objective_info = step.objective_info if isinstance(step.objective_info, dict) else {}
            resolved_objective_name = str(objective_info.get("resolved_objective_name", "unknown"))
            step_text = format_scalar_material_optimization_step(step)
            report_lines.append(step_text)
            if print_steps:
                print(step_text)
                if (
                    step.iteration == 0
                    and preferred_objective_name is not None
                    and resolved_objective_name != str(preferred_objective_name)
                ):
                    print(
                        f"objective fallback: preferred {preferred_objective_name!r} is not available "
                        f"in the loaded extension, so using legacy {resolved_objective_name!r}"
                    )
            completed_steps.append(step)
            write_reports()
    finally:
        if completed_steps:
            write_reports()
        if release_solver:
            problem.release_solver()

    return completed_steps


__all__ = [
    "ScalarMaterialOptimizationProblem",
    "ScalarMaterialOptimizationStep",
    "format_scalar_material_optimization_history_summary",
    "format_scalar_material_optimization_step",
    "make_material_optimizer",
    "prepare_material_differentiable_simulation",
    "prepare_material_optimization_problem",
    "prepare_scalar_youngs_material_problem",
    "run_scalar_material_optimization",
]
