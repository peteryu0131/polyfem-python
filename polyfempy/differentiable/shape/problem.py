"""Shape optimization problem objects.

This module owns the reusable PyTorch problem payloads. Higher-level helpers
for preparing problems, building losses, formatting output, and running reports
stay in ``shape.optimization``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional, Tuple, Union

import numpy as np
import torch

from ..design import ParameterizedVertexDesign, parameter_name
from ..optimization.optimizers import make_torch_optimizer
from ..runtime.autograd import PolyFEMFunction
from ..runtime.result import DifferentiableResult
from ..runtime.solve import prepare_differentiable_simulation
from ...api.report import summarize_history_bundle


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
    parameter_values_before: Optional[dict[str, torch.Tensor]] = None
    parameter_values_after: Optional[dict[str, torch.Tensor]] = None


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
        return make_torch_optimizer([self.vertices], name=name, lr=lr)

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

                step_norm = float(torch.linalg.norm(update).cpu().item())

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


@dataclass
class ParameterizedShapeOptimizationProblem:
    """Reusable payload for optimizing user parameters through a vertex map."""

    design: ParameterizedVertexDesign
    cells: torch.Tensor
    cfg: dict[str, Any]
    report_cfg: Any = None
    body_ids: Optional[np.ndarray] = None
    boundary_ids: Optional[np.ndarray] = None
    derivative_type: str = "shape"
    solver: Any = None
    solve_log_level: int = 2
    reference_vertices: Optional[torch.Tensor] = None

    def solve(self) -> DifferentiableResult:
        """Run one differentiable solve at the current parameterized vertices."""
        vertices = self.design.vertices()
        if self.reference_vertices is not None and tuple(vertices.shape) != tuple(self.reference_vertices.shape):
            raise ValueError(
                "parameterized vertex map changed vertex shape: "
                f"expected {tuple(self.reference_vertices.shape)}, got {tuple(vertices.shape)}"
            )
        if vertices.requires_grad:
            vertices.retain_grad()

        if self.solver is not None:
            solutions = PolyFEMFunction.apply(
                self.solver,
                vertices,
                self.derivative_type,
                int(self.solve_log_level),
            )
            return DifferentiableResult(
                u=solutions,
                solver=self.solver,
                derivative_type=self.derivative_type,
                differentiable_params=self.design.differentiable_param_names(),
                vertices=vertices,
                meta={"_solve_settings": self.cfg},
            )

        return prepare_differentiable_simulation(
            V=vertices,
            C=self.cells,
            cfg=self.cfg,
            body_ids=self.body_ids,
            boundary_ids=self.boundary_ids,
            derivative_type=self.derivative_type,
            differentiable_params=self.design.differentiable_param_names(),
        )

    def make_optimizer(self, *, name: str = "adam", lr: float = 1e-2) -> torch.optim.Optimizer:
        """Create a PyTorch optimizer for the user design parameters."""
        params = self.design.torch_parameters()
        return make_torch_optimizer(
            params,
            name=name,
            lr=lr,
            empty_error="parameterized shape design has no optimizer parameters",
        )

    def release_solver(self) -> None:
        """Drop the reusable C++ solver reference held by this problem."""
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
        if max_vertex_step_value is not None:
            raise ValueError(
                "max_vertex_step is not supported for parameterized shape designs; "
                "implement projection in the design instead"
            )

        def parameter_snapshot() -> dict[str, torch.Tensor]:
            params = self.design.torch_parameters()
            design_names = self.design.differentiable_param_names()
            return {
                str(
                    parameter_name(
                        parameter,
                        design_names[index] if index < len(design_names) else f"param_{index}",
                    )
                ): parameter.detach().clone()
                for index, parameter in enumerate(params)
            }

        for iteration in range(int(steps)):
            optimizer.zero_grad(set_to_none=True)
            result = self.solve()
            try:
                parameter_values_before = parameter_snapshot()
                vertices_before = result.vertices.detach().clone()
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

                optimizer.step()
                self.design.project_()
                with torch.no_grad():
                    parameter_values_after = parameter_snapshot()
                    vertices_after = self.design.vertices().detach()
                    update = vertices_after - vertices_before
                    per_vertex_update = torch.linalg.norm(update.reshape(update.shape[0], -1), dim=1)
                    max_vertex_update = float(per_vertex_update.max().cpu().item()) if per_vertex_update.numel() else 0.0
                    step_scale = 1.0

                step_norm = float(torch.linalg.norm(update).cpu().item())

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
                    parameter_values_before=parameter_values_before,
                    parameter_values_after=parameter_values_after,
                )
            finally:
                result.release_solver()


__all__ = [
    "LossBuilder",
    "LossOutput",
    "ParameterizedShapeOptimizationProblem",
    "ShapeOptimizationProblem",
    "ShapeOptimizationStep",
]
