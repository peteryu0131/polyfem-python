"""PyTorch objective bridge helpers for differentiable PolyFEM runs.

This module contains the reusable layer that lets experiment code build a
PolyFEM objective, expose it as a PyTorch autograd operator, and keep direct
solver-derived fields such as von Mises stress as monitor-only values.
"""

# pyright: reportMissingImports=false

from __future__ import annotations

import json
from typing import Any, Optional, Union, overload

import numpy as np
import polyfempy as pf
import torch
from torch.autograd import Function

from ._objective_common import (
    BodySelection,
    ObjectiveLossBuilder,
    ObjectiveLossInfo,
    ObjectiveLossResult,
    ObjectiveLossWithInfo,
    SmoothTimeAggregationName,
    TimeAggregation,
    TimeAggregationName,
    _auto_smooth_max_sharpness,
    _resolve_smooth_max_sharpness,
    _resolve_time_aggregation,
    _resolve_volume_selection,
    objective_state_columns,
    resolve_objective_state_column,
)


def _as_f64_vec(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(x, dtype=np.float64).reshape(-1), dtype=np.float64)


def _array_summary(arr: Any) -> dict[str, Any]:
    a = np.asarray(arr)
    out: dict[str, Any] = {
        "shape": [int(x) for x in a.shape],
        "dtype": str(a.dtype),
        "size": int(a.size),
    }
    if a.size == 0:
        return out

    flat = a.astype(np.float64, copy=False).reshape(-1)
    out.update(
        {
            "min": float(np.min(flat)),
            "max": float(np.max(flat)),
            "mean": float(np.mean(flat)),
            "p95": float(np.percentile(flat, 95.0)),
        }
    )
    return out


def _is_material_result(result: Any) -> bool:
    """Return True for differentiable material results without importing classes."""
    return hasattr(result, "lam") and hasattr(result, "mu")


def create_shape_objective(
    *,
    solver: Any,
    objective_name: str,
    power: int,
    volume_selection: int,
    state: int,
) -> Any:
    """Create a PolyFEM shape objective from simple Python arguments."""
    return create_polyfem_objective(
        solver=solver,
        objective_name=objective_name,
        param_type="shape",
        power=power,
        volume_selection=volume_selection,
        state=state,
    )


def create_polyfem_objective(
    *,
    solver: Any,
    objective_name: str,
    param_type: str,
    power: int,
    volume_selection: int,
    state: int,
) -> Any:
    """Create a PolyFEM objective from simple Python arguments."""
    args = {
        "state": int(state),
        "type": str(objective_name),
        "volume_selection": [int(volume_selection)],
        "power": int(power),
        "weight": 1,
    }
    return pf.create_objective(str(objective_name), str(param_type), solver, json.dumps(args))


def _adjoint_rhs_to_grad_u_tensor(
    raw: np.ndarray,
    ushape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
    state_col: int = 0,
) -> torch.Tensor:
    """Map PolyFEM objective dJ/du to the same layout as ``get_solutions()``."""
    g = np.asarray(raw, dtype=np.float64).ravel()
    if len(ushape) == 1:
        if g.size != int(ushape[0]):
            raise RuntimeError(f"adjoint rhs length {g.size} != solution shape {ushape}")
        return torch.as_tensor(g, dtype=dtype, device=device).reshape(ushape)
    if len(ushape) == 2:
        r, c = int(ushape[0]), int(ushape[1])
        if g.size == r * c:
            return torch.as_tensor(g, dtype=dtype, device=device).reshape(ushape)
        if g.size == r:
            out = torch.zeros(r, c, dtype=dtype, device=device)
            col = min(max(int(state_col), 0), max(c - 1, 0))
            out[:, col] = torch.as_tensor(g, dtype=dtype, device=device)
            return out
        raise RuntimeError(
            f"adjoint rhs length {g.size} incompatible with solution shape {(r, c)}"
        )
    raise RuntimeError(f"unsupported solution rank {len(ushape)}")


def material_design_vector(lam: torch.Tensor, mu: torch.Tensor) -> np.ndarray:
    """Pack per-element Lamé tensors into the flat elastic design vector used by PolyFEM."""
    lam_np = np.asarray(lam.detach().cpu().numpy(), dtype=np.float64).reshape(-1)
    mu_np = np.asarray(mu.detach().cpu().numpy(), dtype=np.float64).reshape(-1)
    return np.ascontiguousarray(np.concatenate([lam_np, mu_np]), dtype=np.float64)


class PolyFEMAutogradObjective(Function):
    """PyTorch autograd bridge for a PolyFEM objective.

    Forward asks PolyFEM for the scalar objective value.
    Backward asks PolyFEM for objective derivatives wrt solution / shape.
    """

    @staticmethod
    def forward(
        ctx: Any,
        obj: Any,
        solver: Any,
        solution: torch.Tensor,
        x: torch.Tensor,
        state_col: int,
    ) -> torch.Tensor:
        ctx.obj = obj
        ctx.solver = solver
        ctx.ushape = tuple(solution.shape)
        ctx.xshape = tuple(x.shape)
        ctx.x = _as_f64_vec(x.detach().cpu().numpy())
        ctx.state_col = int(state_col)
        val = float(obj.value(ctx.x))
        return torch.tensor(val, dtype=solution.dtype, device=solution.device)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[Any, ...]:
        raw = np.asarray(ctx.obj.derivative(ctx.solver, ctx.x, wrt="solution"), dtype=np.float64)
        raw_x = np.asarray(ctx.obj.derivative(ctx.solver, ctx.x, wrt="shape"), dtype=np.float64).ravel()
        grad_u = _adjoint_rhs_to_grad_u_tensor(
            raw,
            ctx.ushape,
            dtype=grad_output.dtype,
            device=grad_output.device,
            state_col=ctx.state_col,
        )

        grad_x = torch.as_tensor(raw_x, dtype=grad_output.dtype, device=grad_output.device).reshape(ctx.xshape)
        return None, None, grad_u * grad_output, grad_x * grad_output, None


class PolyFEMElasticAutogradObjective(Function):
    """PyTorch autograd bridge for a PolyFEM elastic/material objective."""

    @staticmethod
    def forward(
        ctx: Any,
        obj: Any,
        solver: Any,
        solution: torch.Tensor,
        lam: torch.Tensor,
        mu: torch.Tensor,
        state_col: int,
    ) -> torch.Tensor:
        ctx.obj = obj
        ctx.solver = solver
        ctx.ushape = tuple(solution.shape)
        ctx.x = material_design_vector(lam, mu)
        ctx.state_col = int(state_col)
        ctx.objective_name = str(obj.name())
        ctx.save_for_backward(lam, mu)
        val = float(obj.value(ctx.x))
        return torch.tensor(val, dtype=solution.dtype, device=solution.device)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[Any, ...]:
        raw = np.asarray(ctx.obj.derivative(ctx.solver, ctx.x, wrt="solution"), dtype=np.float64)
        grad_u = _adjoint_rhs_to_grad_u_tensor(
            raw,
            ctx.ushape,
            dtype=grad_output.dtype,
            device=grad_output.device,
            state_col=ctx.state_col,
        )
        lam, mu = ctx.saved_tensors
        grad_lam = torch.zeros_like(lam)
        grad_mu = torch.zeros_like(mu)
        raw_elastic = np.asarray(
            ctx.obj.derivative(ctx.solver, ctx.x, wrt="elastic"),
            dtype=np.float64,
        ).reshape(-1)
        n_el = int(lam.numel())
        if raw_elastic.size != 2 * n_el:
            raise RuntimeError(
                f"elastic objective gradient has size {raw_elastic.size}, expected {2 * n_el}"
            )
        grad_lam = torch.as_tensor(
            raw_elastic[:n_el],
            dtype=lam.dtype,
            device=lam.device,
        ).reshape(lam.shape)
        grad_mu = torch.as_tensor(
            raw_elastic[n_el:],
            dtype=mu.dtype,
            device=mu.device,
        ).reshape(mu.shape)
        return None, None, grad_u * grad_output, grad_lam * grad_output, grad_mu * grad_output, None


# Backward-compatible alias matching older experiment naming.
ShapeObjectiveLoss = PolyFEMAutogradObjective


def get_direct_von_mises_monitor(result: Any) -> dict[str, Any]:
    """Return direct solver-derived von Mises data as a monitor-only channel."""
    vm_np = result.get_von_mises_numpy()
    return {
        "available": bool(vm_np is not None),
        "summary": None if vm_np is None else _array_summary(vm_np),
        "role": "monitor_only",
    }


def make_polyfem_autograd_loss(
    *,
    result: Any,
    objective_name: str,
    objective_power: int,
    objective_volume_selection: int,
    objective_state: Optional[Union[str, int]] = "last",
    time_aggregation: Optional[TimeAggregation] = None,
    smooth_max_sharpness: Optional[float] = None,
    time_reduction: Optional[TimeAggregation] = None,
    smooth_max_beta: Optional[float] = None,
) -> ObjectiveLossWithInfo:
    """Build a differentiable loss from a PolyFEM objective.

    This is the reusable optimization-loss path for differentiable runs.
    """
    aggregation = _resolve_time_aggregation(
        time_aggregation=time_aggregation,
        time_reduction=time_reduction,
    )
    requested_sharpness = _resolve_smooth_max_sharpness(
        smooth_max_sharpness=smooth_max_sharpness,
        smooth_max_beta=smooth_max_beta,
    )
    sol = result.u
    sol_np = sol.detach().cpu().numpy()
    n_cols = int(sol_np.shape[1]) if sol_np.ndim == 2 else 1
    state_cols, reduction = objective_state_columns(
        n_cols=n_cols,
        state=objective_state,
        time_aggregation=aggregation,
    )

    losses = []
    for state_col in state_cols:
        obj = create_shape_objective(
            solver=result.solver,
            objective_name=str(objective_name),
            power=int(objective_power),
            volume_selection=int(objective_volume_selection),
            state=int(state_col),
        )
        losses.append(
            PolyFEMAutogradObjective.apply(
                obj,
                result.solver,
                result.u,
                result.vertices.reshape(-1),
                int(state_col),
            )
        )

    if reduction in ("state", "last", "first"):
        return losses[0], state_cols[0]

    stacked = torch.stack(losses)
    if reduction == "max":
        reduced_loss = torch.max(stacked)
    elif reduction == "mean":
        reduced_loss = torch.mean(stacked)
    elif reduction == "sum":
        reduced_loss = torch.sum(stacked)
    elif reduction == "smooth_max":
        sharpness = (
            _auto_smooth_max_sharpness(stacked)
            if requested_sharpness is None
            else float(requested_sharpness)
        )
        if sharpness <= 0:
            raise ValueError(f"smooth_max_sharpness must be positive; got {sharpness!r}")
        reduced_loss = torch.logsumexp(sharpness * stacked, dim=0) / sharpness
    else:
        raise RuntimeError(f"unsupported resolved time aggregation {reduction!r}")

    selected_state_col = None
    if reduction == "max":
        selected_idx = int(torch.argmax(stacked).detach().cpu().item())
        selected_state_col = int(state_cols[selected_idx])

    return reduced_loss, {
        "time_aggregation": reduction,
        "time_reduction": reduction,
        "state_cols": state_cols,
        "selected_state_col": selected_state_col,
        "smooth_max_sharpness": float(sharpness) if reduction == "smooth_max" else None,
        "smooth_max_beta": float(sharpness) if reduction == "smooth_max" else None,
    }


def make_polyfem_elastic_loss(
    *,
    result: Any,
    objective_names: tuple[str, ...],
    objective_power: int,
    objective_volume_selection: int,
    objective_state: Optional[Union[str, int]] = "last",
    time_aggregation: Optional[TimeAggregation] = None,
    smooth_max_sharpness: Optional[float] = None,
    time_reduction: Optional[TimeAggregation] = None,
    smooth_max_beta: Optional[float] = None,
) -> ObjectiveLossWithInfo:
    """Build a differentiable elastic/material loss from one or more PolyFEM objectives."""
    aggregation = _resolve_time_aggregation(
        time_aggregation=time_aggregation,
        time_reduction=time_reduction,
    )
    requested_sharpness = _resolve_smooth_max_sharpness(
        smooth_max_sharpness=smooth_max_sharpness,
        smooth_max_beta=smooth_max_beta,
    )
    sol = result.u
    n_cols = int(sol.shape[1]) if sol.dim() == 2 else 1
    state_cols, reduction = objective_state_columns(
        n_cols=n_cols,
        state=objective_state,
        time_aggregation=aggregation,
    )

    losses = []
    resolved_names: list[str] = []
    for state_col in state_cols:
        errors: list[str] = []
        obj = None
        resolved_name = None
        for objective_name in objective_names:
            try:
                obj = create_polyfem_objective(
                    solver=result.solver,
                    objective_name=str(objective_name),
                    param_type="elastic",
                    power=int(objective_power),
                    volume_selection=int(objective_volume_selection),
                    state=int(state_col),
                )
                resolved_name = str(objective_name)
                break
            except Exception as exc:
                errors.append(f"{objective_name}: {type(exc).__name__}: {exc}")
        if obj is None or resolved_name is None:
            raise RuntimeError("failed to create elastic objective; " + " | ".join(errors))

        resolved_names.append(resolved_name)
        losses.append(
            PolyFEMElasticAutogradObjective.apply(
                obj,
                result.solver,
                result.u,
                result.lam,
                result.mu,
                int(state_col),
            )
        )

    resolved_objective_name = resolved_names[0]
    if any(name != resolved_objective_name for name in resolved_names):
        raise RuntimeError(f"inconsistent elastic objective resolution across states: {resolved_names}")

    if reduction in ("state", "last", "first"):
        return losses[0], {
            "preferred_objective_name": objective_names[0] if objective_names else None,
            "resolved_objective_name": resolved_objective_name,
            "time_aggregation": reduction,
            "time_reduction": reduction,
            "state_cols": state_cols,
            "selected_state_col": state_cols[0],
            "smooth_max_sharpness": None,
            "smooth_max_beta": None,
        }

    stacked = torch.stack(losses)
    sharpness = None
    if reduction == "max":
        reduced_loss = torch.max(stacked)
    elif reduction == "mean":
        reduced_loss = torch.mean(stacked)
    elif reduction == "sum":
        reduced_loss = torch.sum(stacked)
    elif reduction == "smooth_max":
        sharpness = (
            _auto_smooth_max_sharpness(stacked)
            if requested_sharpness is None
            else float(requested_sharpness)
        )
        if sharpness <= 0:
            raise ValueError(f"smooth_max_sharpness must be positive; got {sharpness!r}")
        reduced_loss = torch.logsumexp(sharpness * stacked, dim=0) / sharpness
    else:
        raise RuntimeError(f"unsupported resolved time aggregation {reduction!r}")

    selected_state_col = None
    if reduction == "max":
        selected_idx = int(torch.argmax(stacked).detach().cpu().item())
        selected_state_col = int(state_cols[selected_idx])

    return reduced_loss, {
        "preferred_objective_name": objective_names[0] if objective_names else None,
        "resolved_objective_name": resolved_objective_name,
        "time_aggregation": reduction,
        "time_reduction": reduction,
        "state_cols": state_cols,
        "selected_state_col": selected_state_col,
        "smooth_max_sharpness": float(sharpness) if reduction == "smooth_max" else None,
        "smooth_max_beta": float(sharpness) if reduction == "smooth_max" else None,
    }


@overload
def make_von_mises_loss(
    *,
    result: Any = None,
    body: Optional[BodySelection] = None,
    volume_selection: int = 1,
    time: Optional[SmoothTimeAggregationName] = None,
    time_aggregation: SmoothTimeAggregationName,
    smooth_max_sharpness: Optional[float] = None,
    return_info: bool = False,
    power: int = 2,
    state: Optional[Union[str, int]] = "last",
) -> Union[ObjectiveLossResult, ObjectiveLossBuilder]:
    ...


@overload
def make_von_mises_loss(
    *,
    result: Any = None,
    body: Optional[BodySelection] = None,
    volume_selection: int = 1,
    time: Optional[TimeAggregationName] = None,
    time_aggregation: TimeAggregationName = "last",
    return_info: bool = False,
    power: int = 2,
    state: Optional[Union[str, int]] = "last",
) -> Union[ObjectiveLossResult, ObjectiveLossBuilder]:
    ...


@overload
def make_von_mises_loss(
    *,
    result: Any = None,
    power: int = 2,
    body: Optional[BodySelection] = None,
    volume_selection: int = 1,
    state: Optional[Union[str, int]] = "last",
    time: Optional[TimeAggregationName] = None,
    time_aggregation: Optional[TimeAggregationName] = None,
    smooth_max_sharpness: Optional[float] = None,
    return_info: bool = False,
    time_reduction: Optional[TimeAggregationName] = None,
    smooth_max_beta: Optional[float] = None,
) -> Union[ObjectiveLossResult, ObjectiveLossBuilder]:
    ...


def make_von_mises_loss(
    *,
    result: Any = None,
    power: int = 2,
    body: Optional[BodySelection] = None,
    volume_selection: int = 1,
    state: Optional[Union[str, int]] = "last",
    time: Optional[TimeAggregation] = None,
    time_aggregation: Optional[TimeAggregation] = None,
    smooth_max_sharpness: Optional[float] = None,
    return_info: bool = False,
    time_reduction: Optional[TimeAggregation] = None,
    smooth_max_beta: Optional[float] = None,
) -> Union[ObjectiveLossResult, ObjectiveLossBuilder]:
    """Build a differentiable von Mises loss.

    Common use:
    - ``time_aggregation="smooth_max"`` for a smooth worst-time-step objective
    - ``time_aggregation="max"`` for a hard worst-time-step objective
    - ``time_aggregation="mean"`` to average all time steps

    Args:
        result: Output from ``prepare_differentiable_simulation`` or from a
            material differentiable solve. Material results are detected from
            their ``lambda/mu`` tensors and routed to the elastic objective
            bridge automatically.
            If omitted, the function returns a reusable loss builder that
            accepts a result later. This is useful for optimization loops.
        power: Objective power. Defaults to ``2``.
        body: Friendly alias for ``volume_selection``. For now this must be an
            integer PolyFEM volume selection id, e.g. ``body=1``.
        volume_selection: Body/volume id used by PolyFEM. In experiment 02,
            ``1`` is the lattice.
        state: Single state to use when ``time_aggregation`` is omitted.
        time: Friendly alias for ``time_aggregation``.
        time_aggregation: How to combine per-time-step losses. Allowed values:
            ``"last"``/``"final"`` uses the final time step, ``"first"`` uses
            the first, ``"max"`` uses the largest per-step loss,
            ``"smooth_max"`` uses a differentiable max-like aggregation,
            ``"mean"`` averages all steps, and ``"sum"`` sums all steps.
        smooth_max_sharpness: Only used with ``time_aggregation="smooth_max"``.
            Larger values behave more like hard max; omit it to auto-scale from
            the current loss magnitude.
        return_info: If ``True``, return ``(loss, info)`` instead of just
            ``loss``. The info is only for diagnostics.
        time_reduction: Backward-compatible alias for ``time_aggregation``.
        smooth_max_beta: Backward-compatible alias for
            ``smooth_max_sharpness``.
    """
    resolved_volume_selection = _resolve_volume_selection(
        volume_selection=volume_selection,
        body=body,
    )
    resolved_time_aggregation = _resolve_time_aggregation(
        time=time,
        time_aggregation=time_aggregation,
        time_reduction=time_reduction,
    )

    if result is None:
        def loss_fn(run_result: Any) -> ObjectiveLossResult:
            return make_von_mises_loss(
                result=run_result,
                power=power,
                volume_selection=resolved_volume_selection,
                state=state,
                time_aggregation=resolved_time_aggregation,
                smooth_max_sharpness=smooth_max_sharpness,
                return_info=return_info,
                smooth_max_beta=smooth_max_beta,
            )

        return loss_fn

    if _is_material_result(result):
        return make_material_von_mises_loss(
            result=result,
            power=power,
            volume_selection=resolved_volume_selection,
            state=state,
            time_aggregation=resolved_time_aggregation,
            smooth_max_sharpness=smooth_max_sharpness,
            return_info=return_info,
            smooth_max_beta=smooth_max_beta,
        )

    loss_with_info = make_polyfem_autograd_loss(
        result=result,
        objective_name="von_mises",
        objective_power=int(power),
        objective_volume_selection=int(resolved_volume_selection),
        objective_state=state,
        time_aggregation=resolved_time_aggregation,
        smooth_max_sharpness=smooth_max_sharpness,
        smooth_max_beta=smooth_max_beta,
    )
    return loss_with_info if return_info else loss_with_info[0]


@overload
def make_stress_norm_loss(
    *,
    result: Any = None,
    body: Optional[BodySelection] = None,
    volume_selection: int = 1,
    time: Optional[SmoothTimeAggregationName] = None,
    time_aggregation: SmoothTimeAggregationName,
    smooth_max_sharpness: Optional[float] = None,
    return_info: bool = False,
    power: int = 2,
    state: Optional[Union[str, int]] = "last",
) -> Union[ObjectiveLossResult, ObjectiveLossBuilder]:
    ...


@overload
def make_stress_norm_loss(
    *,
    result: Any = None,
    body: Optional[BodySelection] = None,
    volume_selection: int = 1,
    time: Optional[TimeAggregationName] = None,
    time_aggregation: TimeAggregationName = "last",
    return_info: bool = False,
    power: int = 2,
    state: Optional[Union[str, int]] = "last",
) -> Union[ObjectiveLossResult, ObjectiveLossBuilder]:
    ...


@overload
def make_stress_norm_loss(
    *,
    result: Any = None,
    power: int = 2,
    body: Optional[BodySelection] = None,
    volume_selection: int = 1,
    state: Optional[Union[str, int]] = "last",
    time: Optional[TimeAggregationName] = None,
    time_aggregation: Optional[TimeAggregationName] = None,
    smooth_max_sharpness: Optional[float] = None,
    return_info: bool = False,
    time_reduction: Optional[TimeAggregationName] = None,
    smooth_max_beta: Optional[float] = None,
) -> Union[ObjectiveLossResult, ObjectiveLossBuilder]:
    ...


def make_stress_norm_loss(
    *,
    result: Any = None,
    power: int = 2,
    body: Optional[BodySelection] = None,
    volume_selection: int = 1,
    state: Optional[Union[str, int]] = "last",
    time: Optional[TimeAggregation] = None,
    time_aggregation: Optional[TimeAggregation] = None,
    smooth_max_sharpness: Optional[float] = None,
    return_info: bool = False,
    time_reduction: Optional[TimeAggregation] = None,
    smooth_max_beta: Optional[float] = None,
) -> Union[ObjectiveLossResult, ObjectiveLossBuilder]:
    """Build a differentiable stress-norm loss.

    Args mirror ``make_von_mises_loss``. In particular, ``time_aggregation`` can
    be ``"last"``, ``"final"``, ``"first"``, ``"max"``, ``"smooth_max"``,
    ``"mean"``, or ``"sum"``.
    """
    resolved_volume_selection = _resolve_volume_selection(
        volume_selection=volume_selection,
        body=body,
    )
    resolved_time_aggregation = _resolve_time_aggregation(
        time=time,
        time_aggregation=time_aggregation,
        time_reduction=time_reduction,
    )

    if result is None:
        def loss_fn(run_result: Any) -> ObjectiveLossResult:
            return make_stress_norm_loss(
                result=run_result,
                power=power,
                volume_selection=resolved_volume_selection,
                state=state,
                time_aggregation=resolved_time_aggregation,
                smooth_max_sharpness=smooth_max_sharpness,
                return_info=return_info,
                smooth_max_beta=smooth_max_beta,
            )

        return loss_fn

    loss_with_info = make_polyfem_autograd_loss(
        result=result,
        objective_name="stress_norm",
        objective_power=int(power),
        objective_volume_selection=int(resolved_volume_selection),
        objective_state=state,
        time_aggregation=resolved_time_aggregation,
        smooth_max_sharpness=smooth_max_sharpness,
        smooth_max_beta=smooth_max_beta,
    )
    return loss_with_info if return_info else loss_with_info[0]


@overload
def make_material_von_mises_loss(
    *,
    result: Any = None,
    body: Optional[BodySelection] = None,
    volume_selection: int = 1,
    time: Optional[SmoothTimeAggregationName] = None,
    time_aggregation: SmoothTimeAggregationName,
    smooth_max_sharpness: Optional[float] = None,
    return_info: bool = False,
    power: int = 2,
    state: Optional[Union[str, int]] = "last",
    objective_names: tuple[str, ...] = ("von_mises_material", "von_mises"),
) -> Union[ObjectiveLossResult, ObjectiveLossBuilder]:
    ...


@overload
def make_material_von_mises_loss(
    *,
    result: Any = None,
    body: Optional[BodySelection] = None,
    volume_selection: int = 1,
    time: Optional[TimeAggregationName] = None,
    time_aggregation: TimeAggregationName = "last",
    return_info: bool = False,
    power: int = 2,
    state: Optional[Union[str, int]] = "last",
    objective_names: tuple[str, ...] = ("von_mises_material", "von_mises"),
) -> Union[ObjectiveLossResult, ObjectiveLossBuilder]:
    ...


@overload
def make_material_von_mises_loss(
    *,
    result: Any = None,
    power: int = 2,
    body: Optional[BodySelection] = None,
    volume_selection: int = 1,
    state: Optional[Union[str, int]] = "last",
    time: Optional[TimeAggregationName] = None,
    time_aggregation: Optional[TimeAggregationName] = None,
    smooth_max_sharpness: Optional[float] = None,
    return_info: bool = False,
    time_reduction: Optional[TimeAggregationName] = None,
    smooth_max_beta: Optional[float] = None,
    objective_names: tuple[str, ...] = ("von_mises_material", "von_mises"),
) -> Union[ObjectiveLossResult, ObjectiveLossBuilder]:
    ...


def make_material_von_mises_loss(
    *,
    result: Any = None,
    power: int = 2,
    body: Optional[BodySelection] = None,
    volume_selection: int = 1,
    state: Optional[Union[str, int]] = "last",
    time: Optional[TimeAggregation] = None,
    time_aggregation: Optional[TimeAggregation] = None,
    smooth_max_sharpness: Optional[float] = None,
    return_info: bool = False,
    time_reduction: Optional[TimeAggregation] = None,
    smooth_max_beta: Optional[float] = None,
    objective_names: tuple[str, ...] = ("von_mises_material", "von_mises"),
) -> Union[ObjectiveLossResult, ObjectiveLossBuilder]:
    """Build a differentiable von Mises loss for material/elastic optimization."""
    resolved_volume_selection = _resolve_volume_selection(
        volume_selection=volume_selection,
        body=body,
    )
    resolved_time_aggregation = _resolve_time_aggregation(
        time=time,
        time_aggregation=time_aggregation,
        time_reduction=time_reduction,
    )

    if result is None:
        def loss_fn(run_result: Any) -> ObjectiveLossResult:
            return make_material_von_mises_loss(
                result=run_result,
                power=power,
                volume_selection=resolved_volume_selection,
                state=state,
                time_aggregation=resolved_time_aggregation,
                smooth_max_sharpness=smooth_max_sharpness,
                return_info=return_info,
                smooth_max_beta=smooth_max_beta,
                objective_names=objective_names,
            )

        return loss_fn

    loss_with_info = make_polyfem_elastic_loss(
        result=result,
        objective_names=objective_names,
        objective_power=int(power),
        objective_volume_selection=int(resolved_volume_selection),
        objective_state=state,
        time_aggregation=resolved_time_aggregation,
        smooth_max_sharpness=smooth_max_sharpness,
        smooth_max_beta=smooth_max_beta,
    )
    return loss_with_info if return_info else loss_with_info[0]


__all__ = [
    "PolyFEMAutogradObjective",
    "ShapeObjectiveLoss",
    "ObjectiveLossBuilder",
    "ObjectiveLossInfo",
    "ObjectiveLossWithInfo",
    "BodySelection",
    "TimeAggregationName",
    "TimeAggregation",
    "SmoothTimeAggregationName",
    "ObjectiveLossResult",
    "create_shape_objective",
    "objective_state_columns",
    "resolve_objective_state_column",
    "make_polyfem_autograd_loss",
    "make_polyfem_elastic_loss",
    "make_von_mises_loss",
    "make_material_von_mises_loss",
    "make_stress_norm_loss",
    "get_direct_von_mises_monitor",
    "create_polyfem_objective",
    "material_design_vector",
]
