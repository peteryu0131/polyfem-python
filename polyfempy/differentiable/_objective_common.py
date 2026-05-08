"""Common type aliases and argument parsing for differentiable objectives."""

from __future__ import annotations

from typing import Any, Callable, Literal, Optional, TypeAlias, Union

import numpy as np
import torch


TimeAggregationName: TypeAlias = Literal[
    "last",
    "final",
    "first",
    "max",
    "smooth_max",
    "mean",
    "sum",
    "softmax",
    "logsumexp",
]
SmoothTimeAggregationName: TypeAlias = Literal["smooth_max", "softmax", "logsumexp"]
TimeAggregation = TimeAggregationName
ObjectiveLossInfo: TypeAlias = Union[int, dict[str, Any]]
ObjectiveLossWithInfo: TypeAlias = tuple[torch.Tensor, ObjectiveLossInfo]
ObjectiveLossResult: TypeAlias = Union[torch.Tensor, ObjectiveLossWithInfo]
ObjectiveLossBuilder: TypeAlias = Callable[[Any], ObjectiveLossResult]
BodySelection: TypeAlias = Union[int, str]


def resolve_objective_state_column(spec: str, n_cols: int) -> int:
    """Resolve ``first`` / ``last`` / integer index to a solution column."""
    s = str(spec).strip().lower()
    if s in ("last", ""):
        return max(n_cols - 1, 0)
    if s in ("first", "0"):
        return 0
    k = int(s)
    if not (0 <= k < n_cols):
        raise ValueError(f"objective state index {k} out of range for n_cols={n_cols}")
    return k


def objective_state_columns(
    *,
    n_cols: int,
    state: Optional[Union[str, int]] = None,
    time_aggregation: Optional[TimeAggregation] = None,
    time_reduction: Optional[TimeAggregation] = None,
) -> tuple[list[int], str]:
    """Resolve a time aggregation into objective state columns.

    ``time_aggregation`` combines per-state losses into one scalar objective. The
    old ``state`` argument remains supported for single-state objectives.
    """
    aggregation = _resolve_time_aggregation(
        time_aggregation=time_aggregation,
        time_reduction=time_reduction,
    )
    reduction = (str(aggregation).strip().lower() if aggregation is not None else "")
    if not reduction:
        state_spec = "last" if state is None else str(state)
        return [resolve_objective_state_column(state_spec, n_cols)], "state"
    if reduction in ("last", "final"):
        return [resolve_objective_state_column("last", n_cols)], "last"
    if reduction == "first":
        return [0], "first"
    if reduction in ("max", "mean", "sum", "smooth_max", "softmax", "logsumexp"):
        if reduction in ("softmax", "logsumexp"):
            reduction = "smooth_max"
        return list(range(max(int(n_cols), 1))), reduction
    raise ValueError(
        "time_aggregation must be one of None, 'last', 'first', 'max', 'mean', 'sum', "
        "'smooth_max', 'softmax', or 'logsumexp'; "
        f"got {aggregation!r}"
    )


def _resolve_time_aggregation(
    *,
    time: Optional[TimeAggregation] = None,
    time_aggregation: Optional[TimeAggregation] = None,
    time_reduction: Optional[TimeAggregation] = None,
) -> Optional[TimeAggregation]:
    """Resolve ``time`` plus the older time aggregation aliases."""
    values = [
        ("time", time),
        ("time_aggregation", time_aggregation),
        ("time_reduction", time_reduction),
    ]
    provided = [(name, value) for name, value in values if value is not None]
    if not provided:
        return None

    _, resolved_value = provided[0]
    resolved_key = str(resolved_value).strip().lower()
    for name, value in provided[1:]:
        if str(value).strip().lower() != resolved_key:
            choices = ", ".join(f"{n}={v!r}" for n, v in provided)
            raise ValueError(
                "Use only one of time, time_aggregation, or time_reduction, "
                f"not conflicting values ({choices})."
            )
    return resolved_value


def _resolve_volume_selection(
    *,
    volume_selection: int = 1,
    body: Optional[BodySelection] = None,
) -> int:
    """Resolve the friendly ``body`` alias to PolyFEM's volume selection id."""
    if body is None:
        return int(volume_selection)

    try:
        body_id = int(body)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "body must be an integer PolyFEM volume selection id for now; "
            f"got {body!r}. Use body=1 or volume_selection=1."
        ) from exc

    selected = int(volume_selection)
    if selected != 1 and selected != body_id:
        raise ValueError(
            "Use either body or volume_selection, not conflicting values "
            f"body={body!r} and volume_selection={volume_selection!r}."
        )
    return body_id


def _resolve_smooth_max_sharpness(
    *,
    smooth_max_sharpness: Optional[float] = None,
    smooth_max_beta: Optional[float] = None,
) -> Optional[float]:
    """Resolve the public smooth-max sharpness name plus the older beta alias."""
    if smooth_max_sharpness is None and smooth_max_beta is None:
        return None
    if smooth_max_sharpness is None:
        return float(smooth_max_beta)
    if smooth_max_beta is None:
        return float(smooth_max_sharpness)
    if float(smooth_max_sharpness) != float(smooth_max_beta):
        raise ValueError(
            "Use either smooth_max_sharpness or smooth_max_beta, not conflicting values "
            f"{smooth_max_sharpness!r} and {smooth_max_beta!r}."
        )
    return float(smooth_max_sharpness)


def _auto_smooth_max_sharpness(stacked: torch.Tensor) -> float:
    """Choose a smooth-max sharpness from the current objective scale."""
    scale = float(torch.max(torch.abs(stacked.detach())).cpu().item())
    if not np.isfinite(scale) or scale <= 0:
        return 1.0
    return 10.0 / scale


__all__ = [
    "BodySelection",
    "ObjectiveLossBuilder",
    "ObjectiveLossInfo",
    "ObjectiveLossResult",
    "ObjectiveLossWithInfo",
    "SmoothTimeAggregationName",
    "TimeAggregation",
    "TimeAggregationName",
    "objective_state_columns",
    "resolve_objective_state_column",
]
