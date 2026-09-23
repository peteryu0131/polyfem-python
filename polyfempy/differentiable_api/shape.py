"""Shape differentiable solve wrapper."""

from __future__ import annotations

import importlib
import copy
from collections.abc import Mapping
from typing import Any

from . import _backend
from .model import DifferentiableModel


def shape_solve(
    *,
    model: DifferentiableModel,
    selection: Any,
    tensor: Any,
    backend: Any | None = None,
) -> Any:
    """Run the shape differentiable solve convenience wrapper for one model.

    ``ShapeOpt.apply(...)`` is the explicit PyTorch autograd API. This wrapper
    keeps the older Python call shape available without making it the primary
    public proposal.
    """

    payload = _single_shape_payload(
        model=model,
        selection=selection,
        tensor=tensor,
    )
    solution, _session = _run_shape_session(
        payload=payload,
        selection=selection,
        tensor=tensor,
        backend=backend,
    )
    return solution


def _single_shape_payload(
    *,
    model: DifferentiableModel,
    selection: Any,
    tensor: Any,
) -> dict[str, Any]:
    if not isinstance(model, DifferentiableModel):
        raise TypeError("model must be a polyfempy.differentiable_api.DifferentiableModel")
    if selection is None:
        raise ValueError("selection must not be None")
    if tensor is None:
        raise ValueError("tensor must not be None")

    payloads = model.as_dicts()
    if len(payloads) != 1:
        raise ValueError("shape_solve MVP supports exactly one model")
    return payloads[0]


def _run_shape_session(
    *,
    payload: dict[str, Any],
    selection: Any,
    tensor: Any,
    objective: Any | None = None,
    backend: Any | None = None,
) -> tuple[Any, Any]:
    backend_module = backend if backend is not None else _load_default_backend()
    session_type = _backend.require_shape_solve_backend(backend_module)
    session = session_type()

    session.set_settings(payload)
    objective_payload = _objective_payload(objective)
    if objective_payload is not None:
        set_objective = getattr(session, "set_objective", None)
        if not callable(set_objective):
            raise _backend.BackendContractError(
                "Objective-aware shape solves require "
                "DifferentiableSession.set_objective."
            )
        set_objective(objective_payload)
    session.set_shape_vertices(tensor, selection=selection)
    if objective_payload is not None:
        solve_objective = getattr(session, "solve_objective", None)
        if not callable(solve_objective):
            raise _backend.BackendContractError(
                "Objective-aware shape solves require "
                "DifferentiableSession.solve_objective."
            )
        return solve_objective(), session
    return session.solve(), session


def _objective_payload(objective: Any | None) -> dict[str, Any] | None:
    if objective is None:
        return None
    if isinstance(objective, Mapping):
        return copy.deepcopy(dict(objective))

    as_dict = getattr(objective, "as_dict", None)
    if callable(as_dict):
        payload = as_dict()
        if not isinstance(payload, dict):
            raise TypeError(
                f"objective.as_dict() must return dict, got {type(payload).__name__}"
            )
        return copy.deepcopy(payload)

    raise TypeError(
        "objective must be an ObjectiveSpec-like object with as_dict() "
        "or a backend objective dict"
    )


def _load_default_backend() -> Any:
    try:
        return importlib.import_module("polyfempy.polyfempy")
    except Exception as exc:
        raise _backend.BackendContractError(
            "Differentiable shape solve requires the compiled backend module "
            "'polyfempy.polyfempy' with DifferentiableSession."
        ) from exc


__all__ = ["shape_solve"]
