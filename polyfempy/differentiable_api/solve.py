"""Differentiable solve entry point."""

from __future__ import annotations

import importlib
from collections.abc import Iterable
from typing import Any

from . import _backend
from .parameter import ShapeParameter
from .result import DifferentiableResult
from .state import State


def solve(
    *,
    state: State,
    parameters: Iterable[Any],
    backend: Any | None = None,
) -> DifferentiableResult:
    """Run a differentiable PolyFEM solve.

    D5-A validates the Python contract and calls the minimal direct-shape
    backend session surface. PyTorch autograd is added later.
    """

    if not isinstance(state, State):
        raise TypeError("state must be a polyfempy.differentiable_api.State")

    params = _normalize_parameters(parameters)
    _validate_parameters(state, params)

    backend_module = backend if backend is not None else _load_default_backend()
    session_type = _backend.require_shape_mvp_backend(backend_module)
    session = session_type()

    session.set_settings(state.as_dict())
    for parameter in params:
        session.set_shape_parameter(parameter.tensor, selection=parameter.selection)

    return DifferentiableResult(u=session.solve(), parameters=params)


def _normalize_parameters(parameters: Iterable[Any]) -> tuple[Any, ...]:
    try:
        params = tuple(parameters)
    except TypeError as exc:
        raise TypeError("parameters must be an iterable of differentiable parameters") from exc

    if not params:
        raise ValueError("D.solve requires at least one differentiable parameter")
    return params


def _validate_parameters(state: State, parameters: tuple[Any, ...]) -> None:
    for parameter in parameters:
        kind = getattr(parameter, "kind", None)
        if kind not in _backend.SUPPORTED_PARAMETER_KINDS:
            raise ValueError(
                "unsupported differentiable parameter kind "
                f"{kind!r}; supported kinds: {_backend.SUPPORTED_PARAMETER_KINDS}"
            )
        if not isinstance(parameter, ShapeParameter):
            raise TypeError(
                "shape parameters passed to D.solve must be created by D.parameter.shape"
            )
        if getattr(parameter, "state", None) is not state:
            raise ValueError("all differentiable parameters must reference the same state")


def _load_default_backend() -> Any:
    try:
        return importlib.import_module("polyfempy.polyfempy")
    except Exception as exc:
        raise _backend.BackendContractError(
            "Differentiable solve requires the compiled backend module "
            "'polyfempy.polyfempy' with DifferentiableSession."
        ) from exc


__all__ = ["solve"]
