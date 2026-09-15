"""Backend contract helpers for the new differentiable API."""

from __future__ import annotations

from typing import Any


REQUIRED_BACKEND_SYMBOLS = ("DifferentiableSession",)

REQUIRED_SESSION_METHODS = (
    "set_settings",
    "set_shape_parameter",
    "solve",
    "backward_shape",
)

REQUIRED_SHAPE_SOLVE_SESSION_METHODS = (
    "set_settings",
    "set_shape_vertices",
    "solve",
    "backward_shape",
)

SUPPORTED_PARAMETER_KINDS = ("shape",)

UNSUPPORTED_OPT_PARAMETER_KINDS = (
    "periodic-shape",
    "elastic",
    "friction",
    "damping",
    "initial",
    "dirichlet-boundary",
    "dirichlet-nodes",
    "pressure",
)


class BackendContractError(RuntimeError):
    """Raised when the compiled backend does not expose the D4-B MVP surface."""


def declared_opt_parameter_kinds() -> tuple[str, ...]:
    """Return native opt parameter kinds classified by this MVP policy."""

    return SUPPORTED_PARAMETER_KINDS + UNSUPPORTED_OPT_PARAMETER_KINDS


def require_shape_mvp_backend(backend: Any) -> type:
    """Return the backend DifferentiableSession type if it satisfies D4-B."""

    session_type = getattr(backend, "DifferentiableSession", None)
    if session_type is None:
        raise BackendContractError(
            "The compiled backend must expose DifferentiableSession for the "
            "direct shape differentiable MVP."
        )

    missing = [
        method
        for method in REQUIRED_SESSION_METHODS
        if not callable(getattr(session_type, method, None))
    ]
    if missing:
        joined = ", ".join(missing)
        raise BackendContractError(
            "DifferentiableSession is missing required method(s): "
            f"{joined}."
        )

    return session_type


def require_shape_solve_backend(backend: Any) -> type:
    """Return the backend session type required by diff_model.shape(...)."""

    session_type = getattr(backend, "DifferentiableSession", None)
    if session_type is None:
        raise BackendContractError(
            "The compiled backend must expose DifferentiableSession for "
            "differentiable shape solves."
        )

    missing = [
        method
        for method in REQUIRED_SHAPE_SOLVE_SESSION_METHODS
        if not callable(getattr(session_type, method, None))
    ]
    if missing:
        joined = ", ".join(missing)
        raise BackendContractError(
            "DifferentiableSession is missing required shape solve method(s): "
            f"{joined}."
        )

    return session_type


__all__ = [
    "BackendContractError",
    "REQUIRED_BACKEND_SYMBOLS",
    "REQUIRED_SHAPE_SOLVE_SESSION_METHODS",
    "REQUIRED_SESSION_METHODS",
    "SUPPORTED_PARAMETER_KINDS",
    "UNSUPPORTED_OPT_PARAMETER_KINDS",
    "declared_opt_parameter_kinds",
    "require_shape_solve_backend",
    "require_shape_mvp_backend",
]
