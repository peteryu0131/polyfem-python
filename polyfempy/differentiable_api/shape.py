"""Shape differentiable solve wrapper."""

from __future__ import annotations

import importlib
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
    """Run the shape differentiable solve wrapper for one model.

    This is the user-facing wrapper behind ``diff_model.shape(...)``. The real
    torch autograd operation is added later; this function fixes the Python
    contract and backend call shape first.
    """

    if not isinstance(model, DifferentiableModel):
        raise TypeError("model must be a polyfempy.differentiable_api.DifferentiableModel")
    if selection is None:
        raise ValueError("selection must not be None")
    if tensor is None:
        raise ValueError("tensor must not be None")

    payloads = model.as_dicts()
    if len(payloads) != 1:
        raise ValueError("shape_solve MVP supports exactly one model")

    backend_module = backend if backend is not None else _load_default_backend()
    session_type = _backend.require_shape_solve_backend(backend_module)
    session = session_type()

    session.set_settings(payloads[0])
    session.set_shape_vertices(tensor, selection=selection)
    return session.solve()


def _load_default_backend() -> Any:
    try:
        return importlib.import_module("polyfempy.polyfempy")
    except Exception as exc:
        raise _backend.BackendContractError(
            "Differentiable shape solve requires the compiled backend module "
            "'polyfempy.polyfempy' with DifferentiableSession."
        ) from exc


__all__ = ["shape_solve"]

