"""Material differentiable objective wrapper."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from . import _backend
from .model import DifferentiableModel
from .shape import _load_default_backend, _objective_payload


class MaterialOptimization:
    """User-facing elastic Lamé material objective problem."""

    def __init__(
        self,
        model: DifferentiableModel,
        lame: Any,
        *,
        objective: Any,
        objective_params: Mapping[str, Any] | None = None,
        backend: Any | None = None,
    ) -> None:
        if objective is None:
            raise ValueError("objective must not be None")
        _single_material_payload(model=model, lame=lame)
        self.model = model
        self.lame = lame
        self.objective = objective
        self.objective_params = (
            None if objective_params is None else dict(objective_params)
        )
        self.backend = backend

    def __call__(self) -> Any:
        """Return one differentiable material objective evaluation."""

        from .torch_ops import MaterialOpt

        kwargs = {"objective": self.objective}
        if self.objective_params is not None:
            kwargs["objective_params"] = self.objective_params
        if self.backend is not None:
            kwargs["backend"] = self.backend
        return MaterialOpt.apply(
            self.model,
            self.lame,
            **kwargs,
        )


def material_opt(
    model: DifferentiableModel,
    lame: Any,
    *,
    objective: Any,
    objective_params: Mapping[str, Any] | None = None,
    backend: Any | None = None,
) -> MaterialOptimization:
    """Create a user-facing elastic Lamé material objective problem."""

    return MaterialOptimization(
        model=model,
        lame=lame,
        objective=objective,
        objective_params=objective_params,
        backend=backend,
    )


def _single_material_payload(
    *,
    model: DifferentiableModel,
    lame: Any,
) -> dict[str, Any]:
    if not isinstance(model, DifferentiableModel):
        raise TypeError("model must be a polyfempy.differentiable_api.DifferentiableModel")
    if lame is None:
        raise ValueError("lame must not be None")

    payloads = model.as_dicts()
    if len(payloads) != 1:
        raise ValueError("material_opt MVP supports exactly one model")
    return payloads[0]


def _run_material_session(
    *,
    payload: dict[str, Any],
    lame: Any,
    objective: Any,
    objective_params: Mapping[str, Any] | None = None,
    backend: Any | None = None,
) -> tuple[Any, Any]:
    backend_module = backend if backend is not None else _load_default_backend()
    session_type = _backend.require_material_opt_backend(backend_module)
    session = session_type()

    session.set_settings(payload)
    session.set_material_lame_parameters(lame)
    objective_payload = _objective_payload(
        objective,
        objective_params=objective_params,
        default_selection="all",
    )
    if objective_payload is None:
        raise ValueError("material_opt requires an objective")
    session.set_objective(objective_payload)
    return session.solve_material_objective(), session


__all__ = [
    "MaterialOptimization",
    "material_opt",
]
