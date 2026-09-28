"""Shape differentiable solve wrapper."""

from __future__ import annotations

import importlib
import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from . import _backend
from .model import DifferentiableModel
from .objectives import Objective, _build_objective


@dataclass(frozen=True)
class ShapeOptimizationResult:
    """Result returned by ShapeOptimization.optimize()."""

    history: tuple[dict[str, Any], ...]
    vertices: Any
    loss: Any


class ShapeOptimization:
    """User-facing shape optimization problem."""

    def __init__(
        self,
        model: DifferentiableModel,
        vertices: Any,
        *,
        objective: Any,
        selection: Any = "all",
        backend: Any | None = None,
    ) -> None:
        if objective is None:
            raise ValueError("objective must not be None")
        _single_shape_payload(
            model=model,
            selection=selection,
            tensor=vertices,
        )
        self.model = model
        self.vertices = vertices
        self.objective = objective
        self.selection = selection
        self.backend = backend
        self.configured_optimizer: Any | None = None
        self.configured_steps: int | None = None

    def __call__(self) -> Any:
        """Return one differentiable objective evaluation."""

        from .torch_ops import ShapeOpt

        kwargs = {"objective": self.objective}
        if self.backend is not None:
            kwargs["backend"] = self.backend
        return ShapeOpt.apply(
            self.model,
            self.selection,
            self.vertices,
            **kwargs,
        )

    def optimizer(self, optimizer: Any, **kwargs: Any) -> ShapeOptimization:
        """Configure the PyTorch optimizer used by optimize()."""

        if _is_optimizer_instance(optimizer):
            if kwargs:
                raise TypeError(
                    "optimizer keyword arguments require an optimizer class or factory"
                )
            configured = optimizer
        elif callable(optimizer):
            configured = optimizer([self.vertices], **kwargs)
        else:
            raise TypeError("optimizer must be an optimizer instance, class, or factory")

        if not _is_optimizer_instance(configured):
            raise TypeError("optimizer must provide zero_grad() and step()")
        self.configured_optimizer = configured
        return self

    def steps(self, steps: int) -> ShapeOptimization:
        """Configure the number of optimization steps."""

        if not isinstance(steps, int):
            raise TypeError("steps must be an int")
        if steps <= 0:
            raise ValueError("steps must be positive")
        self.configured_steps = steps
        return self

    def optimize(
        self,
        *,
        optimizer: Any | None = None,
        steps: int | None = None,
    ) -> ShapeOptimizationResult:
        """Run the configured PyTorch optimization loop."""

        if optimizer is not None:
            self.optimizer(optimizer)
        if steps is not None:
            self.steps(steps)
        if self.configured_optimizer is None:
            raise ValueError("optimizer must be configured before optimize()")
        if self.configured_steps is None:
            raise ValueError("steps must be configured before optimize()")

        history = []
        loss = None
        for step in range(self.configured_steps):
            self.configured_optimizer.zero_grad()
            loss = self()
            loss.backward()
            history.append(
                {
                    "step": step,
                    "loss": _as_float(loss),
                    "gradient_norm": _gradient_norm(self.vertices),
                }
            )
            self.configured_optimizer.step()

        return ShapeOptimizationResult(
            history=tuple(history),
            vertices=self.vertices,
            loss=loss,
        )


def shape_opt(
    model: DifferentiableModel,
    vertices: Any,
    *,
    objective: Any,
    selection: Any = "all",
    backend: Any | None = None,
) -> ShapeOptimization:
    """Create a user-facing shape optimization problem."""

    return ShapeOptimization(
        model=model,
        vertices=vertices,
        objective=objective,
        selection=selection,
        backend=backend,
    )


def shape_solve(
    *,
    model: DifferentiableModel,
    selection: Any,
    tensor: Any,
    backend: Any | None = None,
) -> Any:
    """Run the shape differentiable solve convenience wrapper for one model.

    ``shape_opt(...)`` is the user-facing objective API. This wrapper keeps
    the older direct shape solve available for solution-valued experiments.
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


def _is_optimizer_instance(value: Any) -> bool:
    if isinstance(value, type):
        return False
    return callable(getattr(value, "zero_grad", None)) and callable(
        getattr(value, "step", None)
    )


def _as_float(value: Any) -> float:
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    item = getattr(value, "item", None)
    if callable(item):
        value = item()
    return float(value)


def _gradient_norm(vertices: Any) -> float:
    grad = getattr(vertices, "grad", None)
    if grad is None:
        raise RuntimeError("vertices.grad is None after loss.backward()")
    norm = getattr(grad, "norm", None)
    if not callable(norm):
        raise TypeError("vertices.grad must provide norm()")
    return _as_float(norm())


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
    objective_params: Mapping[str, Any] | None = None,
    backend: Any | None = None,
) -> tuple[Any, Any]:
    backend_module = backend if backend is not None else _load_default_backend()
    session_type = _backend.require_shape_solve_backend(backend_module)
    session = session_type()

    session.set_settings(payload)
    objective_payload = _objective_payload(
        objective,
        objective_params=objective_params,
        default_selection=selection,
    )
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


def _objective_payload(
    objective: Any | None,
    *,
    objective_params: Mapping[str, Any] | None = None,
    default_selection: Any = None,
) -> dict[str, Any] | None:
    if objective is None and objective_params is not None:
        raise TypeError("objective_params requires objective")
    if objective is None:
        return None
    if isinstance(objective, Objective):
        objective = _build_objective(
            objective,
            params=objective_params,
            default_selection=default_selection,
        )
        return objective.as_dict()
    if isinstance(objective, Mapping):
        if objective_params is not None:
            raise TypeError("objective_params is only supported with diff.Objective")
        return copy.deepcopy(dict(objective))

    as_dict = getattr(objective, "as_dict", None)
    if callable(as_dict):
        if objective_params is not None:
            raise TypeError("objective_params is only supported with diff.Objective")
        payload = as_dict()
        if not isinstance(payload, dict):
            raise TypeError(
                f"objective.as_dict() must return dict, got {type(payload).__name__}"
            )
        return copy.deepcopy(payload)

    raise TypeError(
        "objective must be a diff.Objective enum value, "
        "an ObjectiveSpec-like object with as_dict(), or a backend objective dict"
    )


def _load_default_backend() -> Any:
    try:
        return importlib.import_module("polyfempy.polyfempy")
    except Exception as exc:
        raise _backend.BackendContractError(
            "Differentiable shape solve requires the compiled backend module "
            "'polyfempy.polyfempy' with DifferentiableSession."
        ) from exc


__all__ = [
    "ShapeOptimization",
    "ShapeOptimizationResult",
    "shape_opt",
    "shape_solve",
]
