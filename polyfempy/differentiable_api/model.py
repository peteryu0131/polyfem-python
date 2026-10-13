"""Differentiable model wrapper for the new differentiable API."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from os import PathLike
from typing import Any


@dataclass(frozen=True)
class ModelMetadata:
    """Snapshot of the loaded mesh and primary FE space (not vertex guesses).

    Query once outside optimization loops. Tensor helpers preserve the input's
    dtype, device and autograd graph; they never create trainable parameters.
    """

    dimension: int
    n_vertices: int
    n_elements: int
    n_bases: int
    value_dim: int
    n_dofs: int
    supports_uniform_initial_condition: bool

    def uniform_initial_condition(self, velocity: Any, *, displacement: Any = None) -> Any:
        """Broadcast component vectors to (n_dofs, 2), columns [u0, v0].

        Currently limited to unconstrained P1 simplex displacement spaces.
        This sets the entire model, not an individual body or selection.
        For other spaces, supply explicit FE coefficients to initial_condition_opt.
        """
        import torch

        if not self.supports_uniform_initial_condition:
            raise ValueError(
                "uniform_initial_condition requires an unconstrained P1 simplex "
                "displacement space without periodic boundaries; use explicit FE "
                "coefficients for other spaces."
            )
        self._check_vector(velocity, self.value_dim, "velocity")
        if displacement is None:
            displacement = torch.zeros_like(velocity)
        self._check_vector(displacement, self.value_dim, "displacement")
        if displacement.dtype != velocity.dtype or displacement.device != velocity.device:
            raise ValueError(
                "displacement and velocity must have the same dtype and device; "
                f"got displacement={displacement.dtype}/{displacement.device}, "
                f"velocity={velocity.dtype}/{velocity.device}. "
                "Use displacement.to(velocity) to match them."
            )
        return torch.stack(
            (displacement.repeat(self.n_bases), velocity.repeat(self.n_bases)), dim=1
        )

    def uniform_material(self, lame: Any) -> Any:
        """Broadcast a [lambda, mu] tensor to all elements, shape (n_elements, 2).

        Gradients from all elements accumulate into the supplied pair. To optimize
        elements independently, create a leaf tensor of shape (n_elements, 2).
        """
        self._check_vector(lame, 2, "lame")
        return lame.unsqueeze(0).repeat(self.n_elements, 1)

    @staticmethod
    def _check_vector(value: Any, size: int, name: str) -> None:
        import torch

        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"{name} must be a floating-point torch.Tensor; got {type(value).__name__}. "
                "Use torch.tensor(values, dtype=torch.float64) for constant input."
            )
        if not value.is_floating_point():
            raise ValueError(
                f"{name} must be floating-point; got {value.dtype}. "
                f"Use {name}.to(dtype=torch.float64)."
            )
        if tuple(value.shape) != (size,):
            components = "[lambda, mu]" if name == "lame" else "one component per spatial dimension"
            raise ValueError(
                f"{name} must have shape ({size},), {components}; "
                f"got {tuple(value.shape)}. This helper broadcasts one vector "
                "to the whole model; pass per-element or per-DOF tensors directly "
                "to the corresponding diff.*_opt operation."
            )
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must contain only finite values")


@dataclass(frozen=True)
class DifferentiableModel:
    """Container for one or more forward models used by differentiable ops."""

    inputs: tuple[Any, ...]
    configs: tuple[Any, ...]
    source_kinds: tuple[str, ...]
    _payloads: tuple[dict[str, Any], ...]

    def as_dicts(self) -> list[dict[str, Any]]:
        """Return JSON-compatible forward config payload copies."""

        return [copy.deepcopy(payload) for payload in self._payloads]

    def metadata(self, *, backend: Any | None = None) -> ModelMetadata:
        """Load one model and inspect its actual mesh/FE layout, without solving.

        Setup may be expensive: retain the result outside optimization loops.
        Requires a compiled backend exposing DifferentiableSession.model_metadata.
        """
        from . import _backend
        from .shape import _load_default_backend

        if len(self._payloads) != 1:
            raise ValueError("metadata supports exactly one model")
        backend_module = backend if backend is not None else _load_default_backend()
        session_type = getattr(backend_module, "DifferentiableSession", None)
        if any(not callable(getattr(session_type, name, None))
               for name in ("set_settings", "model_metadata")):
            raise _backend.BackendContractError(
                "Model metadata requires DifferentiableSession.model_metadata; "
                "rebuild the compiled PolyFEM backend."
            )
        session = session_type()
        session.set_settings(self.as_dicts()[0])
        return ModelMetadata(**session.model_metadata())

    def initial_condition_dof_count(self, *, backend: Any | None = None) -> int:
        """Return the row count for an initial-condition tensor (n_dofs, 2).

        The columns hold initial solution and velocity, respectively. This
        prepares a backend session, so query once before an optimization loop.
        Only a single forward model is supported by initial_condition_opt.
        """

        from . import _backend
        from .shape import _load_default_backend

        payloads = self.as_dicts()
        if len(payloads) != 1:
            raise ValueError("initial_condition_dof_count supports exactly one model")
        backend_module = backend if backend is not None else _load_default_backend()
        session_type = _backend.require_initial_condition_opt_backend(backend_module)
        session = session_type()
        session.set_settings(payloads[0])
        return int(session.initial_condition_dof_count())

    def shape(
        self,
        *,
        selection: Any,
        tensor: Any,
        backend: Any | None = None,
    ) -> Any:
        """Convenience wrapper for the shape differentiable operation."""

        from .shape import shape_solve

        return shape_solve(
            model=self,
            selection=selection,
            tensor=tensor,
            backend=backend,
        )


def model(models: Any) -> DifferentiableModel:
    """Wrap forward model builders, generated configs, or backend-shaped dicts."""

    entries = _normalize_model_entries(models)
    configs: list[Any] = []
    source_kinds: list[str] = []
    payloads: list[dict[str, Any]] = []

    for entry in entries:
        source_kind, config, payload = _normalize_forward_model_entry(entry)
        source_kinds.append(source_kind)
        configs.append(config)
        payloads.append(payload)

    return DifferentiableModel(
        inputs=entries,
        configs=tuple(configs),
        source_kinds=tuple(source_kinds),
        _payloads=tuple(payloads),
    )


def _normalize_model_entries(models: Any) -> tuple[Any, ...]:
    if models is None:
        raise ValueError("diff.model requires at least one forward model")
    if isinstance(models, (str, bytes, PathLike, dict)):
        entries = (models,)
    else:
        try:
            entries = tuple(models)
        except TypeError:
            entries = (models,)

    if not entries:
        raise ValueError("diff.model requires at least one forward model")
    return entries


def _normalize_forward_model_entry(entry: Any) -> tuple[str, Any, dict[str, Any]]:
    if isinstance(entry, dict):
        return "dict", entry, copy.deepcopy(entry)

    if isinstance(entry, (str, PathLike)):
        raise TypeError(
            "forward model entries must be model builders with config(), "
            "generated config objects with as_dict(), or backend-shaped dicts; "
            "JSON path support is not implemented yet"
        )

    config = _config_from_model_builder(entry)
    if config is not None:
        return "model", config, _payload_from_config_result(config)

    as_dict = getattr(entry, "as_dict", None)
    if callable(as_dict):
        payload = as_dict()
        if not isinstance(payload, dict):
            raise TypeError(
                f"as_dict() must return dict, got {type(payload).__name__}"
            )
        return "generated", entry, copy.deepcopy(payload)

    raise TypeError(
        "forward model entries must be model builders with config(), "
        "generated config objects with as_dict(), or backend-shaped dicts; "
        f"got {type(entry).__name__}"
    )


def _config_from_model_builder(entry: Any) -> Any | None:
    config = getattr(entry, "config", None)
    if not callable(config):
        return None
    return config()


def _payload_from_config_result(config: Any) -> dict[str, Any]:
    if isinstance(config, dict):
        return copy.deepcopy(config)

    as_dict = getattr(config, "as_dict", None)
    if callable(as_dict):
        payload = as_dict()
        if isinstance(payload, dict):
            return copy.deepcopy(payload)
        raise TypeError(
            f"config().as_dict() must return dict, got {type(payload).__name__}"
        )

    raise TypeError(
        "config() must return a backend-shaped dict or a generated config "
        f"object with as_dict(); got {type(config).__name__}"
    )


__all__ = ["DifferentiableModel", "ModelMetadata", "model"]
