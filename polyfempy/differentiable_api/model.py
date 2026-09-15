"""Differentiable model wrapper for the new differentiable API."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from os import PathLike
from typing import Any


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

    def shape(
        self,
        *,
        selection: Any,
        tensor: Any,
        backend: Any | None = None,
    ) -> Any:
        """Run a shape differentiable solve for this differentiable model."""

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


__all__ = ["DifferentiableModel", "model"]
