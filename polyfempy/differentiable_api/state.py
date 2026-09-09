"""State wrapper for the new differentiable API."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from os import PathLike
from typing import Any


@dataclass(frozen=True)
class State:
    """Container for a forward config accepted by future differentiable solves."""

    config: Any
    source_kind: str
    _payload: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible forward config payload copy."""

        return copy.deepcopy(self._payload)


def state(config: Any) -> State:
    """Wrap a generated forward config or backend-shaped dict."""

    source_kind, payload = _normalize_state_config(config)
    return State(config=config, source_kind=source_kind, _payload=payload)


def _normalize_state_config(config: Any) -> tuple[str, dict[str, Any]]:
    if config is None:
        raise ValueError("state config must not be None")

    if isinstance(config, dict):
        return "dict", copy.deepcopy(config)

    if isinstance(config, (str, PathLike)):
        raise TypeError(
            "D.state currently supports dicts and generated config objects with as_dict(); "
            "JSON path support is not implemented in D2-A."
        )

    as_dict = getattr(config, "as_dict", None)
    if callable(as_dict):
        payload = as_dict()
        if not isinstance(payload, dict):
            raise TypeError(
                f"config.as_dict() must return dict, got {type(payload).__name__}"
            )
        return "generated", copy.deepcopy(payload)

    raise TypeError(
        "D.state currently supports dicts and generated config objects with as_dict(); "
        f"got {type(config).__name__}"
    )


__all__ = ["State", "state"]
