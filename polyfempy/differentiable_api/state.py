"""State wrapper for the new differentiable API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class State:
    """Container for a forward config accepted by future differentiable solves."""

    config: Any


def state(config: Any) -> State:
    """Wrap a generated forward config, backend-shaped dict, or JSON path."""

    return State(config=config)


__all__ = ["State", "state"]

