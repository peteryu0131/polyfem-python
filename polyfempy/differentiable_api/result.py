"""Result container for future differentiable solves."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DifferentiableResult:
    """Lightweight result shape for the new differentiable API."""

    u: Any
    parameters: tuple[Any, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = ["DifferentiableResult"]

