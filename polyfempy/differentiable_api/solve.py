"""Differentiable solve entry point."""

from __future__ import annotations

from typing import Any


def solve(*args: Any, **kwargs: Any) -> None:
    """Run a differentiable PolyFEM solve.

    The real forward/autograd path is planned for D5-D6. This D1 skeleton only
    establishes the public import surface.
    """

    raise NotImplementedError("polyfempy.differentiable_api.solve will be implemented in D5.")


__all__ = ["solve"]

