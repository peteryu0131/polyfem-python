"""Parameter namespace for the new differentiable API."""

from __future__ import annotations

from typing import Any


class ParameterNamespace:
    """Factory namespace for differentiable design parameters."""

    def shape(self, *args: Any, **kwargs: Any) -> None:
        """Create a shape parameter.

        The real shape parameter contract is planned for D3. This D1 skeleton
        only establishes the import surface.
        """

        raise NotImplementedError(
            "polyfempy.differentiable_api.parameter.shape will be implemented in D3."
        )


parameter = ParameterNamespace()


__all__ = ["ParameterNamespace", "parameter"]

