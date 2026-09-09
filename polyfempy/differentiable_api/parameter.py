"""Parameter namespace for the new differentiable API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .state import State


@dataclass(frozen=True)
class ShapeParameter:
    """Direct shape design parameter for a differentiable state."""

    state: State
    selection: Any
    tensor: Any = None
    parametrization: str = "direct"

    @property
    def kind(self) -> str:
        """Return the native parameter family represented by this object."""

        return "shape"

    def tensors(self) -> tuple[Any, ...]:
        """Return user-owned tensors used by optimizers."""

        if self.tensor is None:
            return ()
        return (self.tensor,)


class ParameterNamespace:
    """Factory namespace for differentiable design parameters."""

    def shape(
        self,
        *,
        state: State,
        selection: Any,
        tensor: Any = None,
        parametrization: str = "direct",
    ) -> ShapeParameter:
        """Create a direct shape parameter without importing torch."""

        if not isinstance(state, State):
            raise TypeError("state must be a polyfempy.differentiable_api.State")
        if selection is None:
            raise ValueError("selection must not be None")
        if parametrization != "direct":
            raise ValueError(
                "only direct shape parametrization is supported in D3; "
                f"got {parametrization!r}"
            )
        return ShapeParameter(
            state=state,
            selection=selection,
            tensor=tensor,
            parametrization=parametrization,
        )


parameter = ParameterNamespace()


__all__ = ["ParameterNamespace", "ShapeParameter", "parameter"]
