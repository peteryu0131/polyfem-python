"""Low-level PyTorch autograd operations for differentiable solves."""

from __future__ import annotations

from typing import Any

from .shape import _run_shape_session, _single_shape_payload

try:
    import torch  # pyright: ignore[reportMissingImports]
    from torch.autograd import Function  # pyright: ignore[reportMissingImports]

    _TORCH_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - exercised only without torch
    torch = None  # type: ignore[assignment]
    Function = object  # type: ignore[assignment,misc]
    _TORCH_IMPORT_ERROR = exc


if _TORCH_IMPORT_ERROR is None:

    class ShapeOpt(Function):
        """Low-level autograd operation for shape differentiable solves."""

        @staticmethod
        def forward(
            ctx: Any,
            model: Any,
            selection: Any,
            tensor: Any,
            backend: Any | None = None,
        ) -> Any:
            payload = _single_shape_payload(
                model=model,
                selection=selection,
                tensor=tensor,
            )
            solution, session = _run_shape_session(
                payload=payload,
                selection=selection,
                tensor=tensor,
                backend=backend,
            )
            ctx.session = session
            return solution

        @staticmethod
        @torch.autograd.function.once_differentiable  # type: ignore[union-attr]
        def backward(ctx: Any, grad_output: Any) -> tuple[Any, ...]:
            grad_tensor = ctx.session.backward_shape(grad_output)
            return None, None, grad_tensor, None

else:

    class ShapeOpt:
        """Unavailable ShapeOpt placeholder used when PyTorch is missing."""

        @staticmethod
        def apply(*args: Any, **kwargs: Any) -> Any:
            raise ImportError(
                "PyTorch is required for ShapeOpt differentiable solves. "
                "Install the 'differentiable' extra or install torch."
            ) from _TORCH_IMPORT_ERROR


__all__ = ["ShapeOpt"]

