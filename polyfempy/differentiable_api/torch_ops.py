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

    def _is_torch_tensor(value: Any) -> bool:
        tensor_type = getattr(torch, "Tensor", None)
        return tensor_type is not None and isinstance(value, tensor_type)

    def _to_backend_array(value: Any) -> Any:
        if not _is_torch_tensor(value):
            return value
        return value.detach().cpu().numpy()

    def _to_torch_tensor(value: Any, *, like: Any) -> Any:
        if _is_torch_tensor(value) or not _is_torch_tensor(like):
            return value
        return torch.as_tensor(value, dtype=like.dtype, device=like.device)

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
            backend_tensor = _to_backend_array(tensor)
            solution, session = _run_shape_session(
                payload=payload,
                selection=selection,
                tensor=backend_tensor,
                backend=backend,
            )
            ctx.session = session
            ctx.input_tensor = tensor
            return _to_torch_tensor(solution, like=tensor)

        @staticmethod
        @torch.autograd.function.once_differentiable  # type: ignore[union-attr]
        def backward(ctx: Any, grad_output: Any) -> tuple[Any, ...]:
            backend_grad_output = _to_backend_array(grad_output)
            grad_tensor = ctx.session.backward_shape(backend_grad_output)
            return None, None, _to_torch_tensor(grad_tensor, like=ctx.input_tensor), None

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
