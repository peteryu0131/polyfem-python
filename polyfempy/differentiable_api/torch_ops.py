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
        array = value.detach().cpu().numpy()
        if getattr(array, "shape", None) == ():
            return float(array)
        return array

    def _to_torch_tensor(value: Any, *, like: Any) -> Any:
        if _is_torch_tensor(value) or not _is_torch_tensor(like):
            return value
        return torch.as_tensor(value, dtype=like.dtype, device=like.device)

    class ShapeOpt(Function):
        """Low-level autograd operation for shape differentiable solves."""

        @classmethod
        def apply(cls, *args: Any, **kwargs: Any) -> Any:
            """Accept the meeting-facing keyword API and call torch positionally."""

            if not kwargs:
                return super().apply(*args)
            if args:
                raise TypeError(
                    "ShapeOpt.apply accepts either positional arguments or "
                    "keyword arguments, not both"
                )

            allowed = {"model", "selection", "tensor", "backend", "objective"}
            unknown = sorted(set(kwargs) - allowed)
            if unknown:
                joined = ", ".join(unknown)
                raise TypeError(f"unexpected ShapeOpt.apply keyword(s): {joined}")

            try:
                model = kwargs["model"]
                selection = kwargs["selection"]
                tensor = kwargs["tensor"]
            except KeyError as exc:
                raise TypeError(
                    "ShapeOpt.apply keyword API requires model, selection, and tensor"
                ) from exc

            return super().apply(
                model,
                selection,
                tensor,
                kwargs.get("backend"),
                kwargs.get("objective"),
            )

        @staticmethod
        def forward(
            ctx: Any,
            model: Any,
            selection: Any,
            tensor: Any,
            backend: Any | None = None,
            objective: Any | None = None,
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
                objective=objective,
                backend=backend,
            )
            ctx.session = session
            ctx.input_tensor = tensor
            ctx.gradient_count = 5 if objective is not None else 4
            return _to_torch_tensor(solution, like=tensor)

        @staticmethod
        @torch.autograd.function.once_differentiable  # type: ignore[union-attr]
        def backward(ctx: Any, grad_output: Any) -> tuple[Any, ...]:
            session = ctx.session
            input_tensor = ctx.input_tensor
            try:
                backend_grad_output = _to_backend_array(grad_output)
                grad_tensor = session.backward_shape(backend_grad_output)
                gradients = (
                    None,
                    None,
                    _to_torch_tensor(grad_tensor, like=input_tensor),
                    None,
                )
                if getattr(ctx, "gradient_count", 4) == 5:
                    return (*gradients, None)
                return gradients
            finally:
                ctx.session = None
                ctx.input_tensor = None
                ctx.gradient_count = None

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
