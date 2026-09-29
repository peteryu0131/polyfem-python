"""Low-level PyTorch autograd operations for differentiable solves."""

from __future__ import annotations

from typing import Any

import numpy as np

from .material import _run_material_session, _single_material_payload
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

    def _material_lame_to_backend_array(value: Any) -> Any:
        if _is_torch_tensor(value):
            if value.ndim != 2 or value.shape[1] != 2:
                raise ValueError("material Lamé tensor must have shape (n_elements, 2)")
            detached = value.detach()
            return torch.cat((detached[:, 0], detached[:, 1]), dim=0).cpu().numpy()

        array = np.asarray(value)
        if array.ndim != 2 or array.shape[1] != 2:
            raise ValueError("material Lamé tensor must have shape (n_elements, 2)")
        return np.concatenate((array[:, 0], array[:, 1]), axis=0)

    def _material_lame_gradient_from_backend(value: Any, *, element_count: int) -> Any:
        array = np.asarray(value)
        if array.ndim != 1:
            array = array.reshape(-1)
        if array.size != 2 * element_count:
            raise RuntimeError(
                "Material backend gradient must have length "
                f"{2 * element_count}; got {array.size}."
            )
        return np.stack((array[:element_count], array[element_count:]), axis=1)

    class ShapeOpt(Function):
        """Low-level autograd operation for shape differentiable solves."""

        @classmethod
        def apply(cls, *args: Any, **kwargs: Any) -> Any:
            """Accept the meeting-facing API and call torch positionally."""

            if not kwargs:
                return super().apply(*args)

            allowed = {
                "model",
                "selection",
                "tensor",
                "backend",
                "objective",
                "objective_params",
            }
            unknown = sorted(set(kwargs) - allowed)
            if unknown:
                joined = ", ".join(unknown)
                raise TypeError(f"unexpected ShapeOpt.apply keyword(s): {joined}")

            if args:
                if len(args) not in (3, 4):
                    raise TypeError(
                        "ShapeOpt.apply positional API requires model, selection, "
                        "tensor, and optional backend"
                    )
                duplicate = sorted({"model", "selection", "tensor"} & set(kwargs))
                if duplicate:
                    joined = ", ".join(duplicate)
                    raise TypeError(
                        "ShapeOpt.apply got positional and keyword value(s) "
                        f"for: {joined}"
                    )
                if len(args) == 4 and "backend" in kwargs:
                    raise TypeError("ShapeOpt.apply got multiple backend values")
                model, selection, tensor = args[:3]
                backend = args[3] if len(args) == 4 else kwargs.get("backend")
            else:
                try:
                    model = kwargs["model"]
                    selection = kwargs["selection"]
                    tensor = kwargs["tensor"]
                except KeyError as exc:
                    raise TypeError(
                        "ShapeOpt.apply keyword API requires model, selection, and tensor"
                    ) from exc
                backend = kwargs.get("backend")

            if "objective" in kwargs:
                if "objective_params" in kwargs:
                    return super().apply(
                        model,
                        selection,
                        tensor,
                        backend,
                        kwargs["objective"],
                        kwargs["objective_params"],
                    )
                return super().apply(
                    model,
                    selection,
                    tensor,
                    backend,
                    kwargs["objective"],
                )
            if "objective_params" in kwargs:
                raise TypeError("objective_params requires objective")
            return super().apply(
                model,
                selection,
                tensor,
                backend,
            )

        @staticmethod
        def forward(
            ctx: Any,
            model: Any,
            selection: Any,
            tensor: Any,
            backend: Any | None = None,
            objective: Any | None = None,
            objective_params: Any | None = None,
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
                objective_params=objective_params,
                backend=backend,
            )
            ctx.session = session
            ctx.input_tensor = tensor
            ctx.gradient_count = 6 if objective_params is not None else (
                5 if objective is not None else 4
            )
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
                if getattr(ctx, "gradient_count", 4) == 6:
                    return (*gradients, None, None)
                return gradients
            finally:
                ctx.session = None
                ctx.input_tensor = None
                ctx.gradient_count = None

    class MaterialOpt(Function):
        """Low-level autograd operation for elastic Lamé material objectives."""

        @classmethod
        def apply(cls, *args: Any, **kwargs: Any) -> Any:
            """Accept the meeting-facing API and call torch positionally."""

            if not kwargs:
                return super().apply(*args)

            allowed = {
                "model",
                "lame",
                "backend",
                "objective",
                "objective_params",
            }
            unknown = sorted(set(kwargs) - allowed)
            if unknown:
                joined = ", ".join(unknown)
                raise TypeError(f"unexpected MaterialOpt.apply keyword(s): {joined}")

            if args:
                if len(args) not in (2, 3):
                    raise TypeError(
                        "MaterialOpt.apply positional API requires model, "
                        "Lamé tensor, and optional backend"
                    )
                duplicate = sorted({"model", "lame"} & set(kwargs))
                if duplicate:
                    joined = ", ".join(duplicate)
                    raise TypeError(
                        "MaterialOpt.apply got positional and keyword value(s) "
                        f"for: {joined}"
                    )
                if len(args) == 3 and "backend" in kwargs:
                    raise TypeError("MaterialOpt.apply got multiple backend values")
                model, lame = args[:2]
                backend = args[2] if len(args) == 3 else kwargs.get("backend")
            else:
                try:
                    model = kwargs["model"]
                    lame = kwargs["lame"]
                except KeyError as exc:
                    raise TypeError(
                        "MaterialOpt.apply keyword API requires model and lame"
                    ) from exc
                backend = kwargs.get("backend")

            if "objective" not in kwargs:
                raise TypeError("MaterialOpt.apply requires objective")
            if "objective_params" in kwargs:
                return super().apply(
                    model,
                    lame,
                    backend,
                    kwargs["objective"],
                    kwargs["objective_params"],
                )
            return super().apply(
                model,
                lame,
                backend,
                kwargs["objective"],
            )

        @staticmethod
        def forward(
            ctx: Any,
            model: Any,
            lame: Any,
            backend: Any | None = None,
            objective: Any | None = None,
            objective_params: Any | None = None,
        ) -> Any:
            payload = _single_material_payload(
                model=model,
                lame=lame,
            )
            backend_lame = _material_lame_to_backend_array(lame)
            objective_value, session = _run_material_session(
                payload=payload,
                lame=backend_lame,
                objective=objective,
                objective_params=objective_params,
                backend=backend,
            )
            ctx.session = session
            ctx.input_lame = lame
            ctx.element_count = int(lame.shape[0]) if hasattr(lame, "shape") else (
                np.asarray(lame).shape[0]
            )
            ctx.gradient_count = 5 if objective_params is not None else 4
            return _to_torch_tensor(objective_value, like=lame)

        @staticmethod
        @torch.autograd.function.once_differentiable  # type: ignore[union-attr]
        def backward(ctx: Any, grad_output: Any) -> tuple[Any, ...]:
            session = ctx.session
            input_lame = ctx.input_lame
            try:
                backend_grad_output = _to_backend_array(grad_output)
                grad_flat = session.backward_material(backend_grad_output)
                grad_lame = _material_lame_gradient_from_backend(
                    grad_flat,
                    element_count=ctx.element_count,
                )
                gradients = (
                    None,
                    _to_torch_tensor(grad_lame, like=input_lame),
                    None,
                    None,
                )
                if getattr(ctx, "gradient_count", 4) == 5:
                    return (*gradients, None)
                return gradients
            finally:
                ctx.session = None
                ctx.input_lame = None
                ctx.element_count = None
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

    class MaterialOpt:
        """Unavailable MaterialOpt placeholder used when PyTorch is missing."""

        @staticmethod
        def apply(*args: Any, **kwargs: Any) -> Any:
            raise ImportError(
                "PyTorch is required for MaterialOpt differentiable objectives. "
                "Install the 'differentiable' extra or install torch."
            ) from _TORCH_IMPORT_ERROR


__all__ = ["MaterialOpt", "ShapeOpt"]
