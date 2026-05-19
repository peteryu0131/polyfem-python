"""Small summary helpers for differentiable experiments."""

# pyright: reportMissingImports=false

from __future__ import annotations

from typing import Any, Mapping, Optional

import torch


def gradient_norm(grad: Optional[torch.Tensor]) -> float:
    """Return a scalar norm for a gradient tensor."""
    if grad is None:
        return float("nan")
    return float(torch.linalg.norm(grad.detach()).cpu().item())


def print_loss_summary(
    *,
    loss: torch.Tensor,
    grad: Optional[torch.Tensor],
    include_grad_shape: bool = True,
) -> None:
    """Print a compact scalar loss + gradient summary."""
    loss_value = float(loss.detach().cpu().item())
    print(f"loss: {loss_value:.6e}")
    print(f"grad_norm: {gradient_norm(grad):.6e}")
    if include_grad_shape and grad is not None:
        print(f"gradient shape: {tuple(grad.shape)}")


def _scalar(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().reshape(-1)[0].item())
    return float(value)


def print_parameterized_shape_summary(
    run: Any,
    *,
    parameters: Mapping[str, torch.Tensor],
    before: Optional[Mapping[str, torch.Tensor]] = None,
) -> None:
    """Print a compact summary for a parameterized-shape optimization run.

    This is only terminal reporting. It does not save files and does not
    participate in autograd.
    """
    step = getattr(run, "final_step", None)
    if before is None and step is not None:
        before = getattr(step, "parameter_values_before", None)
    after = None if step is None else getattr(step, "parameter_values_after", None)

    print(f"optimization_steps: {getattr(run, 'iterations', 'unknown')}")
    print(f"final_loss: {getattr(run, 'final_loss', None)}")

    for name, param in parameters.items():
        before_value = None if before is None or name not in before else _scalar(before[name])
        after_value = _scalar(after[name]) if after is not None and name in after else _scalar(param)
        grad_value = _scalar(param.grad)
        if before_value is None:
            print(f"{name}: {after_value}")
        else:
            print(f"{name}: {before_value} -> {after_value}")
        print(f"dL/d{name}: {grad_value}")


def print_scalar_material_summary(
    run: Any,
    *,
    parameter: torch.Tensor,
    name: Optional[str] = None,
    unit: Optional[str] = None,
) -> None:
    """Print a compact summary for one scalar material optimization run.

    This is only terminal reporting. It does not save files and does not
    participate in autograd.
    """
    step = getattr(run, "final_step", None)
    parameter_name = name or getattr(parameter, "_polyfem_design_name", "E")
    parameter_unit = unit
    before_value = None

    if step is not None:
        before_value = _scalar(getattr(step, "E_value", None))
        parameter_unit = parameter_unit or getattr(step, "E_unit", None)

    after_value = _scalar(parameter)
    grad_value = _scalar(parameter.grad)
    unit_suffix = f" {parameter_unit}" if parameter_unit else ""

    print(f"optimization_steps: {getattr(run, 'iterations', 'unknown')}")
    print(f"final_loss: {getattr(run, 'final_loss', None)}")
    if before_value is None:
        print(f"{parameter_name}: {after_value}{unit_suffix}")
    else:
        print(f"{parameter_name}: {before_value} -> {after_value}{unit_suffix}")
    print(f"dL/d{parameter_name}: {grad_value}")


__all__ = [
    "gradient_norm",
    "print_loss_summary",
    "print_parameterized_shape_summary",
    "print_scalar_material_summary",
]
