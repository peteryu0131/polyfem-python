"""Small summary helpers for differentiable experiments."""

# pyright: reportMissingImports=false

from __future__ import annotations

from typing import Optional

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


__all__ = ["gradient_norm", "print_loss_summary"]
