"""Shared PyTorch optimizer construction helpers."""

from __future__ import annotations

from collections.abc import Iterable

import torch


def make_torch_optimizer(
    parameters: Iterable[torch.nn.Parameter],
    *,
    name: str,
    lr: float,
    empty_error: str = "no optimizer parameters",
) -> torch.optim.Optimizer:
    """Create a small supported PyTorch optimizer from a parameter iterable."""
    params = list(parameters)
    if not params:
        raise ValueError(empty_error)

    opt_name = str(name).strip().lower()
    if opt_name == "sgd":
        return torch.optim.SGD(params, lr=float(lr))
    if opt_name == "adam":
        return torch.optim.Adam(params, lr=float(lr))
    raise ValueError(f"unsupported optimizer {name!r}")


__all__ = ["make_torch_optimizer"]
