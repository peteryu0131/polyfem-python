"""A PyTorch loop shared by direct parameters and differentiable loss closures."""

from dataclasses import dataclass
from typing import Any
import math


@dataclass(frozen=True)
class OptimizationResult:
    """Updated parameters, pre-update history, and the last pre-update loss."""

    parameters: tuple[Any, ...]
    history: tuple[dict[str, Any], ...]
    loss: Any


class Optimization:
    """Evaluate a zero-argument loss afresh on every optimizer iteration.

    Parameters must be leaf tensors. Put mappings from these tensors inside the
    loss closure; never reuse an already evaluated loss or a non-leaf mapping.
    Supports optimizers with a no-argument step(), such as SGD and Adam, not LBFGS.
    """

    def __init__(self, loss, *, parameters):
        import torch

        if not callable(loss):
            raise TypeError("loss must be a callable evaluated on each iteration")
        self.parameters = tuple(parameters)
        if not self.parameters:
            raise ValueError("parameters must not be empty")
        if len({id(p) for p in self.parameters}) != len(self.parameters):
            raise ValueError("parameters must not contain duplicates")
        for p in self.parameters:
            if not isinstance(p, torch.Tensor) or not p.is_leaf or not p.requires_grad:
                raise ValueError("parameters must be trainable leaf tensors; optimize the original parameter")
        self.loss = loss
        self._optimizer = None
        self._steps = None

    @property
    def configured_optimizer(self):
        return self._optimizer

    @property
    def configured_steps(self):
        return self._steps

    def optimizer(self, optimizer, **kwargs):
        import torch

        if isinstance(optimizer, torch.optim.Optimizer):
            if kwargs:
                raise TypeError("pass optimizer options when constructing the optimizer instance")
            configured = optimizer
        else:
            configured = optimizer(self.parameters, **kwargs)
        if not isinstance(configured, torch.optim.Optimizer) or isinstance(configured, torch.optim.LBFGS):
            raise TypeError("use a PyTorch optimizer with a no-argument step(), such as SGD or Adam")
        actual = [p for group in configured.param_groups for p in group["params"]]
        if len(actual) != len(self.parameters) or {id(p) for p in actual} != {id(p) for p in self.parameters}:
            raise ValueError("optimizer must own exactly the supplied parameters")
        self._optimizer = configured
        return self

    def steps(self, steps):
        """Set optimization updates, independently of simulation dt/tend."""
        if isinstance(steps, bool) or not isinstance(steps, int):
            raise TypeError("steps must be an integer")
        if steps <= 0:
            raise ValueError("steps must be positive")
        self._steps = steps
        return self

    def optimize(self, *, record=None, bounds=None):
        """Run updates; record(step, loss) returns extra pre-update history fields.

        Optional bounds are one (lower, upper) scalar pair per parameter tensor.
        They project parameters after each update. Initialize within bounds.
        Re-evaluate your loss after completion if you need the post-update value.
        """
        import torch

        if self._optimizer is None or self._steps is None:
            raise ValueError("configure optimizer(...) and steps(...) before optimize()")
        if bounds is not None:
            bounds = tuple(bounds)
            if len(bounds) != len(self.parameters):
                raise ValueError("bounds must contain one (lower, upper) pair per parameter")
            for p, (lo, hi) in zip(self.parameters, bounds):
                if not math.isfinite(lo) or not math.isfinite(hi) or lo > hi:
                    raise ValueError("bounds must be finite and ordered lower <= upper")
                if not bool(((p >= lo) & (p <= hi)).all()):
                    raise ValueError("initial parameters must be within bounds")
        history = []
        for step in range(self._steps):
            self._optimizer.zero_grad()
            value = self.loss()
            if not isinstance(value, torch.Tensor) or value.numel() != 1 or not bool(torch.isfinite(value).all()):
                raise ValueError("loss must return one finite scalar tensor")
            value.backward()
            if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in self.parameters):
                raise ValueError("loss must produce finite gradients for every supplied parameter")
            row = {"step": step, "loss": float(value.detach()),
                   "gradient_norm": math.sqrt(sum(float(p.grad.detach().square().sum()) for p in self.parameters))}
            if record is not None:
                row.update(record(step, value))
            history.append(row)
            self._optimizer.step()
            if bounds is not None:
                with torch.no_grad():
                    for p, (lo, hi) in zip(self.parameters, bounds):
                        p.clamp_(lo, hi)
        return OptimizationResult(self.parameters, tuple(history), value)


def optimization(loss, *, parameters):
    """Wrap a callable loss and the leaf parameters to optimize."""
    return Optimization(loss, parameters=parameters)
