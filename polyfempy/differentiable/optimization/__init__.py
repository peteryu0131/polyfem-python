"""Generic optimization entry points and reporting helpers."""

from .optimizers import make_torch_optimizer
from .problem import (
    OptimizationKind,
    OptimizationProblem,
    OptimizationRunResult,
    make_optimizer,
    prepare_optimization_baseline_simulation,
    prepare_optimization_problem,
    report_optimization_baseline,
    run_optimization,
)

__all__ = [
    "OptimizationKind",
    "OptimizationProblem",
    "OptimizationRunResult",
    "make_optimizer",
    "make_torch_optimizer",
    "prepare_optimization_baseline_simulation",
    "prepare_optimization_problem",
    "report_optimization_baseline",
    "run_optimization",
]
