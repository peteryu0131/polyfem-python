from __future__ import annotations

from dataclasses import dataclass

import pytest

from polyfempy.differentiable.optimization.result import (
    OptimizationRunResult as InternalOptimizationRunResult,
)

OptimizationRunResult = InternalOptimizationRunResult


def test_optimization_run_result_is_reexported_from_runner():
    pytest.importorskip("torch", exc_type=ImportError)

    from polyfempy.differentiable.optimization.runner import (
        OptimizationRunResult as RunnerOptimizationRunResult,
    )

    assert RunnerOptimizationRunResult is InternalOptimizationRunResult


@dataclass
class _Step:
    iteration: int
    loss: float
    step_norm: float = 0.0
    max_vertex_update: float = 0.0


def test_optimization_run_result_reports_final_and_best_loss():
    steps = [
        _Step(iteration=0, loss=5.0),
        _Step(iteration=1, loss=3.0),
        _Step(iteration=2, loss=4.0),
    ]
    run = OptimizationRunResult(
        problem=object(),
        steps=steps,
        success=True,
        message="completed 3 optimization steps",
    )

    assert run.iterations == 3
    assert run.final_step is steps[2]
    assert run.final_loss == 4.0
    assert run.best_step is steps[1]
    assert run.best_loss == 3.0
    assert run.best_iteration == 1
    assert run.summary()["success"] is True
    assert run.summary()["message"] == "completed 3 optimization steps"


def test_empty_optimization_run_result_is_json_friendly():
    run = OptimizationRunResult(
        problem=object(),
        steps=[],
        success=True,
        message="completed 0 optimization steps",
    )
    summary = run.summary()

    assert run.final_step is None
    assert run.final_loss is None
    assert run.best_step is None
    assert run.best_loss is None
    assert summary["optimization_steps"] == 0
    assert summary["final_loss"] is None
    assert summary["best_loss"] is None
