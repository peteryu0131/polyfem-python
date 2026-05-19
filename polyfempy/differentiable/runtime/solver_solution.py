"""Solver solution extraction helpers for differentiable autograd bridges."""

from __future__ import annotations

from typing import Any

import numpy as np


def solver_solution_array(
    solver: Any,
    solve_return: Any,
    *,
    error_message: str,
) -> np.ndarray:
    """Return the displacement array produced by a PolyFEM solve.

    Prefer ``solver.get_solutions()`` because it matches the cached transient
    layout used by adjoint solves. Older bindings and older examples may only
    expose the solution through the solve return value or the solution cache, so
    those paths remain as compatibility fallbacks.
    """
    solutions_np = None
    if hasattr(solver, "get_solutions"):
        try:
            solutions_np = np.asarray(solver.get_solutions())
            if solutions_np.size == 0:
                solutions_np = None
        except Exception:
            solutions_np = None

    if (
        solutions_np is None
        and isinstance(solve_return, dict)
        and solve_return.get("_result_bundle")
        and "u" in solve_return
    ):
        solutions_np = np.asarray(solve_return["u"])
    elif (
        solutions_np is None
        and isinstance(solve_return, (tuple, list))
        and len(solve_return) > 0
    ):
        solutions_np = np.asarray(solve_return[0])
    elif solutions_np is None and hasattr(solver, "get_solution_cache"):
        cache = solver.get_solution_cache()
        if hasattr(cache, "solution"):
            solutions_np = np.asarray(cache.solution(0))
        elif hasattr(cache, "displacement"):
            solutions_np = np.asarray(cache.displacement(0))

    if solutions_np is None:
        raise RuntimeError(error_message)
    return solutions_np


__all__ = ["solver_solution_array"]
