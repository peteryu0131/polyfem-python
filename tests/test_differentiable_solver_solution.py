from __future__ import annotations

import numpy as np
import pytest

from polyfempy.differentiable._solver_solution import solver_solution_array


class _GetterSolver:
    def get_solutions(self):
        return np.array([[1.0, 2.0]])


class _EmptyGetterSolver:
    def get_solutions(self):
        return np.array([])


class _SolutionCache:
    def solution(self, index):
        assert index == 0
        return np.array([7.0, 8.0])


class _DisplacementCache:
    def displacement(self, index):
        assert index == 0
        return np.array([9.0, 10.0])


class _CacheSolver:
    def __init__(self, cache):
        self._cache = cache

    def get_solution_cache(self):
        return self._cache


def test_solver_solution_prefers_get_solutions_over_solve_return():
    out = solver_solution_array(
        _GetterSolver(),
        {"_result_bundle": True, "u": np.array([[99.0]])},
        error_message="missing",
    )

    assert out.tolist() == [[1.0, 2.0]]


def test_solver_solution_uses_result_bundle_when_getter_empty():
    out = solver_solution_array(
        _EmptyGetterSolver(),
        {"_result_bundle": True, "u": np.array([[3.0, 4.0]])},
        error_message="missing",
    )

    assert out.tolist() == [[3.0, 4.0]]


def test_solver_solution_uses_tuple_fallback():
    out = solver_solution_array(
        object(),
        (np.array([5.0, 6.0]),),
        error_message="missing",
    )

    assert out.tolist() == [5.0, 6.0]


def test_solver_solution_uses_solution_cache_fallbacks():
    assert solver_solution_array(
        _CacheSolver(_SolutionCache()),
        None,
        error_message="missing",
    ).tolist() == [7.0, 8.0]
    assert solver_solution_array(
        _CacheSolver(_DisplacementCache()),
        None,
        error_message="missing",
    ).tolist() == [9.0, 10.0]


def test_solver_solution_raises_requested_error_message():
    with pytest.raises(RuntimeError, match="custom missing solution"):
        solver_solution_array(object(), None, error_message="custom missing solution")
