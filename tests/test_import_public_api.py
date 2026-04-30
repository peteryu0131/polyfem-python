"""Smoke tests for public import paths.

These checks intentionally do not run the C++ backend.  The goal is to ensure
the installed package exposes the documented Python API surface even when the
compiled solver extension is unavailable.
"""


def test_public_api_imports():
    from polyfempy.api import Result, SimulationConfig, solve

    assert solve is not None
    assert SimulationConfig is not None
    assert Result is not None


def test_guided_api_imports():
    from polyfempy.api.guided import (
        body_section,
        build_config,
        contact_section,
        material_section,
    )

    assert body_section is not None
    assert material_section is not None
    assert contact_section is not None
    assert build_config is not None


def test_predefined_problem_helpers_import():
    from polyfempy.api.problems import Problem, get_problem_class

    assert Problem is not None
    assert get_problem_class("Gravity") is not None
    assert get_problem_class("TorsionElastic") is not None


def test_differentiable_api_imports():
    from polyfempy.differentiable import (
        prepare_differentiable_simulation,
        solve_differentiable,
    )

    assert solve_differentiable is not None
    assert prepare_differentiable_simulation is not None
