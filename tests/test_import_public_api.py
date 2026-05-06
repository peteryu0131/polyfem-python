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


def test_public_api_recommended_surface_is_small():
    import polyfempy.api as api

    assert api.CORE_API == ["solve", "SimulationConfig", "Result"]
    for name in api.CORE_API:
        assert name in api.__all__

    # Internal implementation modules may be importable by tests, but they are
    # not part of the documented public star-import surface.
    assert "_solve_pipeline" not in api.__all__
    assert "_guided_array_mesh" not in api.__all__
    assert "batch_solve" not in api.__all__
    assert not hasattr(api, "batch_solve")


def test_guided_api_imports():
    from polyfempy.api.guided import (
        body_section,
        build_config,
        contact_section,
        material_section,
        simulation_template,
    )

    assert body_section is not None
    assert material_section is not None
    assert contact_section is not None
    assert simulation_template is not None
    assert build_config is not None


def test_guided_recommended_factories_import():
    from polyfempy.api.guided import (
        bodies_section,
        body_section,
        build_config,
        experiment_template,
        fixed_surface_section,
        loads_section,
        material_section,
        output_section,
        problem_section,
        results_section,
        simulation_template,
        solver_section,
        space_section,
        time_section,
        units_section,
    )

    for obj in (
        bodies_section,
        body_section,
        build_config,
        experiment_template,
        fixed_surface_section,
        loads_section,
        material_section,
        output_section,
        problem_section,
        results_section,
        simulation_template,
        solver_section,
        space_section,
        time_section,
        units_section,
    ):
        assert obj is not None


def test_guided_simulation_template_is_generic_public_name():
    import polyfempy.api.guided as g

    assert g.SimulationTemplate is g.ExperimentTemplate
    assert "simulation_template" in g.__all__
    assert "experiment_template" in g.__all__

    body = g.body_section(
        name="body",
        mesh="mesh.msh",
        material=g.material_section(model="NeoHookean", E=1.0, nu=0.3),
    )
    legacy_body = g.body_section(
        name="body",
        mesh="mesh.msh",
        material=g.material_section(model="NeoHookean", E=1.0, nu=0.3),
    )

    template = g.simulation_template(bodies=[body])
    legacy_template = g.experiment_template(
        bodies=[legacy_body]
    )

    assert isinstance(template, g.SimulationTemplate)
    assert isinstance(legacy_template, g.ExperimentTemplate)


def test_solve_compatibility_aliases_are_not_public_all():
    import importlib

    solve_module = importlib.import_module("polyfempy.api.solve")
    for name in solve_module.COMPATIBILITY_ALIASES:
        assert hasattr(solve_module, name)
        assert name not in solve_module.__all__


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
