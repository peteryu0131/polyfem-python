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
    assert api.__all__ == api.CORE_API

    # Internal implementation modules may be importable by tests, but they are
    # not part of the documented public star-import surface.
    assert "_solve_pipeline" not in api.__all__
    assert "_guided_array_mesh" not in api.__all__
    assert "batch_solve" not in api.__all__
    assert not hasattr(api, "batch_solve")


def test_public_api_advanced_compat_names_are_explicit_only():
    import polyfempy.api as api

    assert "Solver" in api.ADVANCED_COMPAT_API
    assert "Solver" not in api.__all__
    assert "result_output" not in api.__all__
    assert api.Solver is not None
    assert api.result_output is not None


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


def test_guided_all_is_factory_surface_only():
    import polyfempy.api.guided as g

    assert g.__all__ == g.GUIDED_CORE_API
    for name in (
        "simulation_template",
        "body_section",
        "material_section",
        "solver_section",
        "time_section",
        "results_section",
        "build_config",
    ):
        assert name in g.__all__

    for name in (
        "SimulationTemplate",
        "ExperimentTemplate",
        "BodySection",
        "MaterialSection",
        "MaterialModelName",
        "experiment_template",
    ):
        assert hasattr(g, name)
        assert name not in g.__all__


def test_guided_simulation_template_is_generic_public_name():
    import polyfempy.api.guided as g

    assert g.SimulationTemplate is g.ExperimentTemplate
    assert "simulation_template" in g.__all__
    assert "experiment_template" not in g.__all__

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


def test_solve_module_all_only_recommends_solve():
    import importlib

    solve_module = importlib.import_module("polyfempy.api.solve")

    assert solve_module.__all__ == ["solve"]


def test_solve_staged_helpers_remain_explicit_import_compat_only():
    import importlib

    solve_module = importlib.import_module("polyfempy.api.solve")
    helper_names = [
        "RuntimeOptions",
        "NormalizedInputs",
        "NativeOutputs",
        "SolverConfigContext",
        "configure_solver",
        "extract_native_outputs",
    ]

    for name in helper_names:
        assert hasattr(solve_module, name)
        assert name not in solve_module.__all__


def test_solve_compatibility_aliases_point_to_pipeline_targets():
    import importlib

    pipeline = importlib.import_module("polyfempy.api._solve_pipeline")
    solve_module = importlib.import_module("polyfempy.api.solve")

    assert tuple(solve_module.COMPATIBILITY_ALIAS_TARGETS) == solve_module.COMPATIBILITY_ALIASES
    for alias, target in solve_module.COMPATIBILITY_ALIAS_TARGETS.items():
        assert getattr(solve_module, alias) is getattr(pipeline, target)


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


def test_differentiable_compatibility_api_is_not_public_all():
    import polyfempy.differentiable as diff

    assert hasattr(diff, "COMPATIBILITY_API")
    for name in diff.COMPATIBILITY_API:
        assert hasattr(diff, name)
        assert name not in diff.__all__


def test_differentiable_export_module_map_matches_declared_surface():
    from polyfempy.differentiable._exports import (
        COMPATIBILITY_API,
        EXPORT_MODULES,
        PUBLIC_API,
    )

    declared = set(PUBLIC_API) | set(COMPATIBILITY_API)
    assert set(EXPORT_MODULES) == declared


def test_differentiable_runtime_and_data_package_paths_preserve_old_imports():
    import importlib

    checks = [
        ("polyfempy.differentiable.solve_diff", "polyfempy.differentiable.runtime.solve", "solve_differentiable"),
        ("polyfempy.differentiable.solve_diff", "polyfempy.differentiable.runtime.solve", "prepare_differentiable_simulation"),
        ("polyfempy.differentiable._solve_settings", "polyfempy.differentiable.runtime.settings", "prepare_differentiable_solve_contract"),
        ("polyfempy.differentiable._solve_settings", "polyfempy.differentiable.runtime.settings", "prepare_settings_only_differentiable_contract"),
        ("polyfempy.differentiable._solve_settings", "polyfempy.differentiable.runtime.settings", "build_solver_from_settings"),
        ("polyfempy.differentiable.torch_integration", "polyfempy.differentiable.runtime.autograd", "PolyFEMFunction"),
        ("polyfempy.differentiable.result_diff", "polyfempy.differentiable.runtime.result", "DifferentiableResult"),
        ("polyfempy.differentiable.cpp_ext", "polyfempy.differentiable.runtime.cpp_ext", "get_cpp_polyfempy"),
        ("polyfempy.differentiable.training_data", "polyfempy.differentiable.data.training", "save_training_sample"),
    ]

    for old_module_name, new_module_name, attr in checks:
        old_module = importlib.import_module(old_module_name)
        new_module = importlib.import_module(new_module_name)
        assert getattr(old_module, attr) is getattr(new_module, attr)


def test_differentiable_exports_prefer_runtime_and_data_modules():
    from polyfempy.differentiable._exports import EXPORT_MODULES

    expected = {
        "prepare_differentiable_simulation": ".runtime.solve",
        "solve_differentiable": ".runtime.solve",
        "solve_differentiable_material": ".runtime.solve",
        "build_solver_from_settings": ".runtime.settings",
        "DifferentiableResult": ".runtime.result",
        "DifferentiableMaterialResult": ".runtime.result",
        "save_training_sample": ".data.training",
    }
    for name, module_path in expected.items():
        assert EXPORT_MODULES[name] == module_path


def test_differentiable_objective_and_optimization_paths_preserve_old_imports():
    import importlib

    checks = [
        ("polyfempy.differentiable.objective_bridge", "polyfempy.differentiable.objectives.bridge", "make_von_mises_loss"),
        ("polyfempy.differentiable.objective_bridge", "polyfempy.differentiable.objectives.bridge", "ObjectiveLossResult"),
        ("polyfempy.differentiable._objective_common", "polyfempy.differentiable.objectives.common", "resolve_objective_state_column"),
        ("polyfempy.differentiable.optimization_problem", "polyfempy.differentiable.optimization.problem", "prepare_optimization_problem"),
        ("polyfempy.differentiable.optimization_problem", "polyfempy.differentiable.optimization.problem", "OptimizationRunResult"),
        ("polyfempy.differentiable.optimization_runner", "polyfempy.differentiable.optimization.runner", "run_optimization"),
        ("polyfempy.differentiable._optimization_result", "polyfempy.differentiable.optimization.result", "OptimizationRunResult"),
        ("polyfempy.differentiable._optimization_reports", "polyfempy.differentiable.optimization.reports", "OptimizationReportWriter"),
        ("polyfempy.differentiable.summary", "polyfempy.differentiable.optimization.summary", "gradient_norm"),
    ]

    for old_module_name, new_module_name, attr in checks:
        old_module = importlib.import_module(old_module_name)
        new_module = importlib.import_module(new_module_name)
        assert getattr(old_module, attr) is getattr(new_module, attr)


def test_differentiable_exports_prefer_objectives_and_optimization_modules():
    from polyfempy.differentiable._exports import EXPORT_MODULES

    expected = {
        "ObjectiveLossResult": ".objectives.bridge",
        "make_von_mises_loss": ".objectives.bridge",
        "make_stress_norm_loss": ".objectives.bridge",
        "create_polyfem_objective": ".objectives.bridge",
        "OptimizationKind": ".optimization.problem",
        "OptimizationProblem": ".optimization.problem",
        "OptimizationRunResult": ".optimization.problem",
        "make_optimizer": ".optimization.problem",
        "run_optimization": ".optimization.problem",
        "gradient_norm": ".optimization.summary",
    }
    for name, module_path in expected.items():
        assert EXPORT_MODULES[name] == module_path


def test_differentiable_shape_and_material_paths_preserve_old_imports():
    import importlib

    checks = [
        ("polyfempy.differentiable.shape_optimization", "polyfempy.differentiable.shape.optimization", "prepare_shape_optimization_problem"),
        ("polyfempy.differentiable.shape_optimization", "polyfempy.differentiable.shape.optimization", "prepare_parameterized_shape_problem"),
        ("polyfempy.differentiable.shape_optimization", "polyfempy.differentiable.shape.optimization", "run_shape_optimization"),
        ("polyfempy.differentiable.shape_problem", "polyfempy.differentiable.shape.problem", "ShapeOptimizationProblem"),
        ("polyfempy.differentiable.shape_mask", "polyfempy.differentiable.shape.mask", "body_vertex_mask"),
        ("polyfempy.differentiable.geometry_maps", "polyfempy.differentiable.shape.geometry_maps", "vertices_y_le"),
        ("polyfempy.differentiable.material_optimization", "polyfempy.differentiable.material.optimization", "prepare_material_optimization_problem"),
        ("polyfempy.differentiable.material_config", "polyfempy.differentiable.material.config", "material_for_body"),
        ("polyfempy.differentiable.material_diagnostics", "polyfempy.differentiable.material.diagnostics", "usable_scalar_gradient"),
        ("polyfempy.differentiable._material_parameters", "polyfempy.differentiable.material.parameters", "youngs_to_lame"),
    ]

    for old_module_name, new_module_name, attr in checks:
        old_module = importlib.import_module(old_module_name)
        new_module = importlib.import_module(new_module_name)
        assert getattr(old_module, attr) is getattr(new_module, attr)


def test_differentiable_exports_prefer_shape_and_material_modules():
    from polyfempy.differentiable._exports import EXPORT_MODULES

    expected = {
        "prepare_parameterized_shape_problem": ".shape.optimization",
        "body_vertex_mask": ".shape.mask",
        "shape_gradient_for_body": ".shape.mask",
        "relative_scale": ".shape.geometry_maps",
        "ShapeOptimizationProblem": ".shape.optimization",
        "run_shape_optimization": ".shape.optimization",
        "prepare_material_optimization_problem": ".material.optimization",
        "run_scalar_material_optimization": ".material.optimization",
        "youngs_to_lame": ".material.parameters",
        "usable_scalar_gradient": ".material.diagnostics",
    }
    for name, module_path in expected.items():
        assert EXPORT_MODULES[name] == module_path
