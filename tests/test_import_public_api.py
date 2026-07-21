"""Smoke tests for the current public import surface."""


def test_public_api_imports():
    from polyfempy.runtime import Result, solve

    assert solve is not None
    assert Result is not None


def test_runtime_imports_are_the_preferred_solve_surface():
    import polyfempy.runtime as runtime
    from polyfempy.runtime import Result, solve

    assert runtime.CORE_RUNTIME == ["solve", "Result"]
    assert runtime.__all__ == runtime.CORE_RUNTIME
    assert solve is not None
    assert Result is not None


def test_generated_api_package_is_the_preferred_generated_surface():
    from polyfempy.generated_api import generated_api as polyfem
    from polyfempy.generated_api import generated_class

    assert polyfem.Root is generated_class.Root


def test_runtime_recommended_surface_is_small():
    import polyfempy.runtime as runtime

    assert runtime.CORE_RUNTIME == ["solve", "Result"]
    assert runtime.__all__ == runtime.CORE_RUNTIME

    for name in (
        "SimulationConfig",
        "Output",
        "ParaviewOutput",
        "result_output",
        "solve_with_timing",
        "configure_windows_runtime",
        "batch_solve",
    ):
        assert not hasattr(runtime, name)


def test_legacy_api_and_generated_packages_are_removed():
    from pathlib import Path

    package_dir = Path(__file__).resolve().parents[1] / "polyfempy"

    assert not (package_dir / "api").exists()
    assert not (package_dir / "generated").exists()


def test_solve_module_all_only_recommends_solve():
    import importlib

    solve_module = importlib.import_module("polyfempy.runtime.solve")

    assert solve_module.__all__ == ["solve"]
