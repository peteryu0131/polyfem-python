"""Smoke tests for the current public import surface."""


def test_public_api_imports():
    from polyfempy.api import Result, solve

    assert solve is not None
    assert Result is not None


def test_public_api_recommended_surface_is_small():
    import polyfempy.api as api

    assert api.CORE_API == ["solve", "Result"]
    assert api.__all__ == api.CORE_API

    for name in (
        "SimulationConfig",
        "Output",
        "ParaviewOutput",
        "result_output",
        "solve_with_timing",
        "configure_windows_runtime",
        "batch_solve",
    ):
        assert not hasattr(api, name)


def test_no_example_runtime_helper_module_in_core_api():
    from pathlib import Path

    api_dir = Path(__file__).resolve().parents[1] / "polyfempy" / "api"
    assert not (api_dir / "runtime.py").exists()


def test_solve_module_all_only_recommends_solve():
    import importlib

    solve_module = importlib.import_module("polyfempy.api.solve")

    assert solve_module.__all__ == ["solve"]
