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


def test_runtime_module_only_exports_workspace_helper():
    import polyfempy.api.runtime as runtime

    assert runtime.__all__ == ["make_timestamped_workspace"]
    assert hasattr(runtime, "make_timestamped_workspace")


def test_report_module_declares_reusable_surface():
    import polyfempy.api.report as report

    assert report.__all__ == [
        "summarize_result",
        "format_result_summary",
        "summarize_history_bundle",
        "format_history_bundle_txt",
        "write_history_bundle_txt",
    ]
    for name in report.__all__:
        assert hasattr(report, name)


def test_solve_module_all_only_recommends_solve():
    import importlib

    solve_module = importlib.import_module("polyfempy.api.solve")

    assert solve_module.__all__ == ["solve"]
