from __future__ import annotations


def test_backend_adapter_exports_pipeline_runtime_helpers():
    from polyfempy.runtime import _solve_backend as backend

    expected = [
        "SolverConfigContext",
        "build_solver",
        "configure_solver",
        "apply_sidesets",
        "run_solver_stage",
    ]

    for name in expected:
        assert hasattr(backend, name)


def test_solve_module_does_not_reexport_backend_helpers():
    import importlib

    pipeline = importlib.import_module("polyfempy.runtime.solve")

    for name in (
        "SolverConfigContext",
        "build_solver",
        "configure_solver",
        "apply_sidesets",
        "run_solver_stage",
    ):
        assert not hasattr(pipeline, name)


def test_solver_config_context_only_keeps_retouch_state():
    from polyfempy.runtime import _solve_backend as backend

    ctx = backend.SolverConfigContext()

    assert hasattr(ctx, "settings_dict")
    assert hasattr(ctx, "bc")
    assert not hasattr(ctx, "use_json_mode")


def test_run_solver_stage_leaves_basis_and_assembly_to_backend_solve():
    from polyfempy.runtime import _solve_backend as backend

    class FakeSolver:
        def build_basis(self):
            raise AssertionError("Python runtime should not call build_basis")

        def assemble(self):
            raise AssertionError("Python runtime should not call assemble")

        def solve(self, log_level=None):
            return {"_result_bundle": True, "log_level": log_level}

    result = backend.run_solver_stage(
        FakeSolver(),
        {"output": {"log": {"level": "debug"}}},
    )

    assert result == {"_result_bundle": True, "log_level": 1}
