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


def test_pipeline_reexports_backend_helpers_for_compatibility():
    from polyfempy.runtime import _solve_backend as backend
    from polyfempy.runtime import _solve_pipeline as pipeline

    assert pipeline.SolverConfigContext is backend.SolverConfigContext
    assert pipeline.build_solver is backend.build_solver
    assert pipeline.configure_solver is backend.configure_solver
    assert pipeline.apply_sidesets is backend.apply_sidesets
    assert pipeline.run_solver_stage is backend.run_solver_stage
