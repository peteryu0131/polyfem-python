from __future__ import annotations


def test_output_adapter_exports_pipeline_output_helpers():
    from polyfempy.runtime import _solve_outputs as outputs

    expected = [
        "NativeOutputs",
        "extract_native_outputs",
        "apply_sampled_vtu_fallback",
        "finalize_result",
    ]

    for name in expected:
        assert hasattr(outputs, name)


def test_pipeline_reexports_output_helpers_for_compatibility():
    from polyfempy.runtime import _solve_outputs as outputs
    from polyfempy.runtime import _solve_pipeline as pipeline

    assert pipeline.NativeOutputs is outputs.NativeOutputs
    assert pipeline.extract_native_outputs is outputs.extract_native_outputs
    assert pipeline.apply_sampled_vtu_fallback is outputs.apply_sampled_vtu_fallback
    assert pipeline.finalize_result is outputs.finalize_result
