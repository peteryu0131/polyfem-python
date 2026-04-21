"""Solver entry point. Uses the polyfempy C++ backend.

``solve()`` is a thin orchestrator. All real work lives in the staged pipeline
in ``_solve_pipeline.py`` so each stage (cfg normalization, solver build, native
extraction, sampled-VTU fallback, finalization) can be unit-tested in isolation.

Public contract (signature, return type, raised errors) is unchanged from the
previous version; this refactor is behavior-preserving.
"""

from typing import Callable, Optional

from . import _solve_pipeline as _p
from ._solve_pipeline import (
    RuntimeOptions,
    NormalizedInputs,
    NativeOutputs,
    SolverConfigContext,
    build_full_json,
    configure_solver,
    build_solver,
    apply_sidesets,
    apply_sampled_vtu_fallback,
    extract_native_outputs,
    finalize_result,
    normalize_cfg,
    normalize_mesh_inputs,
    resolve_runtime_options,
    run_solver_stage,
)

__all__ = [
    "solve",
    "RuntimeOptions",
    "NormalizedInputs",
    "NativeOutputs",
    "SolverConfigContext",
]


# ---------------------------------------------------------------------------
# Backward-compat aliases for internal callers that previously imported the
# private helpers from this module. New code should use ``_solve_pipeline``
# directly.
# ---------------------------------------------------------------------------

_process_json_config = _p.process_json_config
_clean_json_for_cpp = _p.clean_json_for_cpp
_merge_user_cfg_over_full_json = _p.merge_user_cfg_over_full_json
_extract_runtime_output_request = _p.resolve_runtime_options  # signature slightly different
_extract_additional_fields = _p._extract_additional_fields
_maybe_fill_result_from_temp_vtu = _p.apply_sampled_vtu_fallback
_finalize_result_output = _p.finalize_result
_reconstruct_sampled_cauchy_stress = _p._reconstruct_sampled_cauchy_stress
_extract_meshio_array = _p._extract_meshio_array
_field_available = _p._field_available


def solve(
    vertices=None,
    cells=None,
    cfg=None,
    sidesets_func: Optional[Callable] = None,
    dtype=None,
    sampled_vtu_fallback: Optional[bool] = None,
):
    """Run PolyFEM solve.

    Either provide ``vertices`` and ``cells`` arrays, or pass a ``cfg`` that
    contains a full ``geometry`` block (JSON mode). Returns a ``Result``.

    This function orchestrates the staged pipeline in ``_solve_pipeline.py``:

        normalize_cfg -> build_full_json -> resolve_runtime_options ->
        normalize_mesh_inputs -> build_solver -> configure_solver ->
        apply_sidesets -> run_solver_stage -> extract_native_outputs ->
        apply_sampled_vtu_fallback -> finalize_result
    """
    return _p.run_pipeline(
        vertices=vertices,
        cells=cells,
        cfg=cfg,
        sidesets_func=sidesets_func,
        dtype=dtype,
        sampled_vtu_fallback=sampled_vtu_fallback,
    )
