"""Solver entry point. Uses the polyfempy C++ backend.

``solve()`` is the public facade. The staged internals live in
``_solve_pipeline.py`` and its contract/backend/output adapters so config
normalization, solver execution, extraction, fallback, and finalization can be
unit-tested in isolation.

Public contract (signature, return type, raised errors) is unchanged from the
previous version; this refactor is behavior-preserving.
"""

from typing import Callable, Optional

from . import _solve_pipeline as _p

__all__ = ["solve"]


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
