"""Compatibility exports for older ``polyfempy.api.solve`` imports.

New code should import staged helpers from their owning internal modules:
``_solve_pipeline``, ``_solve_contract``, ``_solve_backend``, or
``_solve_outputs``. This module keeps historical explicit imports working
without making those helpers part of the recommended ``solve.py`` public
surface.
"""

from __future__ import annotations

from . import _solve_pipeline as _p


STAGED_HELPER_NAMES = (
    "RuntimeOptions",
    "NormalizedInputs",
    "NativeOutputs",
    "SolverConfigContext",
    "build_full_json",
    "configure_solver",
    "build_solver",
    "apply_sidesets",
    "apply_sampled_vtu_fallback",
    "extract_native_outputs",
    "finalize_result",
    "normalize_cfg",
    "normalize_mesh_inputs",
    "resolve_runtime_options",
    "run_solver_stage",
)

COMPATIBILITY_ALIAS_TARGETS = {
    "_process_json_config": "process_json_config",
    "_clean_json_for_cpp": "clean_json_for_cpp",
    "_merge_user_cfg_over_full_json": "merge_user_cfg_over_full_json",
    # Historical name; the staged pipeline now resolves all runtime options in
    # one place, so the replacement function has a broader contract.
    "_extract_runtime_output_request": "resolve_runtime_options",
    "_extract_additional_fields": "_extract_additional_fields",
    "_maybe_fill_result_from_temp_vtu": "apply_sampled_vtu_fallback",
    "_finalize_result_output": "finalize_result",
    "_reconstruct_sampled_cauchy_stress": "_reconstruct_sampled_cauchy_stress",
    "_extract_meshio_array": "_extract_meshio_array",
    "_field_available": "_field_available",
}

COMPATIBILITY_ALIASES = tuple(COMPATIBILITY_ALIAS_TARGETS)


def install_solve_compat(module_globals: dict) -> None:
    """Populate ``solve.py`` globals with historical explicit-import helpers."""
    for name in STAGED_HELPER_NAMES:
        module_globals[name] = getattr(_p, name)
    for alias, target in COMPATIBILITY_ALIAS_TARGETS.items():
        module_globals[alias] = getattr(_p, target)


__all__ = [
    "COMPATIBILITY_ALIASES",
    "COMPATIBILITY_ALIAS_TARGETS",
    "STAGED_HELPER_NAMES",
    "install_solve_compat",
]
