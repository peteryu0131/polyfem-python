"""Material-specific differentiable optimization helpers."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_MODULES = {
    "ScalarMaterialOptimizationProblem": ".optimization",
    "ScalarMaterialOptimizationStep": ".optimization",
    "format_scalar_material_optimization_history_summary": ".optimization",
    "format_scalar_material_optimization_step": ".optimization",
    "make_material_optimizer": ".optimization",
    "prepare_material_differentiable_simulation": ".optimization",
    "prepare_material_optimization_problem": ".optimization",
    "prepare_scalar_youngs_material_problem": ".optimization",
    "run_scalar_material_optimization": ".optimization",
    "as_materials_list": ".config",
    "material_for_body": ".config",
    "material_id": ".config",
    "nu_from_material": ".config",
    "other_material_for_body": ".config",
    "value_and_unit": ".config",
    "youngs_from_material": ".config",
    "apply_finite_difference_gradient_fallback": ".diagnostics",
    "collect_material_chain_diagnostics": ".diagnostics",
    "evaluate_material_loss_value_for_E": ".diagnostics",
    "finite_difference_material_E_gradient": ".diagnostics",
    "format_optional": ".diagnostics",
    "objective_solution_rhs_diagnostics": ".diagnostics",
    "print_material_chain_diagnostics": ".diagnostics",
    "print_objective_rhs_diagnostics": ".diagnostics",
    "usable_scalar_gradient": ".diagnostics",
    "build_lame_from_youngs": ".parameters",
    "solver_body_ids_for_assembly": ".parameters",
    "solver_body_slot_mask": ".parameters",
    "youngs_to_lame": ".parameters",
    "youngs_value_to_internal": ".parameters",
}

__all__ = sorted(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, package=__name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *__all__])
