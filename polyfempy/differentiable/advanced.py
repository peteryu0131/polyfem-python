"""Advanced helpers for differentiable PolyFEM workflows.

Most user scripts should import from ``polyfempy.differentiable`` directly.
This module exposes debugging probes, finite-difference checks, config parsing
helpers, and material diagnostics without making those names part of the
recommended API path.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_MATERIAL_CONFIG_NAMES = {
    "as_materials_list",
    "material_for_body",
    "material_id",
    "nu_from_material",
    "other_material_for_body",
    "value_and_unit",
    "youngs_from_material",
}

_MATERIAL_DIAGNOSTIC_NAMES = {
    "apply_finite_difference_gradient_fallback",
    "collect_material_chain_diagnostics",
    "evaluate_material_loss_value_for_E",
    "finite_difference_material_E_gradient",
    "format_optional",
    "objective_solution_rhs_diagnostics",
    "print_material_chain_diagnostics",
    "print_objective_rhs_diagnostics",
    "usable_scalar_gradient",
}

_EXPORT_MODULES = {
    **{name: ".material_config" for name in _MATERIAL_CONFIG_NAMES},
    **{name: ".material_diagnostics" for name in _MATERIAL_DIAGNOSTIC_NAMES},
}

__all__ = sorted(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    """Load advanced helpers only when the caller asks for them."""
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(module_name, package=__package__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *__all__])
