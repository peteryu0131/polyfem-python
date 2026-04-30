"""Advanced and legacy helpers for differentiable PolyFEM workflows.

Most user scripts should import from ``polyfempy.differentiable`` directly.
This module exposes debugging probes, finite-difference checks, config parsing
helpers, and the older torch-bridge experiment helpers without making those
names part of the recommended API path.

Exports are loaded lazily. Importing this module does not load the legacy
torch bridge unless a torch-bridge symbol is actually requested.
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

_TORCH_BRIDGE_NAMES = {
    "TORCH_BRIDGE_METHOD_NAME",
    "TORCH_BRIDGE_METHOD_PATTERN",
    "PolyFEMTorchBridgeOptimizerProbe",
    "PolyFEMTorchBridgeOptimizerStep",
    "PolyFEMTorchBridgeStep",
    "apply_differentiable_runtime_patches",
    "array_summary",
    "console_log_level_from_settings",
    "evaluate_polyfem_loss",
    "gradient_summary",
    "make_torch_optimizer",
    "print_polyfem_bridge_optimizer_probe_summary",
    "print_polyfem_bridge_step_summary",
    "run_backward_if_requested",
    "run_polyfem_bridge_optimizer_probe",
    "run_polyfem_bridge_step",
    "run_polyfem_differentiable_forward",
    "solver_set_log_level_off",
    "summarize_gradient_norm",
    "write_polyfem_bridge_optimizer_probe_report",
    "write_polyfem_bridge_step_report",
}

_EXPORT_MODULES = {
    **{name: ".material_config" for name in _MATERIAL_CONFIG_NAMES},
    **{name: ".material_diagnostics" for name in _MATERIAL_DIAGNOSTIC_NAMES},
    **{name: ".torch_bridge" for name in _TORCH_BRIDGE_NAMES},
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
