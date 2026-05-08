"""Material-parameter helpers for differentiable PolyFEM solves."""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np

try:
    import torch  # pyright: ignore[reportMissingImports]

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for differentiable simulations. "
            "Please install PyTorch: pip install torch"
        )


def _solver_n_element_assembly_slots(solver: Any) -> int:
    """Length required by ``set_per_element_material``."""
    if hasattr(solver, "n_element_assembly_slots"):
        return int(solver.n_element_assembly_slots())
    if hasattr(solver, "n_bases"):
        return int(solver.n_bases())
    raise AttributeError("solver does not expose n_element_assembly_slots() or n_bases()")


def solver_body_ids_for_assembly(solver: Any) -> np.ndarray:
    """Return body ids aligned with per-element assembly slots."""
    n_slots = _solver_n_element_assembly_slots(solver)
    if hasattr(solver, "get_body_ids_for_assembly"):
        ids = np.asarray(solver.get_body_ids_for_assembly(), dtype=np.int32).reshape(-1)
        if int(ids.size) != int(n_slots):
            raise RuntimeError(
                f"get_body_ids_for_assembly length {ids.size} != n_element_assembly_slots {n_slots}"
            )
        return ids

    mesh = solver.mesh()
    body_ids = np.asarray(mesh.get_body_ids(), dtype=np.int32).reshape(-1)
    n_el = int(mesh.n_elements())
    if body_ids.size != n_el:
        raise RuntimeError(
            f"get_body_ids length {body_ids.size} != n_elements {n_el}; check mesh / PolyFEM version."
        )
    if n_el < n_slots:
        raise RuntimeError(
            f"mesh n_elements={n_el} < assembly slots {n_slots}; cannot build material vectors."
        )
    return body_ids[:n_slots]


def solver_body_slot_mask(solver: Any, *, body_id: int) -> "torch.Tensor":
    """Boolean mask over assembly slots for one body id."""
    _require_torch()
    body_ids = solver_body_ids_for_assembly(solver)
    return torch.as_tensor(body_ids == int(body_id), dtype=torch.bool)


def youngs_to_lame(
    E: "torch.Tensor",
    nu: Union[float, "torch.Tensor"],
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Convert isotropic ``E, nu`` parameters to Lamé ``lambda, mu``."""
    _require_torch()
    nu_t = torch.as_tensor(nu, dtype=E.dtype, device=E.device)
    lam = E * nu_t / ((1.0 + nu_t) * (1.0 - 2.0 * nu_t))
    mu = E / (2.0 * (1.0 + nu_t))
    return lam, mu


def _pressure_unit_to_internal_scale(
    *,
    units_length: str,
    units_mass: str,
    units_time: str,
    pressure_unit: str,
) -> float:
    """Scale a named pressure unit into the current solver unit system."""
    length_to_m = {"m": 1.0, "cm": 1e-2, "mm": 1e-3}
    mass_to_kg = {"kg": 1.0, "g": 1e-3}
    time_to_s = {"s": 1.0}
    pressure_to_pa = {"Pa": 1.0, "kPa": 1e3, "MPa": 1e6, "GPa": 1e9}

    try:
        length_scale = length_to_m[str(units_length)]
        mass_scale = mass_to_kg[str(units_mass)]
        time_scale = time_to_s[str(units_time)]
        pressure_pa = pressure_to_pa[str(pressure_unit)]
    except KeyError as exc:
        raise ValueError(
            "unsupported unit conversion for differentiable material solve: "
            f"length={units_length!r}, mass={units_mass!r}, time={units_time!r}, "
            f"pressure={pressure_unit!r}"
        ) from exc

    internal_pressure_pa = mass_scale / (length_scale * (time_scale ** 2))
    return pressure_pa / internal_pressure_pa


def _units_triplet_from_settings(settings: dict[str, Any]) -> tuple[str, str, str]:
    units = settings.get("units", {})
    if not isinstance(units, dict):
        units = {}
    return (
        str(units.get("length", "m")),
        str(units.get("mass", "kg")),
        str(units.get("time", "s")),
    )


def youngs_value_to_internal(
    value: Union[float, "torch.Tensor"],
    *,
    pressure_unit: str,
    solver_settings: dict[str, Any],
) -> Union[float, "torch.Tensor"]:
    """Convert user-facing ``E`` values (MPa/GPa/...) into solver internal units."""
    units_length, units_mass, units_time = _units_triplet_from_settings(solver_settings)
    scale = _pressure_unit_to_internal_scale(
        units_length=units_length,
        units_mass=units_mass,
        units_time=units_time,
        pressure_unit=str(pressure_unit),
    )
    return value * scale


def _expand_material_parameter_to_slots(
    value: Union[float, "torch.Tensor"],
    *,
    n_slots: int,
    dtype: "torch.dtype",
    device: "torch.device",
) -> "torch.Tensor":
    """Expand a scalar or per-slot material parameter to ``(n_slots,)``."""
    _require_torch()
    t = torch.as_tensor(value, dtype=dtype, device=device)
    if t.ndim == 0:
        return t.reshape(()).expand(n_slots)
    if t.numel() != int(n_slots):
        raise ValueError(
            f"material parameter has {t.numel()} values but expected 1 or {n_slots}"
        )
    return t.reshape(n_slots)


def build_lame_from_youngs(
    E: Union[float, "torch.Tensor"],
    nu: Union[float, "torch.Tensor"],
    *,
    slot_mask: Optional["torch.Tensor"] = None,
    other_E: Optional[Union[float, "torch.Tensor"]] = None,
    other_nu: Optional[Union[float, "torch.Tensor"]] = None,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Build Lamé tensors from ``E, nu`` for one or two material regions."""
    _require_torch()
    E_t = torch.as_tensor(E, dtype=torch.get_default_dtype())
    if slot_mask is None:
        return youngs_to_lame(E_t, nu)

    if other_E is None or other_nu is None:
        raise ValueError("slot_mask requires other_E and other_nu for the non-masked region")

    mask = torch.as_tensor(slot_mask, dtype=torch.bool, device=E_t.device).reshape(-1)
    n_slots = int(mask.numel())

    E_primary = _expand_material_parameter_to_slots(
        E,
        n_slots=n_slots,
        dtype=E_t.dtype,
        device=E_t.device,
    )
    nu_primary = _expand_material_parameter_to_slots(
        nu,
        n_slots=n_slots,
        dtype=E_t.dtype,
        device=E_t.device,
    )
    E_secondary = _expand_material_parameter_to_slots(
        other_E,
        n_slots=n_slots,
        dtype=E_t.dtype,
        device=E_t.device,
    )
    nu_secondary = _expand_material_parameter_to_slots(
        other_nu,
        n_slots=n_slots,
        dtype=E_t.dtype,
        device=E_t.device,
    )

    E_full = torch.where(mask, E_primary, E_secondary)
    nu_full = torch.where(mask, nu_primary, nu_secondary)
    return youngs_to_lame(E_full, nu_full)


__all__ = [
    "build_lame_from_youngs",
    "solver_body_ids_for_assembly",
    "solver_body_slot_mask",
    "youngs_to_lame",
    "youngs_value_to_internal",
]
