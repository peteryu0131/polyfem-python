"""Material-config parsing helpers for scalar material optimization."""

from __future__ import annotations

from typing import Any, Optional


def as_materials_list(settings: dict[str, Any]) -> list[dict[str, Any]]:
    """Return cfg materials as a normalized list of dictionaries."""
    materials = settings.get("materials", [])
    if isinstance(materials, dict):
        materials = [materials]
    if not isinstance(materials, list):
        return []
    return [dict(item) for item in materials if isinstance(item, dict)]


def material_id(material: dict[str, Any]) -> int | None:
    """Return one material id, or None when the material does not identify one body."""
    raw = material.get("id")
    if isinstance(raw, list):
        if len(raw) != 1:
            return None
        raw = raw[0]
    if raw in (None, ""):
        return None
    return int(raw)


def material_for_body(settings: dict[str, Any], *, body_id: int) -> dict[str, Any]:
    """Find the material record attached to one body id."""
    for material in as_materials_list(settings):
        if material_id(material) == int(body_id):
            return material
    raise ValueError(f"could not find material with id/body_id={body_id!r} in cfg.materials")


def other_material_for_body(
    settings: dict[str, Any],
    *,
    body_id: int,
    other_body_id: Optional[int] = None,
) -> dict[str, Any] | None:
    """Infer the fixed non-design material for the current scalar-E examples."""
    if other_body_id is not None:
        return material_for_body(settings, body_id=int(other_body_id))

    others = [
        material
        for material in as_materials_list(settings)
        if material_id(material) != int(body_id)
    ]
    if not others:
        return None
    if len(others) == 1:
        return others[0]
    ids = [material_id(material) for material in others]
    raise ValueError(
        "material optimization can only infer one non-design material; "
        f"found other material ids {ids}. Pass other_body_id or explicit other_E_value/other_nu."
    )


def value_and_unit(raw: Any, *, default_unit: str = "Pa") -> tuple[float, str]:
    """Parse either a raw value or a PolyFEM-style value/unit object."""
    if isinstance(raw, dict):
        if "value" in raw:
            unit = str(raw.get("unit", default_unit))
            return float(raw["value"]), unit
        if "amount" in raw:
            unit = str(raw.get("unit", default_unit))
            return float(raw["amount"]), unit
    return float(raw), default_unit


def youngs_from_material(
    material: dict[str, Any],
    *,
    default_unit: str = "Pa",
) -> tuple[float, str]:
    """Read Young's modulus and unit from one material section."""
    for key in ("E", "e", "young", "youngs", "youngs_modulus", "young_modulus"):
        if key in material:
            return value_and_unit(material[key], default_unit=default_unit)
    raise ValueError(f"material id={material.get('id')} does not define Young's modulus E")


def nu_from_material(material: dict[str, Any]) -> float:
    """Read Poisson ratio from one material section."""
    for key in ("nu", "poisson", "poisson_ratio"):
        if key in material:
            return float(material[key])
    raise ValueError(f"material id={material.get('id')} does not define Poisson ratio nu")


__all__ = [
    "as_materials_list",
    "material_for_body",
    "material_id",
    "nu_from_material",
    "other_material_for_body",
    "value_and_unit",
    "youngs_from_material",
]
