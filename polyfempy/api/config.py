"""Typed configuration model for the public PolyFEM Python API.

This module is intentionally broad: it preserves the stable import path
``polyfempy.api.config`` for ``SimulationConfig`` and the typed config blocks
used by existing scripts. Keep behavior changes narrow here; full/minimal JSON
round-trips and ``from polyfempy.api import SimulationConfig`` are public
contracts.
"""

from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Optional, TYPE_CHECKING, Union, List, Dict, Any, overload
import copy
import json
import warnings

if TYPE_CHECKING:
    from .selection import Selection
    # Forward references for classes defined later in this file
    from typing import TYPE_CHECKING as _TYPE_CHECKING
else:
    _TYPE_CHECKING = False


# ============================================================================
# Internal Normalization and Compatibility Helpers
# ============================================================================


def _merge_dicts_deep(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins on conflicts)."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_dicts_deep(result[key], value)
        else:
            result[key] = value
    return result


_PDE_ALIASES = {
    "poisson": "Poisson",
    "scalar": "Poisson",
    "genericscalar": "Poisson",
    "linear_elasticity": "LinearElasticity",
    "linear-elasticity": "LinearElasticity",
    "elastic": "LinearElasticity",
    "elasticity": "LinearElasticity",
    "generictensor": "LinearElasticity",
}

_MAT_ALIASES = {
    "e": "E",
    "young": "E",
    "youngs": "E",
    "youngs_modulus": "E",
    "young_modulus": "E",
    "nu": "nu",
    "poisson": "nu",
    "poisson_ratio": "nu",
}


def _has_value(entry: Dict[str, Any], key: str) -> bool:
    return key in entry and entry[key] is not None


def _validate_mode_choice(
    entry: Dict[str, Any],
    *,
    prefix: str,
    material_type: str,
    modes: List[tuple[str, tuple[str, ...]]],
) -> None:
    """Validate mutually-exclusive material parameterizations.

    The material classes are intentionally IDE-friendly: users can instantiate a
    blank object and fill fields incrementally. Completeness is enforced here
    during ``validate()`` / ``solve()``, not at construction time.
    """

    active_modes: List[str] = []
    partial_modes: List[tuple[str, List[str], List[str]]] = []

    for label, keys in modes:
        present = [key for key in keys if _has_value(entry, key)]
        if len(present) == len(keys):
            active_modes.append(label)
        elif present:
            missing = [key for key in keys if key not in present]
            partial_modes.append((label, present, missing))

    available = " or ".join(label for label, _ in modes)

    if len(active_modes) > 1 or (active_modes and partial_modes):
        raise ValueError(
            f"{prefix} ({material_type}) mixes incompatible parameterizations; "
            f"use {available}, not multiple modes at once"
        )

    if partial_modes:
        label, present, missing = partial_modes[0]
        raise ValueError(
            f"{prefix} ({material_type}) has an incomplete {label} parameterization: "
            f"present {present}, missing {missing}"
        )

    if not active_modes and entry.get("type") == material_type:
        raise ValueError(
            f"{prefix} ({material_type}) is missing a complete parameterization; "
            f"blank construction is allowed for IDE autocomplete, but solve()/validate() "
            f"require {available}"
        )

def _validate_positive_int(v):
    v = int(v)
    if v <= 0:
        raise ValueError("must be positive")
    return v

def _validate_int_or_none(v):
    if v is None:
        return None
    return int(v)

_EXTRAS_PROMOTION_RULES = {
    "max_iters": (
        _validate_positive_int,
        "extras['max_iters'] must be a positive integer, got {value!r} (type: {type_name})"
    ),
    "random_seed": (
        _validate_int_or_none,
        "extras['random_seed'] must be an integer or None, got {value!r} (type: {type_name})"
    ),
}

_LEGACY_MINIMAL_JSON_KEYS = frozenset({
    "pde",
    "discr_order",
    "materials",
    "boundary_conditions",
    "extras",
})

_FULL_JSON_HINT_KEYS = frozenset({
    "geometry",
    "solver",
    "time",
    "output",
    "contact",
    "initial_conditions",
    "constraints",
    "input",
    "problem_type",
    "problem_params",
    "selection",
    "space",
    "tests",
    "root_path",
    "common",
})


def _canon_pde(name: str) -> str:
    if not name:
        return "LinearElasticity"
    key = name.replace(" ", "_").lower()
    return _PDE_ALIASES.get(key, name)


def _canon_materials(mat: dict) -> dict:
    out = {}
    for k, v in (mat or {}).items():
        out[_MAT_ALIASES.get(k.lower(), k)] = v
    return out


def _jsonable_param(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(getattr(value, "to_dict")):
        try:
            return value.to_dict()
        except TypeError:
            return value
    return value


def _to_plain_value(value: Any) -> Any:
    """Recursively convert nested config objects into JSON-ready plain values."""
    if hasattr(value, "to_dict") and callable(getattr(value, "to_dict")):
        try:
            value = value.to_dict()
        except TypeError:
            return value

    if isinstance(value, dict):
        return {k: _to_plain_value(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_to_plain_value(v) for v in value]
    if isinstance(value, list):
        return [_to_plain_value(v) for v in value]
    return value


def _maybe_add(result: Dict[str, Any], key: str, value: Any) -> None:
    """Populate ``result[key]`` when ``value`` is meaningfully set."""
    if value is None:
        return
    result[key] = _to_plain_value(value)


# ============================================================================
# Units and Material Configuration Classes
# ============================================================================


@dataclass
class Quantity:
    """Unit-wrapped scalar value for a Python-first config style.

    Example:
        >>> Quantity.value(30.0, "MPa")
        >>> Quantity(30.0, "MPa")
    """

    amount: float
    unit: str

    @classmethod
    def value(cls, amount: float, unit: str) -> "Quantity":
        return cls(amount=amount, unit=unit)

    def to_dict(self) -> Dict[str, Any]:
        return {"value": self.amount, "unit": self.unit}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Quantity":
        return cls(amount=float(d["value"]), unit=str(d["unit"]))


def _with_optional_unit(value: Optional[float], unit: Optional[str]) -> Any:
    if value is None:
        return None
    if unit:
        return Quantity.value(float(value), unit)
    return value


@dataclass
class Material:
    """Material params. E, nu, rho. type defaults to LinearElasticity."""
    E: Optional[float] = None
    nu: Optional[float] = None
    rho: Optional[float] = None
    type: str = "LinearElasticity"
    
    def to_dict(self) -> Dict[str, Any]:
        result = {"type": self.type}
        if self.E is not None:
            result["E"] = _jsonable_param(self.E)
        if self.nu is not None:
            result["nu"] = _jsonable_param(self.nu)
        if self.rho is not None:
            result["rho"] = _jsonable_param(self.rho)
        return result
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Material":
        E = d.get("E") or d.get("e") or d.get("young") or d.get("youngs") or d.get("youngs_modulus") or d.get("young_modulus")
        nu = d.get("nu") or d.get("poisson") or d.get("poisson_ratio")
        return cls(
            E=E,
            nu=nu,
            rho=d.get("rho"),
            type=d.get("type", "LinearElasticity")
        )


_ParamType = Union[float, str, Any]
_IdType = Union[int, List[int]]


@dataclass
class NeoHookean:
    """NeoHookean material with IDE-friendly incremental construction.

    You can instantiate ``NeoHookean()`` and fill parameters later so IDE
    autocomplete shows the available knobs. Completeness is enforced by
    ``SimulationConfig.validate()`` / ``solve()``, not at construction time.
    """
    type: str = "NeoHookean"
    id: _IdType = 0
    rho: _ParamType = 1
    phi: _ParamType = 0
    psi: _ParamType = 0
    
    # E-nu mode
    E: Optional[_ParamType] = None
    nu: Optional[_ParamType] = None
    
    # lambda-mu mode
    lambda_: Optional[_ParamType] = None
    mu: Optional[_ParamType] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {"type": self.type}
        if self.E is not None:
            result["E"] = _jsonable_param(self.E)
        if self.nu is not None:
            result["nu"] = _jsonable_param(self.nu)
        if self.lambda_ is not None:
            result["lambda"] = _jsonable_param(self.lambda_)
        if self.mu is not None:
            result["mu"] = _jsonable_param(self.mu)
        if self.id != 0:
            result["id"] = self.id
        if self.rho != 1:
            result["rho"] = _jsonable_param(self.rho)
        if self.phi != 0:
            result["phi"] = _jsonable_param(self.phi)
        if self.psi != 0:
            result["psi"] = _jsonable_param(self.psi)
        return result

    @classmethod
    def young_poisson(
        cls,
        *,
        id: _IdType = 0,
        E: float,
        nu: float,
        rho: Optional[float] = None,
        E_unit: Optional[str] = None,
        rho_unit: Optional[str] = None,
        phi: _ParamType = 0,
        psi: _ParamType = 0,
    ) -> "NeoHookean":
        """Construct a NeoHookean material from the common ``(E, nu)`` inputs."""
        return cls(
            id=id,
            E=_with_optional_unit(E, E_unit),
            nu=nu,
            rho=_with_optional_unit(rho, rho_unit) if rho is not None else 1,
            phi=phi,
            psi=psi,
        )

    @classmethod
    def lame(
        cls,
        *,
        id: _IdType = 0,
        lambda_: float,
        mu: float,
        rho: Optional[float] = None,
        lambda_unit: Optional[str] = None,
        mu_unit: Optional[str] = None,
        rho_unit: Optional[str] = None,
        phi: _ParamType = 0,
        psi: _ParamType = 0,
    ) -> "NeoHookean":
        """Construct a NeoHookean material from Lamé parameters."""
        return cls(
            id=id,
            lambda_=_with_optional_unit(lambda_, lambda_unit),
            mu=_with_optional_unit(mu, mu_unit),
            rho=_with_optional_unit(rho, rho_unit) if rho is not None else 1,
            phi=phi,
            psi=psi,
        )


@dataclass
class IsochoricNeoHookean:
    """IsochoricNeoHookean with IDE-friendly incremental construction."""
    type: str = "IsochoricNeoHookean"
    id: _IdType = 0
    rho: _ParamType = 1
    phi: _ParamType = 0
    psi: _ParamType = 0
    
    # E-nu mode
    E: Optional[_ParamType] = None
    nu: Optional[_ParamType] = None
    
    # lambda-mu mode
    lambda_: Optional[_ParamType] = None
    mu: Optional[_ParamType] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {"type": self.type}
        if self.E is not None:
            result["E"] = _jsonable_param(self.E)
        if self.nu is not None:
            result["nu"] = _jsonable_param(self.nu)
        if self.lambda_ is not None:
            result["lambda"] = _jsonable_param(self.lambda_)
        if self.mu is not None:
            result["mu"] = _jsonable_param(self.mu)
        if self.id != 0:
            result["id"] = self.id
        if self.rho != 1:
            result["rho"] = _jsonable_param(self.rho)
        if self.phi != 0:
            result["phi"] = _jsonable_param(self.phi)
        if self.psi != 0:
            result["psi"] = _jsonable_param(self.psi)
        return result


@dataclass
class MooneyRivlin:
    """MooneyRivlin. c1, c2, k required."""
    c1: _ParamType = field()
    c2: _ParamType = field()
    k: _ParamType = field()
    type: str = "MooneyRivlin"
    id: _IdType = 0
    rho: _ParamType = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {
            "type": self.type,
            "c1": self.c1,
            "c2": self.c2,
            "k": self.k,
        }
        if self.id != 0:
            result["id"] = self.id
        if self.rho != 1:
            result["rho"] = self.rho
        return result


@dataclass
class MooneyRivlin3Param:
    """MooneyRivlin3Param. c1, c2, c3, d1 required."""
    c1: _ParamType = field()
    c2: _ParamType = field()
    c3: _ParamType = field()
    d1: _ParamType = field()
    type: str = "MooneyRivlin3Param"
    id: _IdType = 0
    rho: _ParamType = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {
            "type": self.type,
            "c1": self.c1,
            "c2": self.c2,
            "c3": self.c3,
            "d1": self.d1,
        }
        if self.id != 0:
            result["id"] = self.id
        if self.rho != 1:
            result["rho"] = self.rho
        return result


@dataclass
class MooneyRivlin3ParamSymbolic:
    """MooneyRivlin3ParamSymbolic. c1, c2, c3, d1 required."""
    c1: _ParamType = field()
    c2: _ParamType = field()
    c3: _ParamType = field()
    d1: _ParamType = field()
    type: str = "MooneyRivlin3ParamSymbolic"
    id: _IdType = 0
    rho: _ParamType = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {
            "type": self.type,
            "c1": self.c1,
            "c2": self.c2,
            "c3": self.c3,
            "d1": self.d1,
        }
        if self.id != 0:
            result["id"] = self.id
        if self.rho != 1:
            result["rho"] = self.rho
        return result


@dataclass
class UnconstrainedOgden:
    """UnconstrainedOgden material - provides IDE autocomplete support.
    
    Attributes:
        alphas: Alpha parameters (required).
        mus: Mu parameters list (required).
        Ds: D parameters list (required).
        id: Material ID or list of IDs. Defaults to 0.
        rho: Density. Defaults to 1.
    
    Example:
        >>> material = UnconstrainedOgden(alphas=2.0, mus=[1.0, 0.5], Ds=[0.1, 0.2])
    """
    alphas: _ParamType = field()
    mus: List[_ParamType] = field()
    Ds: List[_ParamType] = field()
    type: str = "UnconstrainedOgden"
    id: _IdType = 0
    rho: _ParamType = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {
            "type": self.type,
            "alphas": self.alphas,
            "mus": self.mus,
            "Ds": self.Ds,
        }
        if self.id != 0:
            result["id"] = self.id
        if self.rho != 1:
            result["rho"] = self.rho
        return result


@dataclass
class IncompressibleOgden:
    """IncompressibleOgden material - provides IDE autocomplete support.
    
    Attributes:
        c: C parameters (required, can be float, string, object, or list).
        m: M parameters (required, can be float, string, object, or list).
        k: Bulk modulus (required).
        id: Material ID or list of IDs. Defaults to 0.
        rho: Density. Defaults to 1.
    
    Example:
        >>> material = IncompressibleOgden(c=1.0, m=2.0, k=1000)
    """
    c: Union[_ParamType, List[_ParamType]] = field()
    m: Union[_ParamType, List[_ParamType]] = field()
    k: _ParamType = field()
    type: str = "IncompressibleOgden"
    id: _IdType = 0
    rho: _ParamType = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {
            "type": self.type,
            "c": self.c,
            "m": self.m,
            "k": self.k,
        }
        if self.id != 0:
            result["id"] = self.id
        if self.rho != 1:
            result["rho"] = self.rho
        return result


@dataclass
class LinearElasticity:
    """LinearElasticity material - provides IDE autocomplete support.
    
    Supports two input modes:
    1. E-nu input (Young's modulus and Poisson's ratio)
    2. lambda-mu input (Lamé parameters)
    
    Attributes:
        E: Young's modulus (required for E-nu mode).
        nu: Poisson's ratio (required for E-nu mode).
        lambda_: First Lamé parameter (required for lambda-mu mode).
        mu: Second Lamé parameter (shear modulus, required for lambda-mu mode).
        id: Material ID or list of IDs. Defaults to 0.
        rho: Density. Defaults to 1.
        phi: First angle (E-nu mode only). Defaults to 0.
        psi: Second angle (E-nu mode only). Defaults to 0.
    
    Construction is intentionally permissive so users can start with
    ``LinearElasticity()`` and fill fields gradually with IDE autocomplete.
    Parameter completeness is checked later during ``validate()`` / ``solve()``.
    """
    type: str = "LinearElasticity"
    id: _IdType = 0
    rho: _ParamType = 1
    
    # E-nu mode
    E: Optional[_ParamType] = None
    nu: Optional[_ParamType] = None
    phi: _ParamType = 0
    psi: _ParamType = 0
    
    # lambda-mu mode
    lambda_: Optional[_ParamType] = None
    mu: Optional[_ParamType] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {"type": self.type}
        if self.E is not None:
            result["E"] = _jsonable_param(self.E)
        if self.nu is not None:
            result["nu"] = _jsonable_param(self.nu)
        if self.lambda_ is not None:
            result["lambda"] = _jsonable_param(self.lambda_)
        if self.mu is not None:
            result["mu"] = _jsonable_param(self.mu)
        if self.id != 0:
            result["id"] = self.id
        if self.rho != 1:
            result["rho"] = _jsonable_param(self.rho)
        if self.phi != 0:
            result["phi"] = _jsonable_param(self.phi)
        if self.psi != 0:
            result["psi"] = _jsonable_param(self.psi)
        return result

    @classmethod
    def young_poisson(
        cls,
        *,
        id: _IdType = 0,
        E: float,
        nu: float,
        rho: Optional[float] = None,
        E_unit: Optional[str] = None,
        rho_unit: Optional[str] = None,
        phi: _ParamType = 0,
        psi: _ParamType = 0,
    ) -> "LinearElasticity":
        """Construct a LinearElasticity material from the common ``(E, nu)`` inputs."""
        return cls(
            id=id,
            E=_with_optional_unit(E, E_unit),
            nu=nu,
            rho=_with_optional_unit(rho, rho_unit) if rho is not None else 1,
            phi=phi,
            psi=psi,
        )

    @classmethod
    def lame(
        cls,
        *,
        id: _IdType = 0,
        lambda_: float,
        mu: float,
        rho: Optional[float] = None,
        lambda_unit: Optional[str] = None,
        mu_unit: Optional[str] = None,
        rho_unit: Optional[str] = None,
        phi: _ParamType = 0,
        psi: _ParamType = 0,
    ) -> "LinearElasticity":
        """Construct a LinearElasticity material from Lamé parameters."""
        return cls(
            id=id,
            lambda_=_with_optional_unit(lambda_, lambda_unit),
            mu=_with_optional_unit(mu, mu_unit),
            rho=_with_optional_unit(rho, rho_unit) if rho is not None else 1,
            phi=phi,
            psi=psi,
        )


@dataclass
class HookeLinearElasticity:
    """HookeLinearElasticity material - provides IDE autocomplete support.
    
    Supports two input modes:
    1. E-nu input (Young's modulus and Poisson's ratio)
    2. elasticity_tensor input (full elasticity tensor)
    
    Attributes:
        E: Young's modulus (required for E-nu mode).
        nu: Poisson's ratio (required for E-nu mode).
        elasticity_tensor: Full elasticity tensor (required for tensor mode).
        id: Material ID or list of IDs. Defaults to 0.
        rho: Density. Defaults to 1.
        fiber_direction: Fiber direction vector. Defaults to [0, 0, 0].
    
    Construction is intentionally permissive so users can start with
    ``HookeLinearElasticity()`` and fill fields incrementally.
    """
    type: str = "HookeLinearElasticity"
    id: _IdType = 0
    rho: _ParamType = 1
    fiber_direction: List[float] = field(default_factory=lambda: [0, 0, 0])
    
    # E-nu mode
    E: Optional[_ParamType] = None
    nu: Optional[_ParamType] = None
    
    # elasticity_tensor mode
    elasticity_tensor: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {"type": self.type}
        if self.E is not None:
            result["E"] = _jsonable_param(self.E)
        if self.nu is not None:
            result["nu"] = _jsonable_param(self.nu)
        if self.elasticity_tensor is not None:
            result["elasticity_tensor"] = self.elasticity_tensor
        if self.id != 0:
            result["id"] = self.id
        if self.rho != 1:
            result["rho"] = _jsonable_param(self.rho)
        if self.fiber_direction != [0, 0, 0]:
            result["fiber_direction"] = self.fiber_direction
        return result

    @classmethod
    def young_poisson(
        cls,
        *,
        id: _IdType = 0,
        E: float,
        nu: float,
        rho: Optional[float] = None,
        E_unit: Optional[str] = None,
        rho_unit: Optional[str] = None,
        fiber_direction: Optional[List[float]] = None,
    ) -> "HookeLinearElasticity":
        """Construct a HookeLinearElasticity material from the common ``(E, nu)`` inputs."""
        return cls(
            id=id,
            E=_with_optional_unit(E, E_unit),
            nu=nu,
            rho=_with_optional_unit(rho, rho_unit) if rho is not None else 1,
            fiber_direction=list(fiber_direction) if fiber_direction is not None else [0, 0, 0],
        )

    @classmethod
    def tensor(
        cls,
        *,
        elasticity_tensor: List[float],
        id: _IdType = 0,
        rho: Optional[float] = None,
        rho_unit: Optional[str] = None,
        fiber_direction: Optional[List[float]] = None,
    ) -> "HookeLinearElasticity":
        """Construct a HookeLinearElasticity material from a full elasticity tensor."""
        return cls(
            id=id,
            elasticity_tensor=list(elasticity_tensor),
            rho=_with_optional_unit(rho, rho_unit) if rho is not None else 1,
            fiber_direction=list(fiber_direction) if fiber_direction is not None else [0, 0, 0],
        )


@dataclass
class SaintVenant:
    """SaintVenant material - provides IDE autocomplete support.
    
    Supports two input modes:
    1. E-nu input (Young's modulus and Poisson's ratio)
    2. elasticity_tensor input (full elasticity tensor)
    
    Attributes:
        E: Young's modulus (required for E-nu mode).
        nu: Poisson's ratio (required for E-nu mode).
        elasticity_tensor: Full elasticity tensor (required for tensor mode).
        id: Material ID or list of IDs. Defaults to 0.
        rho: Density. Defaults to 1.
        phi: First angle. Defaults to 0.
        psi: Second angle. Defaults to 0.
        fiber_direction: Fiber direction vector. Defaults to [0, 0, 0].
    
    Construction is intentionally permissive so users can start with
    ``SaintVenant()`` and fill fields incrementally.
    """
    type: str = "SaintVenant"
    id: _IdType = 0
    rho: _ParamType = 1
    phi: _ParamType = 0
    psi: _ParamType = 0
    fiber_direction: List[float] = field(default_factory=lambda: [0, 0, 0])
    
    # E-nu mode
    E: Optional[_ParamType] = None
    nu: Optional[_ParamType] = None
    
    # elasticity_tensor mode
    elasticity_tensor: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {"type": self.type}
        if self.E is not None:
            result["E"] = _jsonable_param(self.E)
        if self.nu is not None:
            result["nu"] = _jsonable_param(self.nu)
        if self.elasticity_tensor is not None:
            result["elasticity_tensor"] = self.elasticity_tensor
        if self.id != 0:
            result["id"] = self.id
        if self.rho != 1:
            result["rho"] = _jsonable_param(self.rho)
        if self.phi != 0:
            result["phi"] = _jsonable_param(self.phi)
        if self.psi != 0:
            result["psi"] = _jsonable_param(self.psi)
        if self.fiber_direction != [0, 0, 0]:
            result["fiber_direction"] = self.fiber_direction
        return result

    @classmethod
    def young_poisson(
        cls,
        *,
        id: _IdType = 0,
        E: float,
        nu: float,
        rho: Optional[float] = None,
        E_unit: Optional[str] = None,
        rho_unit: Optional[str] = None,
        phi: _ParamType = 0,
        psi: _ParamType = 0,
        fiber_direction: Optional[List[float]] = None,
    ) -> "SaintVenant":
        """Construct a SaintVenant material from the common ``(E, nu)`` inputs."""
        return cls(
            id=id,
            E=_with_optional_unit(E, E_unit),
            nu=nu,
            rho=_with_optional_unit(rho, rho_unit) if rho is not None else 1,
            phi=phi,
            psi=psi,
            fiber_direction=list(fiber_direction) if fiber_direction is not None else [0, 0, 0],
        )

    @classmethod
    def tensor(
        cls,
        *,
        elasticity_tensor: List[float],
        id: _IdType = 0,
        rho: Optional[float] = None,
        rho_unit: Optional[str] = None,
        phi: _ParamType = 0,
        psi: _ParamType = 0,
        fiber_direction: Optional[List[float]] = None,
    ) -> "SaintVenant":
        """Construct a SaintVenant material from a full elasticity tensor."""
        return cls(
            id=id,
            elasticity_tensor=list(elasticity_tensor),
            rho=_with_optional_unit(rho, rho_unit) if rho is not None else 1,
            phi=phi,
            psi=psi,
            fiber_direction=list(fiber_direction) if fiber_direction is not None else [0, 0, 0],
        )


@dataclass
class Stokes:
    """Stokes material - provides IDE autocomplete support.
    
    Attributes:
        viscosity: Viscosity (required).
        id: Material ID or list of IDs. Defaults to 0.
        rho: Density. Defaults to 1.
    
    Example:
        >>> material = Stokes(viscosity=0.1)
    """
    viscosity: _ParamType = field()
    type: str = "Stokes"
    id: _IdType = 0
    rho: _ParamType = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {
            "type": self.type,
            "viscosity": self.viscosity,
        }
        if self.id != 0:
            result["id"] = self.id
        if self.rho != 1:
            result["rho"] = self.rho
        return result


@dataclass
class NavierStokes:
    """NavierStokes material - provides IDE autocomplete support.
    
    Attributes:
        viscosity: Viscosity (required).
        id: Material ID or list of IDs. Defaults to 0.
        rho: Density. Defaults to 1.
    
    Example:
        >>> material = NavierStokes(viscosity=0.1)
    """
    viscosity: _ParamType = field()
    type: str = "NavierStokes"
    id: _IdType = 0
    rho: _ParamType = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {
            "type": self.type,
            "viscosity": self.viscosity,
        }
        if self.id != 0:
            result["id"] = self.id
        if self.rho != 1:
            result["rho"] = self.rho
        return result


@dataclass
class OperatorSplitting:
    """OperatorSplitting material - provides IDE autocomplete support.
    
    Attributes:
        viscosity: Viscosity (required).
        id: Material ID or list of IDs. Defaults to 0.
        rho: Density. Defaults to 1.
    
    Example:
        >>> material = OperatorSplitting(viscosity=0.1)
    """
    viscosity: _ParamType = field()
    type: str = "OperatorSplitting"
    id: _IdType = 0
    rho: _ParamType = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {
            "type": self.type,
            "viscosity": self.viscosity,
        }
        if self.id != 0:
            result["id"] = self.id
        if self.rho != 1:
            result["rho"] = self.rho
        return result


@dataclass
class Electrostatics:
    """Electrostatics material - provides IDE autocomplete support.
    
    Attributes:
        epsilon: Permittivity (required).
        id: Material ID or list of IDs. Defaults to 0.
        rho: Density. Defaults to 1.
    
    Example:
        >>> material = Electrostatics(epsilon=8.85e-12)
    """
    epsilon: _ParamType = field()
    type: str = "Electrostatics"
    id: _IdType = 0
    rho: _ParamType = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {
            "type": self.type,
            "epsilon": self.epsilon,
        }
        if self.id != 0:
            result["id"] = self.id
        if self.rho != 1:
            result["rho"] = self.rho
        return result


@dataclass
class IncompressibleLinearElasticity:
    """IncompressibleLinearElasticity material - provides IDE autocomplete support.
    
    Attributes:
        E: Young's modulus (required).
        nu: Poisson's ratio (required).
        id: Material ID or list of IDs. Defaults to 0.
        rho: Density. Defaults to 1.
    
    Example:
        >>> material = IncompressibleLinearElasticity(E=2100, nu=0.3)
    """
    E: _ParamType = field()
    nu: _ParamType = field()
    type: str = "IncompressibleLinearElasticity"
    id: _IdType = 0
    rho: _ParamType = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {
            "type": self.type,
            "E": self.E,
            "nu": self.nu,
        }
        if self.id != 0:
            result["id"] = self.id
        if self.rho != 1:
            result["rho"] = self.rho
        return result


# ============================================================================
# Boundary and Initial Condition Classes
# ============================================================================


@dataclass
class DirichletBoundary:
    """Dirichlet boundary condition (fixed displacement).
    
    Attributes:
        id: Boundary ID (sideset ID).
        selection: Alternative to id (selection identifier).
        value: Displacement values (list of floats, e.g., [0.0, 0.0] for 2D).
    
    Example:
        >>> bc = DirichletBoundary(id=4, value=[0.0, 0.0])
        >>> # bc.id  # IDE will autocomplete
        >>> # bc.value  # IDE will autocomplete
    """
    id: Optional[int] = None
    selection: Optional[int] = None
    value: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {"value": self.value}
        if self.id is not None:
            result["id"] = self.id
        if self.selection is not None:
            result["selection"] = self.selection
        return result


@dataclass
class NeumannBoundary:
    """Neumann boundary condition (applied force/traction).
    
    Attributes:
        id: Boundary ID (sideset ID).
        selection: Alternative to id (selection identifier).
        value: Force/traction values (list of floats).
    
    Example:
        >>> bc = NeumannBoundary(id=2, value=[0.0, -1000.0])
    """
    id: Optional[int] = None
    selection: Optional[int] = None
    value: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {"value": self.value}
        if self.id is not None:
            result["id"] = self.id
        if self.selection is not None:
            result["selection"] = self.selection
        return result


@dataclass
class NormalAlignedNeumannBoundary(NeumannBoundary):
    """Neumann boundary aligned with the outward normal."""


@dataclass
class PressureBoundary(NeumannBoundary):
    """Pressure boundary condition entry."""


@dataclass
class PressureCavity(NeumannBoundary):
    """Pressure cavity boundary condition entry."""


@dataclass
class ObstacleDisplacement(DirichletBoundary):
    """Obstacle displacement entry."""


@dataclass
class PeriodicBoundary:
    """Periodic boundary-condition options."""

    enabled: bool = False
    tolerance: float = 1e-5
    correspondence: List[Any] = field(default_factory=list)
    fixed_macro_strain: List[float] = field(default_factory=list)
    linear_displacement_offset: List[float] = field(default_factory=list)
    force_zero_mean: bool = False

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        if self.enabled:
            result["enabled"] = True
        if self.tolerance != 1e-5:
            result["tolerance"] = self.tolerance
        if self.correspondence:
            result["correspondence"] = _to_plain_value(self.correspondence)
        if self.fixed_macro_strain:
            result["fixed_macro_strain"] = list(self.fixed_macro_strain)
        if self.linear_displacement_offset:
            result["linear_displacement_offset"] = list(self.linear_displacement_offset)
        if self.force_zero_mean:
            result["force_zero_mean"] = True
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PeriodicBoundary":
        return cls(
            enabled=bool(d.get("enabled", False)),
            tolerance=float(d.get("tolerance", 1e-5)),
            correspondence=list(d.get("correspondence", [])),
            fixed_macro_strain=list(d.get("fixed_macro_strain", [])),
            linear_displacement_offset=list(d.get("linear_displacement_offset", [])),
            force_zero_mean=bool(d.get("force_zero_mean", False)),
        )


@dataclass
class BoundaryConditions:
    """Boundary conditions container - provides IDE autocomplete support.
    
    This class allows users to set boundary conditions with IDE autocomplete,
    instead of using dictionaries where IDE cannot suggest available keys.
    
    Attributes:
        dirichlet_boundary: List of Dirichlet boundary conditions.
        neumann_boundary: List of Neumann boundary conditions.
        rhs: Body force (list of floats, e.g., [1.0, 0.0] for 2D).
        pressure: Pressure boundary conditions (optional).
    
    Example:
        >>> bc = BoundaryConditions()
        >>> bc.add_dirichlet(id=4, value=[0.0, 0.0])
        >>> bc.add_neumann(id=2, value=[0.0, -1000.0])
        >>> cfg = SimulationConfig(boundary_conditions=bc)
    """
    dirichlet_boundary: List[Union[DirichletBoundary, Dict[str, Any]]] = field(default_factory=list)
    neumann_boundary: List[Union[NeumannBoundary, Dict[str, Any]]] = field(default_factory=list)
    normal_aligned_neumann_boundary: List[Union[NormalAlignedNeumannBoundary, Dict[str, Any]]] = field(default_factory=list)
    pressure_boundary: List[Union[PressureBoundary, Dict[str, Any]]] = field(default_factory=list)
    pressure_cavity: List[Union[PressureCavity, Dict[str, Any]]] = field(default_factory=list)
    obstacle_displacements: List[Union[ObstacleDisplacement, Dict[str, Any]]] = field(default_factory=list)
    periodic_boundary: Optional[Union[PeriodicBoundary, Dict[str, Any]]] = None
    rhs: Optional[Any] = None
    pressure: Optional[List[Dict[str, Any]]] = None

    @classmethod
    def dirichlet_rhs(
        cls,
        *,
        id: Optional[int] = None,
        selection: Optional[int] = None,
        value: Optional[List[float]] = None,
        rhs: Optional[List[float]] = None,
    ) -> "BoundaryConditions":
        """Construct the common ``fixed boundary + body force`` pattern."""
        bc = cls()
        if id is not None or selection is not None or value is not None:
            bc.add_dirichlet(id=id, selection=selection, value=value or [])
        if rhs is not None:
            bc.set_rhs(rhs)
        return bc

    @classmethod
    def neumann_rhs(
        cls,
        *,
        id: Optional[int] = None,
        selection: Optional[int] = None,
        value: Optional[List[float]] = None,
        rhs: Optional[List[float]] = None,
    ) -> "BoundaryConditions":
        """Construct the common ``traction boundary + body force`` pattern."""
        bc = cls()
        if id is not None or selection is not None or value is not None:
            bc.add_neumann(id=id, selection=selection, value=value or [])
        if rhs is not None:
            bc.set_rhs(rhs)
        return bc

    @classmethod
    def periodic(
        cls,
        *,
        tolerance: float = 1e-5,
        correspondence: Optional[List[Any]] = None,
        fixed_macro_strain: Optional[List[float]] = None,
        linear_displacement_offset: Optional[List[float]] = None,
        force_zero_mean: bool = False,
    ) -> "BoundaryConditions":
        """Construct periodic boundary conditions."""
        return cls(
            periodic_boundary=PeriodicBoundary(
                enabled=True,
                tolerance=tolerance,
                correspondence=list(correspondence or []),
                fixed_macro_strain=list(fixed_macro_strain or []),
                linear_displacement_offset=list(linear_displacement_offset or []),
                force_zero_mean=force_zero_mean,
            )
        )
    
    def add_dirichlet(self, id: Optional[int] = None, selection: Optional[int] = None, 
                     value: List[float] = None) -> "BoundaryConditions":
        """Add a Dirichlet boundary condition.
        
        Args:
            id: Boundary ID (sideset ID).
            selection: Alternative to id (selection identifier).
            value: Displacement values.
            
        Returns:
            self for method chaining.
        """
        if value is None:
            value = []
        self.dirichlet_boundary.append(DirichletBoundary(id=id, selection=selection, value=value))
        return self
    
    def add_neumann(self, id: Optional[int] = None, selection: Optional[int] = None,
                   value: List[float] = None) -> "BoundaryConditions":
        """Add a Neumann boundary condition.
        
        Args:
            id: Boundary ID (sideset ID).
            selection: Alternative to id (selection identifier).
            value: Force/traction values.
            
        Returns:
            self for method chaining.
        """
        if value is None:
            value = []
        self.neumann_boundary.append(NeumannBoundary(id=id, selection=selection, value=value))
        return self

    def add_normal_aligned_neumann(
        self,
        id: Optional[int] = None,
        selection: Optional[int] = None,
        value: List[float] = None,
    ) -> "BoundaryConditions":
        if value is None:
            value = []
        self.normal_aligned_neumann_boundary.append(
            NormalAlignedNeumannBoundary(id=id, selection=selection, value=value)
        )
        return self

    def add_pressure_boundary(
        self,
        id: Optional[int] = None,
        selection: Optional[int] = None,
        value: List[float] = None,
    ) -> "BoundaryConditions":
        if value is None:
            value = []
        self.pressure_boundary.append(PressureBoundary(id=id, selection=selection, value=value))
        return self

    def add_pressure_cavity(
        self,
        id: Optional[int] = None,
        selection: Optional[int] = None,
        value: List[float] = None,
    ) -> "BoundaryConditions":
        if value is None:
            value = []
        self.pressure_cavity.append(PressureCavity(id=id, selection=selection, value=value))
        return self

    def add_obstacle_displacement(
        self,
        id: Optional[int] = None,
        selection: Optional[int] = None,
        value: List[float] = None,
    ) -> "BoundaryConditions":
        if value is None:
            value = []
        self.obstacle_displacements.append(
            ObstacleDisplacement(id=id, selection=selection, value=value)
        )
        return self
    
    def set_rhs(self, value: List[float]) -> "BoundaryConditions":
        """Set body force (right-hand side).
        
        Args:
            value: Body force values (e.g., [1.0, 0.0] for 2D).
            
        Returns:
            self for method chaining.
        """
        self.rhs = value
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format (for backend compatibility).
        
        Returns:
            Dictionary with boundary conditions.
        """
        result = {}
        
        if self.dirichlet_boundary:
            result["dirichlet_boundary"] = [
                bc.to_dict() if isinstance(bc, DirichletBoundary) else bc
                for bc in self.dirichlet_boundary
            ]
        
        if self.neumann_boundary:
            result["neumann_boundary"] = [
                bc.to_dict() if isinstance(bc, NeumannBoundary) else bc
                for bc in self.neumann_boundary
            ]
        
        if self.rhs is not None:
            result["rhs"] = _to_plain_value(self.rhs)

        if self.pressure is not None:
            result["pressure"] = self.pressure

        if self.normal_aligned_neumann_boundary:
            result["normal_aligned_neumann_boundary"] = [
                bc.to_dict() if hasattr(bc, "to_dict") else bc
                for bc in self.normal_aligned_neumann_boundary
            ]

        if self.pressure_boundary:
            result["pressure_boundary"] = [
                bc.to_dict() if hasattr(bc, "to_dict") else bc
                for bc in self.pressure_boundary
            ]

        if self.pressure_cavity:
            result["pressure_cavity"] = [
                bc.to_dict() if hasattr(bc, "to_dict") else bc
                for bc in self.pressure_cavity
            ]

        if self.obstacle_displacements:
            result["obstacle_displacements"] = [
                bc.to_dict() if hasattr(bc, "to_dict") else bc
                for bc in self.obstacle_displacements
            ]

        if self.periodic_boundary is not None:
            result["periodic_boundary"] = (
                self.periodic_boundary.to_dict()
                if isinstance(self.periodic_boundary, PeriodicBoundary)
                else dict(self.periodic_boundary)
            )

        return result
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BoundaryConditions":
        """Create BoundaryConditions from dictionary (backward compatibility).
        
        Args:
            d: Dictionary with boundary conditions.
            
        Returns:
            BoundaryConditions instance.
        """
        bc = cls()
        
        # Convert dirichlet_boundary
        if "dirichlet_boundary" in d:
            for item in d["dirichlet_boundary"]:
                if isinstance(item, dict):
                    bc.dirichlet_boundary.append(DirichletBoundary(
                        id=item.get("id"),
                        selection=item.get("selection"),
                        value=item.get("value", [])
                    ))
                else:
                    bc.dirichlet_boundary.append(item)
        
        # Convert neumann_boundary
        if "neumann_boundary" in d:
            for item in d["neumann_boundary"]:
                if isinstance(item, dict):
                    bc.neumann_boundary.append(NeumannBoundary(
                        id=item.get("id"),
                        selection=item.get("selection"),
                        value=item.get("value", [])
                    ))
                else:
                    bc.neumann_boundary.append(item)
        
        if "rhs" in d:
            bc.rhs = d["rhs"]

        if "pressure" in d:
            bc.pressure = d["pressure"]

        def _load_entries(name: str, ctor):
            for item in d.get(name, []):
                if isinstance(item, dict):
                    getattr(bc, name).append(
                        ctor(
                            id=item.get("id"),
                            selection=item.get("selection"),
                            value=item.get("value", []),
                        )
                    )
                else:
                    getattr(bc, name).append(item)

        _load_entries("normal_aligned_neumann_boundary", NormalAlignedNeumannBoundary)
        _load_entries("pressure_boundary", PressureBoundary)
        _load_entries("pressure_cavity", PressureCavity)
        _load_entries("obstacle_displacements", ObstacleDisplacement)

        if "periodic_boundary" in d and isinstance(d["periodic_boundary"], dict):
            bc.periodic_boundary = PeriodicBoundary.from_dict(d["periodic_boundary"])
        elif "periodic_boundary" in d:
            bc.periodic_boundary = d["periodic_boundary"]
        
        return bc


@dataclass
class InitialConditionEntry:
    """Initial-condition value assigned by a volume selection ID."""

    id: Optional[int] = None
    selection: Optional[int] = None
    value: Any = None

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"value": _to_plain_value(self.value)}
        if self.id is not None:
            result["id"] = self.id
        if self.selection is not None:
            result["selection"] = self.selection
        return result


@dataclass
class InitialConditions:
    """Initial conditions for solution, velocity, and acceleration."""

    solution: List[Union[InitialConditionEntry, Dict[str, Any]]] = field(default_factory=list)
    velocity: List[Union[InitialConditionEntry, Dict[str, Any]]] = field(default_factory=list)
    acceleration: List[Union[InitialConditionEntry, Dict[str, Any]]] = field(default_factory=list)

    @classmethod
    def velocity_only(
        cls,
        *,
        id: Optional[int] = None,
        selection: Optional[int] = None,
        value: Any = None,
    ) -> "InitialConditions":
        """Construct the common ``initial velocity only`` pattern."""
        return cls().add_velocity(id=id, selection=selection, value=value)

    @classmethod
    def solution_only(
        cls,
        *,
        id: Optional[int] = None,
        selection: Optional[int] = None,
        value: Any = None,
    ) -> "InitialConditions":
        """Construct the common ``initial solution only`` pattern."""
        return cls().add_solution(id=id, selection=selection, value=value)

    @classmethod
    def acceleration_only(
        cls,
        *,
        id: Optional[int] = None,
        selection: Optional[int] = None,
        value: Any = None,
    ) -> "InitialConditions":
        """Construct the common ``initial acceleration only`` pattern."""
        return cls().add_acceleration(id=id, selection=selection, value=value)

    def add_solution(self, *, id: Optional[int] = None, selection: Optional[int] = None, value: Any = None) -> "InitialConditions":
        self.solution.append(InitialConditionEntry(id=id, selection=selection, value=value))
        return self

    def add_velocity(self, *, id: Optional[int] = None, selection: Optional[int] = None, value: Any = None) -> "InitialConditions":
        self.velocity.append(InitialConditionEntry(id=id, selection=selection, value=value))
        return self

    def add_acceleration(self, *, id: Optional[int] = None, selection: Optional[int] = None, value: Any = None) -> "InitialConditions":
        self.acceleration.append(InitialConditionEntry(id=id, selection=selection, value=value))
        return self

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key in ("solution", "velocity", "acceleration"):
            entries = getattr(self, key)
            if entries:
                result[key] = [
                    item.to_dict() if hasattr(item, "to_dict") else _to_plain_value(item)
                    for item in entries
                ]
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "InitialConditions":
        out = cls()
        for key in ("solution", "velocity", "acceleration"):
            raw = d.get(key, [])
            target = getattr(out, key)
            for item in raw:
                if isinstance(item, dict):
                    target.append(
                        InitialConditionEntry(
                            id=item.get("id"),
                            selection=item.get("selection"),
                            value=item.get("value"),
                        )
                    )
                else:
                    target.append(item)
        return out


# ============================================================================
# Body, Constraint, Space, and Input Helper Classes
# ============================================================================


@dataclass
class SurfaceSelection:
    """Typed surface-selection descriptor for geometry/body operations."""

    id: Optional[int] = None
    axis: Optional[int] = None
    position: Optional[float] = None
    relative: bool = True
    center: Optional[List[float]] = None
    radius: Optional[float] = None
    box_bounds: Optional[List[List[float]]] = None
    normal: Optional[List[float]] = None
    offset: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        if self.id is not None:
            result["id"] = self.id
        if self.axis is not None:
            result["axis"] = self.axis
        if self.position is not None:
            result["position"] = self.position
        if self.relative is not True:
            result["relative"] = self.relative
        if self.center is not None:
            result["center"] = list(self.center)
        if self.radius is not None:
            result["radius"] = self.radius
        if self.box_bounds is not None:
            result["box"] = [list(v) for v in self.box_bounds]
        if self.normal is not None:
            result["normal"] = list(self.normal)
        if self.offset is not None:
            result["offset"] = self.offset
        return result

    @classmethod
    def position(
        cls,
        *,
        axis: int,
        position: float,
        id: Optional[int] = None,
        relative: bool = True,
    ) -> "SurfaceSelection":
        return cls(id=id, axis=axis, position=position, relative=relative)

    @classmethod
    def sphere(
        cls,
        *,
        center: List[float],
        radius: float,
        id: Optional[int] = None,
    ) -> "SurfaceSelection":
        return cls(id=id, center=list(center), radius=radius)

    @classmethod
    def box(
        cls,
        *,
        box_min: List[float],
        box_max: List[float],
        id: Optional[int] = None,
    ) -> "SurfaceSelection":
        return cls(id=id, box_bounds=[list(box_min), list(box_max)])

    @classmethod
    def plane(
        cls,
        *,
        normal: List[float],
        offset: float,
        id: Optional[int] = None,
    ) -> "SurfaceSelection":
        return cls(id=id, normal=list(normal), offset=offset)


@dataclass
class Body:
    """Python-side helper that keeps one body's IDs and operations aligned."""

    config: Any = field(repr=False)
    geometry: Any
    material: Any
    volume_id: int
    name: str = ""

    def _require_surface_capable_geometry(self) -> None:
        if not hasattr(self.geometry, "surface_selection"):
            raise TypeError(
                f"Body '{self.name or self.volume_id}' uses geometry type "
                f"{type(self.geometry).__name__}, which does not support surface selections"
            )

    def _surface_selection_list(self) -> List[Any]:
        self._require_surface_capable_geometry()
        current = getattr(self.geometry, "surface_selection", None)
        if current is None:
            current = []
            setattr(self.geometry, "surface_selection", current)
        elif not isinstance(current, list):
            current = [current]
            setattr(self.geometry, "surface_selection", current)
        return current

    def _attach_surface_selection(self, selection: Union[SurfaceSelection, Dict[str, Any]]) -> int:
        entries = self._surface_selection_list()
        if isinstance(selection, dict):
            selection_id = selection.get("id")
            if selection_id is None:
                selection_id = self.config._next_surface_selection_id()
                selection = dict(selection)
                selection["id"] = selection_id
            entries.append(selection)
            return int(selection_id)

        if selection.id is None:
            selection.id = self.config._next_surface_selection_id()
        entries.append(selection)
        return int(selection.id)

    def fix_surface(
        self,
        selection: Union[SurfaceSelection, Dict[str, Any]],
        *,
        value: List[float],
    ) -> "Body":
        selection_id = self._attach_surface_selection(selection)
        self.config._ensure_boundary_conditions_object().add_dirichlet(id=selection_id, value=list(value))
        return self

    def apply_neumann(
        self,
        selection: Union[SurfaceSelection, Dict[str, Any]],
        *,
        value: List[float],
    ) -> "Body":
        selection_id = self._attach_surface_selection(selection)
        self.config._ensure_boundary_conditions_object().add_neumann(id=selection_id, value=list(value))
        return self

    def set_initial_velocity(self, value: Any) -> "Body":
        self.config._ensure_initial_conditions_object().add_velocity(id=self.volume_id, value=_to_plain_value(value))
        return self

    def set_initial_solution(self, value: Any) -> "Body":
        self.config._ensure_initial_conditions_object().add_solution(id=self.volume_id, value=_to_plain_value(value))
        return self

    def set_initial_acceleration(self, value: Any) -> "Body":
        self.config._ensure_initial_conditions_object().add_acceleration(id=self.volume_id, value=_to_plain_value(value))
        return self


@dataclass
class SoftConstraint:
    """Soft constraint entry."""

    weight: float = 0.0
    data: str = ""

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        if self.weight != 0.0:
            result["weight"] = self.weight
        if self.data:
            result["data"] = self.data
        return result


@dataclass
class Constraints:
    """Hard/soft solver constraints."""

    hard: List[Any] = field(default_factory=list)
    soft: List[Union[SoftConstraint, Dict[str, Any]]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        if self.hard:
            result["hard"] = _to_plain_value(self.hard)
        if self.soft:
            result["soft"] = [
                item.to_dict() if hasattr(item, "to_dict") else _to_plain_value(item)
                for item in self.soft
            ]
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Constraints":
        soft = [
            SoftConstraint(weight=item.get("weight", 0.0), data=item.get("data", ""))
            if isinstance(item, dict)
            else item
            for item in d.get("soft", [])
        ]
        return cls(hard=list(d.get("hard", [])), soft=soft)


@dataclass
class Space:
    """Discretization-space configuration."""

    discr_order: Optional[Union[int, List[Dict[str, Any]]]] = None
    pressure_discr_order: Optional[int] = None
    use_p_ref: Optional[bool] = None
    polynomial_type: Optional[str] = None
    advanced: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        _maybe_add(result, "discr_order", self.discr_order)
        _maybe_add(result, "pressure_discr_order", self.pressure_discr_order)
        _maybe_add(result, "use_p_ref", self.use_p_ref)
        _maybe_add(result, "polynomial_type", self.polynomial_type)
        _maybe_add(result, "advanced", self.advanced)
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Space":
        return cls(
            discr_order=d.get("discr_order"),
            pressure_discr_order=d.get("pressure_discr_order"),
            use_p_ref=d.get("use_p_ref"),
            polynomial_type=d.get("polynomial_type"),
            advanced=d.get("advanced"),
        )


@dataclass
class Tests:
    """Schema-backed validation/test options."""

    margin: float = 1e-5
    time_steps: int = 1

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        if self.margin != 1e-5:
            result["margin"] = self.margin
        if self.time_steps != 1:
            result["time_steps"] = self.time_steps
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Tests":
        return cls(
            margin=float(d.get("margin", 1e-5)),
            time_steps=int(d.get("time_steps", 1)),
        )


@dataclass
class Input:
    """Input helper block for restart/state files."""

    data: Optional[str] = None
    state: Optional[str] = None
    directory: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        _maybe_add(result, "data", self.data)
        _maybe_add(result, "state", self.state)
        _maybe_add(result, "directory", self.directory)
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Input":
        return cls(
            data=d.get("data"),
            state=d.get("state"),
            directory=d.get("directory"),
        )


# ============================================================================
# Problem Parameters Classes (for better IDE support)
# ============================================================================

@dataclass
class GravityParams:
    """Parameters for Gravity problem - provides IDE autocomplete support.
    
    Attributes:
        force: Gravity force magnitude. Defaults to 0.1.
    
    Example:
        >>> params = GravityParams(force=0.1)
        >>> cfg = SimulationConfig(problem_type="Gravity", problem_params=params)
        >>> # params.force  # IDE will autocomplete
    """
    force: float = 0.1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format (for backend compatibility)."""
        return {"force": self.force}
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GravityParams":
        """Create GravityParams from dictionary (backward compatibility)."""
        return cls(force=d.get("force", 0.1))


@dataclass
class TorsionParams:
    """Parameters for Torsion problem - provides IDE autocomplete support.
    
    Attributes:
        axis_coordinate: Axis direction (1=x, 2=y, 3=z). Defaults to 2 (y-axis).
        n_turns: Number of turns. Defaults to 0.5.
        fixed_boundary: Sideset ID for fixed boundary. Defaults to 5.
        turning_boundary: Sideset ID for turning boundary. Defaults to 6.
    
    Example:
        >>> params = TorsionParams(axis_coordinate=2, n_turns=0.5)
        >>> cfg = SimulationConfig(problem_type="TorsionElastic", problem_params=params)
        >>> # params.axis_coordinate  # IDE will autocomplete
    """
    axis_coordinate: int = 2
    n_turns: float = 0.5
    fixed_boundary: int = 5
    turning_boundary: int = 6
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format (for backend compatibility).
        
        Note: Legacy API may use "axis_coordiante" (typo), but we use correct spelling.
        The backend should handle both spellings.
        """
        return {
            "axis_coordinate": self.axis_coordinate,
            "n_turns": self.n_turns,
            "fixed_boundary": self.fixed_boundary,
            "turning_boundary": self.turning_boundary,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TorsionParams":
        """Create TorsionParams from dictionary (backward compatibility).
        
        Handles both "axis_coordinate" and "axis_coordiante" (typo in legacy API).
        """
        # Handle both spellings
        axis_coord = d.get("axis_coordinate") or d.get("axis_coordiante", 2)
        return cls(
            axis_coordinate=axis_coord,
            n_turns=d.get("n_turns", 0.5),
            fixed_boundary=d.get("fixed_boundary", 5),
            turning_boundary=d.get("turning_boundary", 6),
        )


@dataclass
class FlowParams:
    """Parameters for Flow problem - provides IDE autocomplete support.
    
    Attributes:
        inflow: Sideset ID for inflow. Defaults to 1.
        outflow: Sideset ID for outflow. Defaults to 3.
        inflow_amount: Inflow amount. Defaults to 0.25.
        outflow_amount: Outflow amount. Defaults to 0.25.
        direction: Flow direction. Defaults to 0.
        obstacle: List of obstacle sideset IDs. Defaults to [7].
    
    Example:
        >>> params = FlowParams(inflow=1, outflow=3, inflow_amount=0.25)
        >>> cfg = SimulationConfig(problem_type="Flow", problem_params=params)
        >>> # params.inflow  # IDE will autocomplete
    """
    inflow: int = 1
    outflow: int = 3
    inflow_amount: float = 0.25
    outflow_amount: float = 0.25
    direction: int = 0
    obstacle: List[int] = field(default_factory=lambda: [7])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format (for backend compatibility).
        
        Note: Legacy API uses "inflow_amout" and "outflow_amout" (typos),
        but we use correct spelling. The to_dict() method handles backward compatibility.
        """
        return {
            "inflow": self.inflow,
            "outflow": self.outflow,
            "inflow_amout": self.inflow_amount,  # Legacy API typo
            "outflow_amout": self.outflow_amount,  # Legacy API typo
            "direction": self.direction,
            "obstacle": self.obstacle,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FlowParams":
        """Create FlowParams from dictionary (backward compatibility).
        
        Handles both "inflow_amount" (correct) and "inflow_amout" (legacy typo).
        """
        # Handle both spellings
        inflow_amt = d.get("inflow_amount") or d.get("inflow_amout", 0.25)
        outflow_amt = d.get("outflow_amount") or d.get("outflow_amout", 0.25)
        return cls(
            inflow=d.get("inflow", 1),
            outflow=d.get("outflow", 3),
            inflow_amount=inflow_amt,
            outflow_amount=outflow_amt,
            direction=d.get("direction", 0),
            obstacle=d.get("obstacle", [7]),
        )


@dataclass
class FlowWithObstacleParams:
    """Parameters for FlowWithObstacle problem - provides IDE autocomplete support.
    
    Attributes:
        U: Flow velocity. Defaults to 1.5.
        time_dependent: Whether the problem is time-dependent. Defaults to True.
    
    Example:
        >>> params = FlowWithObstacleParams(U=1.5, time_dependent=True)
        >>> cfg = SimulationConfig(problem_type="FlowWithObstacle", problem_params=params)
        >>> # params.U  # IDE will autocomplete
    """
    U: float = 1.5
    time_dependent: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format (for backend compatibility)."""
        return {
            "U": self.U,
            "time_dependent": self.time_dependent,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FlowWithObstacleParams":
        """Create FlowWithObstacleParams from dictionary (backward compatibility)."""
        return cls(
            U=d.get("U", 1.5),
            time_dependent=d.get("time_dependent", True),
        )


@dataclass
class Units:
    """Physical unit system for wrapped JSON values.

    Example:
        >>> units = Units(length="cm", mass="g", time="s")
    """

    length: str = "m"
    mass: str = "kg"
    time: str = "s"
    characteristic_length: Optional[_ParamType] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "length": self.length,
            "mass": self.mass,
            "time": self.time,
        }
        if self.characteristic_length is not None:
            result["characteristic_length"] = _jsonable_param(self.characteristic_length)
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Units":
        return cls(
            length=str(d.get("length", "m")),
            mass=str(d.get("mass", "kg")),
            time=str(d.get("time", "s")),
            characteristic_length=d.get("characteristic_length"),
        )

    @classmethod
    def set_units(
        cls,
        mapping: Optional[Dict[str, Any]] = None,
        *,
        length: str = "m",
        mass: str = "kg",
        time: str = "s",
        characteristic_length: Optional[_ParamType] = None,
    ) -> "Units":
        """Construct a unit system from explicit named fields.

        Examples:
            ``Units.set_units(length="cm", mass="g", time="s")``
            ``Units.set_units({"length": "cm", "mass": "g", "time": "s"})``
        """
        if mapping is not None:
            length = str(mapping.get("length", length))
            mass = str(mapping.get("mass", mass))
            time = str(mapping.get("time", time))
            characteristic_length = mapping.get("characteristic_length", characteristic_length)
        return cls(
            length=length,
            mass=mass,
            time=time,
            characteristic_length=characteristic_length,
        )

    @classmethod
    def si(cls, *, characteristic_length: Optional[_ParamType] = None) -> "Units":
        """Construct the default SI unit system."""
        return cls(length="m", mass="kg", time="s", characteristic_length=characteristic_length)

    @classmethod
    def cgs(cls, *, characteristic_length: Optional[_ParamType] = None) -> "Units":
        """Construct the centimeter-gram-second unit system."""
        return cls(length="cm", mass="g", time="s", characteristic_length=characteristic_length)


# Union type for all problem params
ProblemParams = Union[GravityParams, TorsionParams, FlowParams, FlowWithObstacleParams, Dict[str, Any]]


# ============================================================================
# Root SimulationConfig Contract
# ============================================================================


@dataclass
class SimulationConfig:
    """Human-friendly configuration → canonical form → PolyFEM Settings/Problem.

    This class stores intuitive configuration fields, provides normalization and
    lightweight validation, and (optionally) constructs a backend Settings/Problem.

    Attributes:
        pde: PDE name. Aliases auto-normalized to 'Poisson' or 'LinearElasticity'.
        discr_order: Polynomial order of the discretization (1, 2, ...).
        materials: Material parameters. Can be Material class, dict, or list of dicts.
                   Supports IDE autocomplete when using Material class.
        boundary_conditions: Boundary conditions. Can be BoundaryConditions class or dict.
                            Supports IDE autocomplete when using BoundaryConditions class.
        geometry: Geometry configuration. Can be Geometry class or dict/list.
                  Supports IDE autocomplete when using Geometry class.
        units: Unit system for wrapped physical quantities (optional).
        solver: Solver configuration. Can be Solver class or dict.
                Supports IDE autocomplete when using Solver class.
        time: Time configuration for transient problems. Can be Time class or dict.
              Supports IDE autocomplete when using Time class.
        output: Output configuration. Can be Output class or dict.
                Supports IDE autocomplete when using Output class.
        contact: Contact configuration. Can be Contact class or dict.
                Supports IDE autocomplete when using Contact class.
        extras: Advanced options; passed through if the backend exposes such hooks.
        selection: Optional Selection object for geometric boundary selection.
        problem_type: Optional predefined problem type (e.g., 'Gravity', 'Franke', 'Torsion').
        problem_params: Optional parameters for predefined problems. Can be ProblemParams class
                       (GravityParams, TorsionParams, FlowParams, FlowWithObstacleParams) or dict.
                       Supports IDE autocomplete when using classes.

    Example:
        >>> # Using classes (recommended - IDE autocomplete)
        >>> from polyfempy.api.config import Material, BoundaryConditions, Geometry, Solver, Time, Output, Contact
        >>> material = Material(E=2100, nu=0.3)
        >>> bc = BoundaryConditions()
        >>> bc.add_dirichlet(id=4, value=[0.0, 0.0])
        >>> geom = Geometry(meshes=["mesh.obj"])
        >>> solver = Solver(linear=LinearSolver(solver_type="Eigen::SparseLU"))
        >>> output = Output(directory="results")
        >>> cfg = SimulationConfig(materials=material, boundary_conditions=bc, geometry=geom, solver=solver, output=output)
        >>> 
        >>> # Using dict (backward compatible)
        >>> cfg = SimulationConfig(materials={"E": 2100, "nu": 0.3})
        >>> 
        >>> # Using convenience methods
        >>> cfg = SimulationConfig()
        >>> cfg.set_material(E=2100, nu=0.3)
        >>> cfg.set_dirichlet_boundary(id=4, value=[0.0, 0.0])
        >>> 
        >>> # Using ProblemParams classes
        >>> from polyfempy.api.config import GravityParams
        >>> params = GravityParams(force=0.1)
        >>> cfg = SimulationConfig(problem_type="Gravity", problem_params=params)
    """

    pde: str = "LinearElasticity"
    discr_order: int = 1
    materials: Union[
        Material, NeoHookean, IsochoricNeoHookean, MooneyRivlin, MooneyRivlin3Param,
        MooneyRivlin3ParamSymbolic, UnconstrainedOgden, IncompressibleOgden,
        LinearElasticity, HookeLinearElasticity, SaintVenant, Stokes, NavierStokes,
        OperatorSplitting, Electrostatics, IncompressibleLinearElasticity,
        Dict[str, Any], List[Dict[str, Any]], dict
    ] = field(default_factory=dict)
    boundary_conditions: Union[BoundaryConditions, Dict[str, Any], dict] = field(default_factory=dict)
    geometry: Optional[Union["Geometry", List[Dict[str, Any]], Dict[str, Any]]] = None
    units: Optional[Union["Units", Dict[str, Any]]] = None
    solver: Optional[Union["Solver", Dict[str, Any]]] = None
    time: Optional[Union["Time", Dict[str, Any]]] = None
    output: Optional[Union["Output", Dict[str, Any]]] = None
    contact: Optional[Union["Contact", Dict[str, Any]]] = None
    initial_conditions: Optional[Union["InitialConditions", Dict[str, Any]]] = None
    constraints: Optional[Union["Constraints", Dict[str, Any]]] = None
    space: Optional[Union["Space", Dict[str, Any]]] = None
    tests: Optional[Union["Tests", Dict[str, Any]]] = None
    input: Optional[Union["Input", Dict[str, Any]]] = None
    extras: dict = field(default_factory=dict)
    selection: Optional["Selection"] = None
    problem_type: Optional[str] = None
    problem_params: ProblemParams = field(default_factory=dict)

    # ---------------- Canonicalization ----------------

    def _get_materials_dict(self) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Internal helper: convert materials to dict format."""
        # Check if it's a material class instance (has to_dict method)
        if hasattr(self.materials, 'to_dict') and callable(getattr(self.materials, 'to_dict')):
            return self.materials.to_dict()
        elif isinstance(self.materials, list) and len(self.materials) > 0:
            mats = []
            for material in self.materials:
                if hasattr(material, 'to_dict') and callable(getattr(material, 'to_dict')):
                    mats.append(material.to_dict())
                elif isinstance(material, dict):
                    mats.append(_canon_materials(material))
            return mats
        elif isinstance(self.materials, dict):
            return _canon_materials(self.materials)
        else:
            return {}
    
    def _get_boundary_conditions_dict(self) -> Dict[str, Any]:
        """Internal helper: convert boundary_conditions to dict format."""
        if isinstance(self.boundary_conditions, BoundaryConditions):
            return self.boundary_conditions.to_dict()
        elif isinstance(self.boundary_conditions, dict):
            return dict(self.boundary_conditions)
        else:
            return {}
    
    def _get_problem_params_dict(self) -> Dict[str, Any]:
        """Internal helper: convert problem_params to dict format."""
        if isinstance(self.problem_params, (GravityParams, TorsionParams, FlowParams, FlowWithObstacleParams)):
            return self.problem_params.to_dict()
        elif isinstance(self.problem_params, dict):
            return dict(self.problem_params)
        else:
            return {}
    
    def _get_geometry_dict(self) -> Optional[Union[List[Dict[str, Any]], Dict[str, Any]]]:
        """Internal helper: convert geometry to dict format."""
        if self.geometry is None:
            return None
        if isinstance(self.geometry, Geometry):
            return self.geometry.to_dict()
        elif isinstance(self.geometry, (list, dict)):
            return self.geometry
        else:
            return None

    def _get_units_dict(self) -> Optional[Dict[str, Any]]:
        """Internal helper: convert units to dict format."""
        if self.units is None:
            return None
        if isinstance(self.units, Units):
            return self.units.to_dict()
        if isinstance(self.units, dict):
            return dict(self.units)
        return None
    
    def _get_solver_dict(self) -> Optional[Dict[str, Any]]:
        """Internal helper: convert solver to dict format."""
        if self.solver is None:
            return None
        if isinstance(self.solver, Solver):
            return self.solver.to_dict()
        elif isinstance(self.solver, dict):
            return dict(self.solver)
        else:
            return None
    
    def _get_time_dict(self) -> Optional[Dict[str, Any]]:
        """Internal helper: convert time to dict format."""
        if self.time is None:
            return None
        if isinstance(self.time, Time):
            return self.time.to_dict()
        elif isinstance(self.time, dict):
            return dict(self.time)
        else:
            return None
    
    def _get_output_dict(self) -> Optional[Dict[str, Any]]:
        """Internal helper: convert output to dict format."""
        if self.output is None:
            return None
        if isinstance(self.output, Output):
            return self.output.to_dict()
        elif isinstance(self.output, dict):
            return dict(self.output)
        else:
            return None
    
    def _get_contact_dict(self) -> Optional[Dict[str, Any]]:
        """Internal helper: convert contact to dict format."""
        if self.contact is None:
            return None
        if isinstance(self.contact, Contact):
            return self.contact.to_dict()
        elif isinstance(self.contact, dict):
            return dict(self.contact)
        else:
            return None

    def _get_initial_conditions_dict(self) -> Optional[Dict[str, Any]]:
        if self.initial_conditions is None:
            return None
        if isinstance(self.initial_conditions, InitialConditions):
            return self.initial_conditions.to_dict()
        if isinstance(self.initial_conditions, dict):
            return dict(self.initial_conditions)
        return None

    def _get_constraints_dict(self) -> Optional[Dict[str, Any]]:
        if self.constraints is None:
            return None
        if isinstance(self.constraints, Constraints):
            return self.constraints.to_dict()
        if isinstance(self.constraints, dict):
            return dict(self.constraints)
        return None

    def _get_space_dict(self) -> Optional[Dict[str, Any]]:
        if self.space is None:
            return None
        if isinstance(self.space, Space):
            return self.space.to_dict()
        if isinstance(self.space, dict):
            return dict(self.space)
        return None

    def _get_tests_dict(self) -> Optional[Dict[str, Any]]:
        if self.tests is None:
            return None
        if isinstance(self.tests, Tests):
            return self.tests.to_dict()
        if isinstance(self.tests, dict):
            return dict(self.tests)
        return None

    def _get_input_dict(self) -> Optional[Dict[str, Any]]:
        if self.input is None:
            return None
        if isinstance(self.input, Input):
            return self.input.to_dict()
        if isinstance(self.input, dict):
            return dict(self.input)
        return None

    def canonicalized(self) -> "SimulationConfig":
        """Return a shallow copy with normalized (canonical) fields.

        PDE aliases are reduced to a canonical name; material keys are normalized
        to 'E'/'nu'. Containers are shallow-copied to avoid external mutation.

        Returns:
            A new `SimulationConfig` instance in canonical form.

        Notes:
            - This method does not perform deep validation of ranges/physics.
            - Use `validate()` for sanity checks before constructing Settings.
        """
        return SimulationConfig(
            pde=_canon_pde(self.pde),
            discr_order=int(self.discr_order),
            materials=self._get_materials_dict(),
            boundary_conditions=self._get_boundary_conditions_dict(),
            geometry=self.geometry,  # Preserve geometry (not canonicalized)
            units=self.units,  # Preserve units
            solver=self.solver,  # Preserve solver
            time=self.time,  # Preserve time
            output=self.output,  # Preserve output
            contact=self.contact,  # Preserve contact
            initial_conditions=self.initial_conditions,
            constraints=self.constraints,
            space=self.space,
            tests=self.tests,
            input=self.input,
            extras=dict(self.extras or {}),
            selection=self.selection,  # Selection objects are not copied
            problem_type=self.problem_type,
            problem_params=self._get_problem_params_dict(),
        )

    # ---------------- Dictionary conversion ----------------
    
    def to_dict(self) -> dict:
        """Convert SimulationConfig to a dictionary representation.
        
        This method provides a stable interface for converting the configuration
        to a plain dictionary, which can be used by backends or serialized.
        
        Returns:
            Dictionary containing all configuration fields. If full JSON config
            is stored in extras, it returns that; otherwise constructs from fields.
            
        Notes:
            - This is the canonical way to get a dict representation
            - Future nanobind implementations should use this interface
            - Structure is stable and documented
        """
        # Start from stored JSON when available so we preserve JSON-only details
        # such as units/problem blocks, but still honor Python-side edits to the
        # dataclass fields after `from_json_file(...)`.
        result = {}
        if self.extras and "_full_json_config" in self.extras:
            try:
                result = copy.deepcopy(self.extras["_full_json_config"])
            except Exception:
                result = dict(self.extras["_full_json_config"])

        # Construct/overlay from current fields.
        c = self.canonicalized()
        # Handle materials: if it's a material class, use to_dict(); otherwise use dict
        materials_dict = c._get_materials_dict() if hasattr(c, '_get_materials_dict') else (
            c.materials.to_dict() if hasattr(c.materials, 'to_dict') else (
                c.materials if isinstance(c.materials, dict) else dict(c.materials)
            )
        )
        # Ensure materials is in array format for JSON mode (C++ backend expects array)
        if isinstance(materials_dict, dict) and not isinstance(materials_dict, list):
            materials_dict = [materials_dict]
        elif not isinstance(materials_dict, list):
            # If materials_dict is not a dict or list, wrap it
            materials_dict = [materials_dict] if materials_dict else []
        
        result["pde"] = c.pde
        result["discr_order"] = c.discr_order
        result["materials"] = materials_dict
        result["boundary_conditions"] = (
            c.boundary_conditions if isinstance(c.boundary_conditions, dict) else dict(c.boundary_conditions)
        )
        
        # Add geometry if provided
        geometry_dict = c._get_geometry_dict() if hasattr(c, '_get_geometry_dict') else None
        if geometry_dict is not None:
            result["geometry"] = geometry_dict

        units_dict = c._get_units_dict() if hasattr(c, '_get_units_dict') else None
        if units_dict is not None:
            result["units"] = units_dict
        
        # Add solver if provided
        solver_dict = c._get_solver_dict() if hasattr(c, '_get_solver_dict') else None
        if solver_dict is not None:
            result["solver"] = solver_dict
        
        # Add time if provided
        time_dict = c._get_time_dict() if hasattr(c, '_get_time_dict') else None
        if time_dict is not None:
            result["time"] = time_dict
        
        # Add output if provided
        output_dict = c._get_output_dict() if hasattr(c, '_get_output_dict') else None
        if output_dict is not None:
            result["output"] = output_dict
        
        # Add contact if provided
        contact_dict = c._get_contact_dict() if hasattr(c, '_get_contact_dict') else None
        if contact_dict is not None:
            result["contact"] = contact_dict

        initial_conditions_dict = c._get_initial_conditions_dict() if hasattr(c, '_get_initial_conditions_dict') else None
        if initial_conditions_dict is not None:
            result["initial_conditions"] = initial_conditions_dict

        constraints_dict = c._get_constraints_dict() if hasattr(c, '_get_constraints_dict') else None
        if constraints_dict is not None:
            result["constraints"] = constraints_dict

        space_dict = c._get_space_dict() if hasattr(c, '_get_space_dict') else None
        if space_dict is not None:
            result["space"] = space_dict

        tests_dict = c._get_tests_dict() if hasattr(c, '_get_tests_dict') else None
        if tests_dict is not None:
            result["tests"] = tests_dict

        input_dict = c._get_input_dict() if hasattr(c, '_get_input_dict') else None
        if input_dict is not None:
            result["input"] = input_dict
        
        # Extract common solver parameters from extras to top level for backend compatibility
        if c.extras:
            public_extras = {
                k: v for k, v in c.extras.items()
                if not str(k).startswith("_")
            }

            # Copy extras but also promote common keys to top level
            if public_extras:
                result["extras"] = dict(public_extras)
            
            # Promote parameters according to _EXTRAS_PROMOTION_RULES
            for param_name, (validator, error_template) in _EXTRAS_PROMOTION_RULES.items():
                if param_name in public_extras:
                    value = public_extras[param_name]
                    try:
                        converted_value = validator(value)
                        result[param_name] = converted_value
                    except (ValueError, TypeError) as e:
                        raise ValueError(
                            error_template.format(
                                value=value,
                                type_name=type(value).__name__
                            )
                        ) from e
        
        if c.problem_type:
            result["problem_type"] = c.problem_type
        problem_params_dict = c._get_problem_params_dict() if hasattr(c, '_get_problem_params_dict') else (dict(c.problem_params) if c.problem_params else {})
        if problem_params_dict:
            result["problem_params"] = problem_params_dict
        if c.selection is not None:
            result["selection"] = c.selection.to_dict()
            
        return result

    # ---------------- JSON I/O ----------------

    def to_full_json_dict(self) -> dict:
        """Return the full current configuration as a JSON-ready dictionary.

        This is the explicit, user-facing name for the complete configuration
        snapshot represented by `to_dict()`. It is useful when the caller wants
        a full configuration object that can be:

        - serialized with `json.dump(...)`
        - modified in Python as a plain dict
        - round-tripped back through `SimulationConfig.from_full_json_dict(...)`

        Returns:
            A full JSON-ready configuration dictionary.

        Example:
            >>> cfg = SimulationConfig.from_json_file("config.json")
            >>> d = cfg.to_full_json_dict()
            >>> cfg2 = SimulationConfig.from_full_json_dict(d)
        """
        return self.to_dict()

    def to_full_json_str(self) -> str:
        """Serialize the full current configuration to JSON.

        This is the recommended JSON export method when the caller expects a
        round-trippable representation of the current `SimulationConfig`,
        including geometry, solver, time, output, contact, and any JSON-derived
        fields preserved by `to_dict()`.

        Returns:
            A JSON string representing the full current configuration.

        Example:
            >>> cfg = SimulationConfig.from_json_file("config.json")
            >>> s = cfg.to_full_json_str()
            >>> cfg2 = SimulationConfig.from_full_json_str(s)
        """
        return json.dumps(self.to_full_json_dict(), separators=(",", ":"))

    def to_minimal_json_dict(self) -> dict:
        """Serialize the legacy minimal configuration shape to a JSON-ready dict.

        This export intentionally keeps only the small historical subset used by
        the old `to_json_str()` helper: PDE, discretization order, materials,
        boundary conditions, and public `extras`.

        It is suitable only for the matching `from_minimal_json_dict()` /
        `from_minimal_json_str()` compatibility path. For a complete snapshot of
        the current configuration, use `to_full_json_dict()` instead.
        """
        c = self.canonicalized()
        obj = {
            "pde": c.pde,
            "discr_order": c.discr_order,
            "materials": c.materials if isinstance(c.materials, dict) else c.materials,
            "boundary_conditions": (
                c.boundary_conditions if isinstance(c.boundary_conditions, dict) else c.boundary_conditions
            ),
        }
        public_extras = {
            k: v for k, v in (c.extras or {}).items()
            if not str(k).startswith("_")
        }
        if public_extras:
            obj["extras"] = public_extras
        return obj

    def to_minimal_json_str(self) -> str:
        """Serialize the legacy minimal configuration shape to JSON.

        This is the explicit name for the old minimal export path. The paired
        import method is `from_minimal_json_str()`.
        """
        return json.dumps(self.to_minimal_json_dict(), separators=(",", ":"))

    def to_json_str(self) -> str:
        """Serialize a minimal canonical configuration to a compact JSON string.

        This legacy helper only includes a small subset of the configuration:
        core PDE/material/boundary-condition fields plus `extras`.
        It does **not** represent the full current configuration and should not
        be used when the caller expects a complete round-trip of geometry, time,
        solver, output, or contact settings.

        Prefer `to_full_json_str()` for full configuration export.

        Returns:
            A compact JSON string representing a minimal canonical configuration.

        Example:
            >>> cfg = SimulationConfig.linear_elasticity(2100, 0.3)
            >>> cfg.to_json_str()
            '{"pde":"LinearElasticity","discr_order":1,"materials":{"E":2100,"nu":0.3},"boundary_conditions":{}}'
        """
        warnings.warn(
            "SimulationConfig.to_json_str() is a deprecated alias for the legacy minimal export; "
            "use to_minimal_json_str() for the same subset or to_full_json_str() for a round-trippable full export.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.to_minimal_json_str()

    @classmethod
    def _load_json_object(cls, s: str, *, source: str) -> Dict[str, Any]:
        d = json.loads(s)
        if not isinstance(d, dict):
            raise TypeError(f"{source} expects a JSON object at the top level, got {type(d).__name__}")
        return d

    @classmethod
    def _looks_like_legacy_minimal_json_dict(cls, d: Dict[str, Any]) -> bool:
        keys = set(d.keys())
        if not keys:
            return True
        if keys & _FULL_JSON_HINT_KEYS:
            return False
        return keys <= _LEGACY_MINIMAL_JSON_KEYS

    @classmethod
    def from_full_json_dict(cls, d: dict) -> "SimulationConfig":
        """Explicit alias for `from_json_dict()` when the input is full JSON."""
        return cls.from_json_dict(d)

    @classmethod
    def from_full_json_str(cls, s: str) -> "SimulationConfig":
        """Deserialize a full PolyFEM JSON string.

        This is the explicit inverse of `to_full_json_str()`.
        """
        return cls.from_json_dict(cls._load_json_object(s, source="SimulationConfig.from_full_json_str()"))

    @classmethod
    def from_minimal_json_dict(cls, d: dict) -> "SimulationConfig":
        """Deserialize the legacy minimal JSON dictionary shape.

        This only accepts the historical subset produced by
        `to_minimal_json_dict()` / `to_minimal_json_str()`. If the dictionary
        contains full-configuration keys such as `geometry` or `time`, use
        `from_full_json_dict()` instead.
        """
        if not isinstance(d, dict):
            raise TypeError(f"SimulationConfig.from_minimal_json_dict() expects a dict, got {type(d).__name__}")

        full_only = sorted(set(d.keys()) & _FULL_JSON_HINT_KEYS)
        if full_only:
            raise ValueError(
                "SimulationConfig.from_minimal_json_dict() only accepts the legacy minimal schema; "
                f"found full-configuration keys: {', '.join(full_only)}. "
                "Use from_full_json_dict() instead."
            )

        materials_raw = d.get("materials", {})
        if isinstance(materials_raw, list):
            materials = [
                _canon_materials(item) if isinstance(item, dict) else copy.deepcopy(item)
                for item in materials_raw
            ]
        elif isinstance(materials_raw, dict):
            materials = _canon_materials(materials_raw)
        else:
            materials = copy.deepcopy(materials_raw)

        bc_raw = d.get("boundary_conditions", {})
        if isinstance(bc_raw, dict):
            boundary_conditions = BoundaryConditions.from_dict(bc_raw)
        else:
            boundary_conditions = copy.deepcopy(bc_raw)

        extras_raw = d.get("extras", {})
        if extras_raw is None:
            extras = {}
        elif isinstance(extras_raw, dict):
            extras = dict(extras_raw)
        else:
            raise TypeError(
                "SimulationConfig.from_minimal_json_dict() expects 'extras' to be a dict when present, "
                f"got {type(extras_raw).__name__}"
            )

        discr_order = d.get("discr_order", 1)
        if discr_order is None:
            discr_order = 1

        return cls(
            pde=d.get("pde", "LinearElasticity"),
            discr_order=int(discr_order),
            materials=materials,
            boundary_conditions=boundary_conditions,
            extras=extras,
        )

    @classmethod
    def from_minimal_json_str(cls, s: str) -> "SimulationConfig":
        """Deserialize the legacy minimal JSON string produced by `to_minimal_json_str()`."""
        return cls.from_minimal_json_dict(
            cls._load_json_object(s, source="SimulationConfig.from_minimal_json_str()")
        )

    @classmethod
    def from_json_str(cls, s: str, *, kind: str = "auto") -> "SimulationConfig":
        """Deserialize a configuration from a JSON string.

        This compatibility helper accepts both:

        - the legacy minimal JSON produced by `to_json_str()`
        - the full configuration JSON produced by `to_full_json_str()`

        Args:
            s: JSON string representing either a minimal or full configuration.
            kind: One of `"auto"`, `"full"`, or `"minimal"`. `"auto"` keeps
                backward compatibility by inspecting the JSON shape, while the
                explicit modes provide stable semantics.

        Returns:
            A `SimulationConfig` instance reconstructed from the JSON.

        Notes:
            - Prefer `from_full_json_str()` for round-trippable full config import.
            - Prefer `from_minimal_json_str()` only when reading the legacy subset.
            - `"auto"` mode is kept for backward compatibility and warns because
              it cannot perfectly infer user intent from every JSON shape.
        """
        d = cls._load_json_object(s, source="SimulationConfig.from_json_str()")

        if kind == "full":
            return cls.from_json_dict(d)
        if kind == "minimal":
            return cls.from_minimal_json_dict(d)
        if kind != "auto":
            raise ValueError(f"kind must be 'auto', 'full', or 'minimal', got {kind!r}")

        warnings.warn(
            "SimulationConfig.from_json_str() is a compatibility helper with auto-detected semantics; "
            "use from_full_json_str() for full configs or from_minimal_json_str() for the legacy minimal subset.",
            DeprecationWarning,
            stacklevel=2,
        )
        if cls._looks_like_legacy_minimal_json_dict(d):
            return cls.from_minimal_json_dict(d)
        return cls.from_json_dict(d)
    
    @classmethod
    def from_json_dict(cls, d: dict) -> "SimulationConfig":
        """Create a SimulationConfig from a JSON dictionary (full PolyFEM JSON format).
        
        This method supports the complete PolyFEM JSON format including:
        - geometry (mesh files, transformations, selections, volume_selection, surface_selection)
        - materials (array format with type, E, nu, rho, id, etc.)
        - time (transient problems: t0, tend, dt, time_steps, integrator)
        - output (Paraview, directory, json, etc.)
        - boundary_conditions (all types: dirichlet, neumann, pressure, rhs)
        - solver (linear, nonlinear, max_iterations, tolerance, etc.)
        - contact (enabled, dhat, mu, epsv, etc.)
        - space (discr_order, etc.)
        - tests (validation settings)
        
        All fields are preserved in the full JSON config for direct use by the solver.
        
        Args:
            d: Dictionary containing PolyFEM JSON configuration (from polyfem-data examples).
            
        Returns:
            A `SimulationConfig` instance. Full JSON is stored in extras["_full_json_config"]
            for direct use by solve(). Known fields are also extracted for convenience.
            
        Example:
            >>> with open("config.json") as f:
            ...     config_dict = json.load(f)
            >>> cfg = SimulationConfig.from_json_dict(config_dict)
            >>> # Full config available via cfg.to_dict() or cfg.extras["_full_json_config"]
        """
        # Make a deep copy to avoid modifying input
        import copy
        full_config = copy.deepcopy(d)
        
        # Extract known fields for convenience
        pde = full_config.get("pde", "LinearElasticity")
        
        # Try to infer PDE from materials if not specified
        if pde == "LinearElasticity" and "materials" in full_config:
            mats = full_config["materials"]
            if isinstance(mats, list) and len(mats) > 0:
                mat_type = mats[0].get("type", "")
                if "NeoHookean" in mat_type or "SaintVenant" in mat_type:
                    pde = "NonLinearElasticity"
        
        # Extract discr_order from various possible locations
        discr_order = 1
        if "discr_order" in full_config:
            discr_order_val = full_config["discr_order"]
            # Handle list format: [{"id": 2, "order": 2}]
            if isinstance(discr_order_val, list) and len(discr_order_val) > 0:
                if isinstance(discr_order_val[0], dict) and "order" in discr_order_val[0]:
                    discr_order = int(discr_order_val[0]["order"])
                else:
                    discr_order = int(discr_order_val[0])
            else:
                discr_order = int(discr_order_val)
        elif "space" in full_config and isinstance(full_config["space"], dict):
            space_discr = full_config["space"].get("discr_order", 1)
            # Handle list format: [{"id": 2, "order": 2}]
            if isinstance(space_discr, list) and len(space_discr) > 0:
                if isinstance(space_discr[0], dict) and "order" in space_discr[0]:
                    discr_order = int(space_discr[0]["order"])
                else:
                    discr_order = int(space_discr[0])
            else:
                discr_order = int(space_discr)
        
        # Extract materials - preserve full array format when provided.
        materials = full_config.get("materials", {})
        materials_dict: Union[Dict[str, Any], List[Dict[str, Any]]] = {}
        if isinstance(materials, list) and len(materials) > 0:
            materials_dict = []
            for mat in materials:
                if isinstance(mat, dict):
                    materials_dict.append(dict(mat))
        elif isinstance(materials, dict):
            materials_dict = dict(materials)
        
        # Extract boundary_conditions - convert to BoundaryConditions if dict
        boundary_conditions_raw = full_config.get("boundary_conditions", {})
        if isinstance(boundary_conditions_raw, dict):
            boundary_conditions = BoundaryConditions.from_dict(boundary_conditions_raw)
        else:
            boundary_conditions = boundary_conditions_raw
        
        # Extract geometry - convert to Geometry if present
        geometry_raw = full_config.get("geometry")
        geometry = None
        if geometry_raw is not None:
            geometry = Geometry.from_dict(geometry_raw)

        units_raw = full_config.get("units")
        units = None
        if units_raw is not None and isinstance(units_raw, dict):
            units = Units.from_dict(units_raw)
        
        # Extract solver - convert to Solver if present
        solver_raw = full_config.get("solver")
        solver = None
        if solver_raw is not None and isinstance(solver_raw, dict):
            solver = Solver.from_dict(solver_raw)
        
        # Extract time - convert to Time if present
        time_raw = full_config.get("time")
        time = None
        if time_raw is not None and isinstance(time_raw, dict):
            time = Time.from_dict(time_raw)
        
        # Extract output - convert to Output if present
        output_raw = full_config.get("output")
        output = None
        if output_raw is not None and isinstance(output_raw, dict):
            output = Output.from_dict(output_raw)
        
        # Extract contact - convert to Contact if present
        contact_raw = full_config.get("contact")
        contact = None
        if contact_raw is not None and isinstance(contact_raw, dict):
            contact = Contact.from_dict(contact_raw)

        initial_conditions_raw = full_config.get("initial_conditions")
        initial_conditions = None
        if initial_conditions_raw is not None and isinstance(initial_conditions_raw, dict):
            initial_conditions = InitialConditions.from_dict(initial_conditions_raw)

        constraints_raw = full_config.get("constraints")
        constraints = None
        if constraints_raw is not None and isinstance(constraints_raw, dict):
            constraints = Constraints.from_dict(constraints_raw)

        space_raw = full_config.get("space")
        space = None
        if space_raw is not None and isinstance(space_raw, dict):
            space = Space.from_dict(space_raw)

        tests_raw = full_config.get("tests")
        tests = None
        if tests_raw is not None and isinstance(tests_raw, dict):
            tests = Tests.from_dict(tests_raw)

        input_raw = full_config.get("input")
        input_cfg = None
        if input_raw is not None and isinstance(input_raw, dict):
            input_cfg = Input.from_dict(input_raw)
        
        # Extract problem_type and problem_params
        problem_type = full_config.get("problem_type")
        problem_params_raw = full_config.get("problem_params", {})
        
        # Convert problem_params dict to appropriate class based on problem_type
        problem_params = problem_params_raw
        if problem_type and isinstance(problem_params_raw, dict) and problem_params_raw:
            if problem_type == "Gravity":
                problem_params = GravityParams.from_dict(problem_params_raw)
            elif problem_type == "TorsionElastic":
                problem_params = TorsionParams.from_dict(problem_params_raw)
            elif problem_type == "Flow":
                problem_params = FlowParams.from_dict(problem_params_raw)
            elif problem_type == "FlowWithObstacle":
                problem_params = FlowWithObstacleParams.from_dict(problem_params_raw)
            # For other problem types, keep as dict
        
        # Store full JSON in extras for direct use by solve()
        # This ensures all fields (geometry, contact, time, output, solver, etc.) are preserved
        extras = dict(full_config.get("extras", {}))
        extras["_full_json_config"] = full_config
        
        return cls(
            pde=pde,
            discr_order=discr_order,
            materials=materials_dict,
            boundary_conditions=boundary_conditions,
            geometry=geometry,
            units=units,
            solver=solver,
            time=time,
            output=output,
            contact=contact,
            initial_conditions=initial_conditions,
            constraints=constraints,
            space=space,
            tests=tests,
            input=input_cfg,
            extras=extras,
            problem_type=problem_type,
            problem_params=problem_params,
        )
    
    @classmethod
    def from_json_file(cls, filepath: str) -> "SimulationConfig":
        """Load configuration from a JSON file.
        
        Args:
            filepath: Path to JSON configuration file.
            
        Returns:
            A `SimulationConfig` instance loaded from the file.
            
        Example:
            >>> cfg = SimulationConfig.from_json_file("config.json")
        """
        from pathlib import Path
        json_path = Path(filepath).resolve()
        
        with open(json_path, "r") as f:
            config_dict = json.load(f)
        
        # Store root_path for resolving relative paths (e.g., mesh paths)
        config_dict["root_path"] = str(json_path)
        
        cfg = cls.from_json_dict(config_dict)
        
        # Also store root_path in extras for solve() to use
        if not hasattr(cfg, 'extras') or cfg.extras is None:
            cfg.extras = {}
        cfg.extras["_root_path"] = str(json_path)
        
        return cfg

    # ---------------- Validation ----------------

    @staticmethod
    def _is_numeric_or_unit_wrapped(value: Any) -> bool:
        """True iff ``value`` is a plain number or a ``{"value": number, "unit": str}``
        dict (the latter is the form the PolyFEM JSON schema accepts for physical
        quantities like ``E``)."""
        if hasattr(value, "to_dict") and callable(getattr(value, "to_dict")):
            try:
                value = value.to_dict()
            except TypeError:
                pass
        if isinstance(value, bool):
            # ``bool`` is an ``int`` subclass; reject it explicitly so that
            # ``E=True`` does not silently pass validation.
            return False
        if isinstance(value, (int, float)):
            return True
        if isinstance(value, dict):
            inner = value.get("value")
            return isinstance(inner, (int, float)) and not isinstance(inner, bool)
        return False

    @classmethod
    def _validate_material_entry(cls, entry: Dict[str, Any], *, prefix: str) -> None:
        """Validate a single material dict in place (no return value).

        Accepts both the plain numeric form (``E = 20``) and the unit-wrapped form
        (``E = {"value": 20, "unit": "MPa"}``) so that JSON configs loaded via
        ``from_json_dict`` validate cleanly without being rewritten first.
        """
        for key in ("E", "nu", "lambda", "mu"):
            if key not in entry:
                continue
            value = entry[key]
            if cls._is_numeric_or_unit_wrapped(value):
                continue
            raise ValueError(
                f"{prefix}['{key}'] must be a number or a "
                f"{{'value': number, 'unit': str}} dict, "
                f"got {type(value).__name__}: {value!r}"
            )

        material_type = entry.get("type")
        if material_type in ("NeoHookean", "IsochoricNeoHookean", "LinearElasticity"):
            _validate_mode_choice(
                entry,
                prefix=prefix,
                material_type=material_type,
                modes=[
                    ("(E, nu)", ("E", "nu")),
                    ("(lambda, mu)", ("lambda", "mu")),
                ],
            )
        elif material_type in ("HookeLinearElasticity", "SaintVenant"):
            _validate_mode_choice(
                entry,
                prefix=prefix,
                material_type=material_type,
                modes=[
                    ("(E, nu)", ("E", "nu")),
                    ("elasticity_tensor", ("elasticity_tensor",)),
                ],
            )

    def validate(self) -> None:
        """Perform lightweight sanity checks on key fields.

        Checks include:
        - ``discr_order`` must be a positive integer.
        - If present, ``E`` / ``nu`` in any material must be either a plain number
          or a unit-wrapped ``{"value": number, "unit": str}`` dict. Works for
          both the single-dict form (``materials = {...}``) and the list form
          (``materials = [{...}, {...}]``) that ``from_json_dict`` produces.

        Raises:
            ValueError: If any check fails.

        Notes:
            - Physics-range checks (e.g., ``0 < nu < 0.5``) are intentionally
              not enforced here; this is a shape/type validator only.
            - Call ``canonicalized()`` beforehand to normalize aliases.
        """
        if not isinstance(self.discr_order, int) or self.discr_order <= 0:
            raise ValueError(
                f"discr_order must be a positive integer, got {self.discr_order!r}"
            )

        mats = self._get_materials_dict()
        if isinstance(mats, dict):
            self._validate_material_entry(mats, prefix="materials")
        elif isinstance(mats, list):
            for idx, entry in enumerate(mats):
                if isinstance(entry, dict):
                    self._validate_material_entry(entry, prefix=f"materials[{idx}]")

        time_cfg = self._get_time_dict()
        if isinstance(time_cfg, dict):
            provided = [
                time_cfg.get("tend") is not None,
                time_cfg.get("dt") is not None,
                time_cfg.get("time_steps") is not None,
            ]
            if sum(provided) < 2:
                raise ValueError(
                    "time requires at least two of tend / dt / time_steps; "
                    f"got {time_cfg!r}"
                )

    def _ensure_materials_list(self) -> List[Any]:
        if isinstance(self.materials, list):
            return self.materials
        if self.materials in ({}, None):
            self.materials = []
        else:
            self.materials = [self.materials]
        return self.materials

    def _ensure_geometry_object(self) -> "Geometry":
        if self.geometry is None:
            self.geometry = Geometry(items=[])
        elif isinstance(self.geometry, Geometry):
            pass
        elif isinstance(self.geometry, (list, dict)):
            self.geometry = Geometry.from_dict(self.geometry)
        else:
            raise TypeError(f"Unsupported geometry container type: {type(self.geometry).__name__}")
        return self.geometry

    def _ensure_boundary_conditions_object(self) -> BoundaryConditions:
        if not isinstance(self.boundary_conditions, BoundaryConditions):
            if isinstance(self.boundary_conditions, dict):
                self.boundary_conditions = BoundaryConditions.from_dict(self.boundary_conditions)
            else:
                self.boundary_conditions = BoundaryConditions()
        return self.boundary_conditions

    def _ensure_initial_conditions_object(self) -> InitialConditions:
        if not isinstance(self.initial_conditions, InitialConditions):
            if isinstance(self.initial_conditions, dict):
                self.initial_conditions = InitialConditions.from_dict(self.initial_conditions)
            else:
                self.initial_conditions = InitialConditions()
        return self.initial_conditions

    @staticmethod
    def _normalize_scalar_id(value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, list):
            if len(value) != 1:
                raise ValueError(f"Expected a single body id, got list {value!r}")
            value = value[0]
        return int(value)

    def _next_body_id(self) -> int:
        existing: List[int] = []

        mats = self._get_materials_dict()
        if isinstance(mats, dict):
            maybe = self._normalize_scalar_id(mats.get("id"))
            if maybe is not None:
                existing.append(maybe)
        elif isinstance(mats, list):
            for entry in mats:
                if isinstance(entry, dict):
                    maybe = self._normalize_scalar_id(entry.get("id"))
                    if maybe is not None:
                        existing.append(maybe)

        geom = self._get_geometry_dict()
        geom_entries = geom if isinstance(geom, list) else ([geom] if isinstance(geom, dict) else [])
        for entry in geom_entries:
            if isinstance(entry, dict):
                maybe = self._normalize_scalar_id(entry.get("volume_selection"))
                if maybe is not None:
                    existing.append(maybe)

        return max(existing or [0]) + 1

    def _next_surface_selection_id(self) -> int:
        existing: List[int] = []
        geom = self._get_geometry_dict()
        geom_entries = geom if isinstance(geom, list) else ([geom] if isinstance(geom, dict) else [])
        for entry in geom_entries:
            if not isinstance(entry, dict):
                continue
            selections = entry.get("surface_selection")
            if isinstance(selections, dict):
                selections = [selections]
            if isinstance(selections, list):
                for item in selections:
                    if isinstance(item, dict) and item.get("id") is not None:
                        existing.append(int(item["id"]))
        return max(existing or [0]) + 1
    
    # ---------------- Convenience methods for setting parameters ----------------

    def add_body(
        self,
        *,
        geometry: Any,
        material: Any,
        name: str = "",
        id: Optional[int] = None,
    ) -> Body:
        """Add a body and keep material/geometry IDs aligned automatically."""
        explicit_body_id = self._normalize_scalar_id(id)

        material_id = None
        if hasattr(material, "id"):
            material_id = self._normalize_scalar_id(getattr(material, "id"))
        elif isinstance(material, dict):
            material_id = self._normalize_scalar_id(material.get("id"))

        geometry_id = None
        if hasattr(geometry, "volume_selection"):
            geometry_id = self._normalize_scalar_id(getattr(geometry, "volume_selection"))
        elif isinstance(geometry, dict):
            geometry_id = self._normalize_scalar_id(geometry.get("volume_selection"))

        candidate_ids = [v for v in (explicit_body_id, material_id, geometry_id) if v not in (None, 0)]
        if candidate_ids and len(set(candidate_ids)) > 1:
            raise ValueError(
                "Body id mismatch between explicit id / material.id / geometry.volume_selection: "
                f"{candidate_ids}"
            )

        body_id = candidate_ids[0] if candidate_ids else self._next_body_id()

        if hasattr(material, "id"):
            setattr(material, "id", body_id)
        elif isinstance(material, dict):
            material = dict(material)
            material["id"] = body_id

        if hasattr(geometry, "volume_selection"):
            setattr(geometry, "volume_selection", body_id)
        elif isinstance(geometry, dict):
            geometry = dict(geometry)
            geometry["volume_selection"] = body_id

        self._ensure_materials_list().append(material)
        self._ensure_geometry_object().add(geometry)

        return Body(config=self, geometry=geometry, material=material, volume_id=body_id, name=name)
    
    def set_material(self, E: Optional[float] = None, nu: Optional[float] = None,
                     rho: Optional[float] = None, material_type: str = "LinearElasticity") -> "SimulationConfig":
        """Set material parameters using convenient method (IDE autocomplete supported).
        
        Args:
            E: Young's modulus.
            nu: Poisson's ratio.
            rho: Density.
            material_type: Material type. Defaults to "LinearElasticity".
            
        Returns:
            self for method chaining.
            
        Example:
            >>> cfg = SimulationConfig()
            >>> cfg.set_material(E=2100, nu=0.3)  # IDE will autocomplete parameters
        """
        self.materials = Material(E=E, nu=nu, rho=rho, type=material_type)
        return self
    
    def set_dirichlet_boundary(self, id: Optional[int] = None, selection: Optional[int] = None,
                              value: List[float] = None) -> "SimulationConfig":
        """Add a Dirichlet boundary condition (IDE autocomplete supported).
        
        Args:
            id: Boundary ID (sideset ID).
            selection: Alternative to id (selection identifier).
            value: Displacement values (e.g., [0.0, 0.0] for 2D).
            
        Returns:
            self for method chaining.
            
        Example:
            >>> cfg = SimulationConfig()
            >>> cfg.set_dirichlet_boundary(id=4, value=[0.0, 0.0])  # IDE will autocomplete
        """
        if not isinstance(self.boundary_conditions, BoundaryConditions):
            self.boundary_conditions = BoundaryConditions()
        self.boundary_conditions.add_dirichlet(id=id, selection=selection, value=value or [])
        return self
    
    def set_neumann_boundary(self, id: Optional[int] = None, selection: Optional[int] = None,
                             value: List[float] = None) -> "SimulationConfig":
        """Add a Neumann boundary condition (IDE autocomplete supported).
        
        Args:
            id: Boundary ID (sideset ID).
            selection: Alternative to id (selection identifier).
            value: Force/traction values (e.g., [0.0, -1000.0] for 2D).
            
        Returns:
            self for method chaining.
            
        Example:
            >>> cfg = SimulationConfig()
            >>> cfg.set_neumann_boundary(id=2, value=[0.0, -1000.0])  # IDE will autocomplete
        """
        if not isinstance(self.boundary_conditions, BoundaryConditions):
            self.boundary_conditions = BoundaryConditions()
        self.boundary_conditions.add_neumann(id=id, selection=selection, value=value or [])
        return self
    
    def set_rhs(self, value: List[float]) -> "SimulationConfig":
        """Set body force (right-hand side) (IDE autocomplete supported).
        
        Args:
            value: Body force values (e.g., [1.0, 0.0] for 2D).
            
        Returns:
            self for method chaining.
            
        Example:
            >>> cfg = SimulationConfig()
            >>> cfg.set_rhs([1.0, 0.0])  # IDE will autocomplete
        """
        if not isinstance(self.boundary_conditions, BoundaryConditions):
            self.boundary_conditions = BoundaryConditions()
        self.boundary_conditions.set_rhs(value)
        return self

    # ---------------- Backend mapping ----------------

    def to_settings(self):
        """Construct a `polyfempy.Settings` object from the canonical configuration.

        Strategy:
            - PDE:
                * 'Poisson'           → `pf.GenericScalar()`
                * 'LinearElasticity'  → `pf.GenericTensor()`
            - Order: pass `discr_order` to `pf.Settings(...)`.
            - Materials: set `E`/`nu` if available.
            - Extras: if supported, pass through via `set_advanced_option`.

        Returns:
            A `pf.Settings` instance configured with problem, order, and materials.

        Raises:
            RuntimeError: If `polyfempy` is not installed or import fails.

        Notes:
            - If the compiled extension is missing, a lightweight `_DummySettings`
              is used to keep the API and tests runnable.
            - Boundary conditions are stored on this config but typically applied
              later in `solve()`, where mesh context is available.
        """
        try:
            import polyfempy as pf
        except Exception as e:
            raise RuntimeError("polyfempy is required to construct Settings.") from e

        c = self.canonicalized()
        self.validate()

        # Create settings (with order). If the compiled extension is missing,
        # fall back to a lightweight placeholder to keep the API usable.
        if hasattr(pf, "Settings"):
            settings = pf.Settings(discr_order=c.discr_order)
        else:
            class _DummySettings:
                """Minimal placeholder for environments without the compiled backend.

                Only records calls for debugging and keeps method names compatible
                with common `polyfempy.Settings` variants.
                """
                def __init__(self, discr_order):
                    self.discr_order = discr_order
                    self._problem = None

                def set_problem(self, problem):
                    """Attach a problem object (placeholder)."""
                    self._problem = problem

                def set_pde(self, pde_enum):
                    """Set PDE by enum (no-op in placeholder)."""
                    pass

                def set_material_params(self, key, value):
                    """Record a material parameter (placeholder)."""
                    if not hasattr(self, "_materials"):
                        self._materials = {}
                    self._materials[key] = value

                def set_advanced_option(self, key, value):
                    """Record an advanced option (placeholder)."""
                    if not hasattr(self, "_extras"):
                        self._extras = {}
                    self._extras[key] = value

            settings = _DummySettings(discr_order=c.discr_order)

        # Choose problem type (predefined problems take precedence)
        problem = None
        if c.problem_type:
            # Use predefined problem
            problem_name = c.problem_type
            problem_class = getattr(pf, problem_name, None)
            if problem_class is None:
                from polyfempy.api.problems import get_problem_class

                problem_class = get_problem_class(problem_name)
            
            if problem_class is None:
                raise ValueError(f"Unknown problem type: {problem_name}. "
                               f"Available: Franke, Gravity, TorsionElastic, Flow, DrivenCavity, FlowWithObstacle, "
                               f"GenericScalar, GenericTensor")
            
            # Create problem with params
            problem_params_dict = c._get_problem_params_dict() if hasattr(c, '_get_problem_params_dict') else (dict(c.problem_params) if c.problem_params else {})
            if problem_params_dict:
                problem = problem_class(**problem_params_dict)
            else:
                problem = problem_class()
        else:
            # Use default based on PDE
            if c.pde == "Poisson":
                problem = getattr(pf, "GenericScalar", lambda: object())()
            else:
                problem = getattr(pf, "GenericTensor", lambda: object())()

        # Attach Problem (handle minor API differences across versions)
        if hasattr(settings, "set_problem"):
            settings.set_problem(problem)
        elif hasattr(settings, "set_pde") and hasattr(pf, "PDEs"):
            enum_val = getattr(pf.PDEs, c.pde, None)
            if enum_val is not None:
                settings.set_pde(enum_val)
            elif hasattr(settings, "_problem"):
                settings._problem = problem  # last resort
        elif hasattr(settings, "_problem"):
            settings._problem = problem

        # Material params (common E / nu)
        mats = c._get_materials_dict() if hasattr(c, '_get_materials_dict') else _canon_materials(c.materials)
        if hasattr(settings, "set_material_params"):
            if "E" in mats:
                settings.set_material_params("E", float(mats["E"]))
            if "nu" in mats:
                settings.set_material_params("nu", float(mats["nu"]))

        # Advanced extras (best-effort; ignore failures)
        if c.extras and hasattr(settings, "set_advanced_option"):
            for k, v in c.extras.items():
                try:
                    settings.set_advanced_option(k, v)
                except Exception:
                    pass
        
        # Selection: store in settings for later use in solve()
        if c.selection is not None:
            # Store selection dict in settings for solve() to use
            if not hasattr(settings, "_selection"):
                settings._selection = c.selection.to_dict()

        return settings

    # ---------------- Convenience factories ----------------

    @classmethod
    def linear_elasticity(cls, E: float, nu: float, order: int = 1) -> "SimulationConfig":
        """Create a linear elasticity configuration.

        Args:
            E: Young's modulus.
            nu: Poisson's ratio.
            order: Polynomial order (`discr_order`). Defaults to 1.

        Returns:
            A `SimulationConfig` with `pde='LinearElasticity'` and given materials.

        Example:
            >>> SimulationConfig.linear_elasticity(2100, 0.3, order=2)
        """
        return cls(pde="LinearElasticity", discr_order=order, materials={"E": E, "nu": nu})

    @classmethod
    def poisson(cls, order: int = 1) -> "SimulationConfig":
        """Create a Poisson configuration.

        Args:
            order: Polynomial order (`discr_order`). Defaults to 1.

        Returns:
            A `SimulationConfig` with `pde='Poisson'`.

        Example:
            >>> SimulationConfig.poisson(order=1)
        """
        return cls(pde="Poisson", discr_order=order)
    
    # ---------------- Predefined problem factories ----------------
    
    @classmethod
    def gravity(cls, force: float = 0.1, E: float = None, nu: float = None, order: int = 1) -> "SimulationConfig":
        """Create a gravity problem configuration.
        
        Args:
            force: Gravity force magnitude. Defaults to 0.1.
            E: Young's modulus (optional).
            nu: Poisson's ratio (optional).
            order: Polynomial order. Defaults to 1.
            
        Returns:
            A `SimulationConfig` with `problem_type='Gravity'`.
            
        Example:
            >>> SimulationConfig.gravity(force=0.1, E=1e6, nu=0.3)
        """
        materials = {}
        if E is not None:
            materials["E"] = E
        if nu is not None:
            materials["nu"] = nu
        return cls(
            pde="LinearElasticity",
            discr_order=order,
            materials=materials,
            problem_type="Gravity",
            problem_params=GravityParams(force=force),
        )
    
    @classmethod
    def franke(cls, order: int = 1) -> "SimulationConfig":
        """Create a Franke problem configuration (scalar problem with exact solution).
        
        Args:
            order: Polynomial order. Defaults to 1.
            
        Returns:
            A `SimulationConfig` with `problem_type='Franke'`.
            
        Example:
            >>> SimulationConfig.franke(order=2)
        """
        return cls(
            pde="Poisson",
            discr_order=order,
            problem_type="Franke",
        )
    
    @classmethod
    def torsion(
        cls,
        axis_coordinate: int = 2,
        n_turns: float = 0.5,
        fixed_boundary: int = 5,
        turning_boundary: int = 6,
        E: float = None,
        nu: float = None,
        order: int = 1,
    ) -> "SimulationConfig":
        """Create a torsion problem configuration (3D).
        
        Args:
            axis_coordinate: Axis direction (1=x, 2=y, 3=z). Defaults to 2 (y-axis).
            n_turns: Number of turns. Defaults to 0.5.
            fixed_boundary: Sideset ID for fixed boundary. Defaults to 5.
            turning_boundary: Sideset ID for turning boundary. Defaults to 6.
            E: Young's modulus (optional).
            nu: Poisson's ratio (optional).
            order: Polynomial order. Defaults to 1.
            
        Returns:
            A `SimulationConfig` with `problem_type='TorsionElastic'` (Legacy API name).
            
        Example:
            >>> SimulationConfig.torsion(axis_coordinate=2, n_turns=0.5)
        """
        materials = {}
        if E is not None:
            materials["E"] = E
        if nu is not None:
            materials["nu"] = nu
        # Note: Legacy API uses "TorsionElastic" as the problem name
        # Also note: Legacy API has a typo "axis_coordiante" but we use the correct spelling
        # The Legacy class handles both spellings
        return cls(
            pde="LinearElasticity",
            discr_order=order,
            materials=materials,
            problem_type="TorsionElastic",  # Legacy API uses "TorsionElastic", not "Torsion"
            problem_params=TorsionParams(
                axis_coordinate=axis_coordinate,
                n_turns=n_turns,
                fixed_boundary=fixed_boundary,
                turning_boundary=turning_boundary,
            ),
        )
    
    @classmethod
    def flow(
        cls,
        inflow: int = 1,
        outflow: int = 3,
        inflow_amount: float = 0.25,
        outflow_amount: float = 0.25,
        direction: int = 0,
        obstacle: list = None,
        order: int = 1,
    ) -> "SimulationConfig":
        """Create a flow problem configuration (inflow/outflow).
        
        Args:
            inflow: Sideset ID for inflow. Defaults to 1.
            outflow: Sideset ID for outflow. Defaults to 3.
            inflow_amount: Inflow amount. Defaults to 0.25.
            outflow_amount: Outflow amount. Defaults to 0.25.
            direction: Flow direction. Defaults to 0.
            obstacle: List of obstacle sideset IDs. Defaults to [7].
            order: Polynomial order. Defaults to 1.
            
        Returns:
            A `SimulationConfig` with `problem_type='Flow'`.
            
        Example:
            >>> SimulationConfig.flow(inflow=1, outflow=3)
        """
        if obstacle is None:
            obstacle = [7]
        return cls(
            pde="Stokes",  # Flow problems use Stokes PDE
            discr_order=order,
            problem_type="Flow",
            problem_params=FlowParams(
                inflow=inflow,
                outflow=outflow,
                inflow_amount=inflow_amount,  # Correct spelling in class
                outflow_amount=outflow_amount,  # Correct spelling in class
                direction=direction,
                obstacle=obstacle,
            ),
        )
    
    @classmethod
    def driven_cavity(cls, order: int = 1) -> "SimulationConfig":
        """Create a driven cavity problem configuration.
        
        Args:
            order: Polynomial order. Defaults to 1.
            
        Returns:
            A `SimulationConfig` with `problem_type='DrivenCavity'`.
            
        Example:
            >>> SimulationConfig.driven_cavity(order=2)
        """
        return cls(
            pde="Stokes",
            discr_order=order,
            problem_type="DrivenCavity",
        )
    
    @classmethod
    def flow_with_obstacle(cls, U: float = 1.5, time_dependent: bool = True, order: int = 1) -> "SimulationConfig":
        """Create a flow with obstacle problem configuration.
        
        Args:
            U: Flow velocity. Defaults to 1.5.
            time_dependent: Whether the problem is time-dependent. Defaults to True.
            order: Polynomial order. Defaults to 1.
            
        Returns:
            A `SimulationConfig` with `problem_type='FlowWithObstacle'`.
            
        Example:
            >>> SimulationConfig.flow_with_obstacle(U=1.5)
        """
        return cls(
            pde="Stokes",
            discr_order=order,
            problem_type="FlowWithObstacle",
            problem_params=FlowWithObstacleParams(U=U, time_dependent=time_dependent),
        )


# ============================================================================
# Geometry Configuration Classes
# ============================================================================

@dataclass
class GeometryTransformation:
    """Per-geometry transformation block."""

    translation: List[float] = field(default_factory=list)
    rotation: List[float] = field(default_factory=list)
    scale: List[float] = field(default_factory=list)
    dimensions: int = 1
    rotation_mode: str = "xyz"

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        if self.translation:
            result["translation"] = list(self.translation)
        if self.rotation:
            result["rotation"] = list(self.rotation)
        if self.scale:
            result["scale"] = list(self.scale)
        if self.dimensions != 1:
            result["dimensions"] = self.dimensions
        if self.rotation_mode != "xyz":
            result["rotation_mode"] = self.rotation_mode
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GeometryTransformation":
        return cls(
            translation=list(d.get("translation", [])),
            rotation=list(d.get("rotation", [])),
            scale=list(d.get("scale", [])),
            dimensions=int(d.get("dimensions", 1)),
            rotation_mode=str(d.get("rotation_mode", "xyz")),
        )


@dataclass
class GeometryAdvanced:
    """Advanced per-geometry mesh options."""

    normalize_mesh: bool = False
    force_linear_geometry: bool = False
    refinement_location: float = 0.5
    min_component: int = -1

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        if self.normalize_mesh:
            result["normalize_mesh"] = True
        if self.force_linear_geometry:
            result["force_linear_geometry"] = True
        if self.refinement_location != 0.5:
            result["refinement_location"] = self.refinement_location
        if self.min_component != -1:
            result["min_component"] = self.min_component
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GeometryAdvanced":
        return cls(
            normalize_mesh=bool(d.get("normalize_mesh", False)),
            force_linear_geometry=bool(d.get("force_linear_geometry", False)),
            refinement_location=float(d.get("refinement_location", 0.5)),
            min_component=int(d.get("min_component", -1)),
        )


@dataclass
class GeometryArray:
    """Array replication options for a mesh."""

    offset: List[float] = field(default_factory=list)
    size: List[int] = field(default_factory=list)
    relative: bool = False

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        if self.offset:
            result["offset"] = list(self.offset)
        if self.size:
            result["size"] = list(self.size)
        if self.relative:
            result["relative"] = True
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GeometryArray":
        return cls(
            offset=list(d.get("offset", [])),
            size=[int(v) for v in d.get("size", [])],
            relative=bool(d.get("relative", False)),
        )


@dataclass
class GeometryMesh:
    """Mesh geometry configuration - provides IDE autocomplete support.
    
    Attributes:
        mesh: Path to mesh file (required).
        volume_selection: Volume selection ID (optional).
        surface_selection: Surface selection descriptor (optional). Can be an
            integer ID or the richer JSON list/dict form used by PolyFEM.
        transformation: Transformation matrix or parameters (optional).
        is_obstacle: Whether this mesh is an obstacle (optional).
    
    Example:
        >>> geom = GeometryMesh(mesh="mesh.obj", volume_selection=1)
        >>> obstacle = GeometryMesh(mesh="plane.obj", is_obstacle=True)
    """
    mesh: str = field()
    type: str = "mesh"
    extract: str = "volume"
    unit: str = ""
    array: Optional[Union[GeometryArray, Dict[str, Any]]] = None
    transformation: Optional[Union[GeometryTransformation, Dict[str, Any]]] = None
    volume_selection: Optional[Any] = None
    surface_selection: Optional[Any] = None
    curve_selection: Optional[Any] = None
    point_selection: Optional[Any] = None
    n_refs: int = 0
    advanced: Optional[Union[GeometryAdvanced, Dict[str, Any]]] = None
    enabled: bool = True
    is_obstacle: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {"mesh": self.mesh}
        if self.type != "mesh":
            result["type"] = self.type
        if self.extract != "volume":
            result["extract"] = self.extract
        if self.unit:
            result["unit"] = self.unit
        if self.array is not None:
            result["array"] = _to_plain_value(self.array)
        if self.transformation is not None:
            result["transformation"] = _to_plain_value(self.transformation)
        if self.volume_selection is not None:
            result["volume_selection"] = _to_plain_value(self.volume_selection)
        if self.surface_selection is not None:
            result["surface_selection"] = _to_plain_value(self.surface_selection)
        if self.curve_selection is not None:
            result["curve_selection"] = _to_plain_value(self.curve_selection)
        if self.point_selection is not None:
            result["point_selection"] = _to_plain_value(self.point_selection)
        if self.n_refs != 0:
            result["n_refs"] = self.n_refs
        if self.advanced is not None:
            advanced = _to_plain_value(self.advanced)
            if advanced:
                result["advanced"] = advanced
        if not self.enabled:
            result["enabled"] = False
        if self.is_obstacle:
            result["is_obstacle"] = True
        return result

    @classmethod
    def from_file(
        cls,
        mesh: Union[str, PathLike[str]],
        *,
        volume_selection: Optional[Any] = None,
        surface_selection: Optional[Any] = None,
        curve_selection: Optional[Any] = None,
        point_selection: Optional[Any] = None,
        unit: str = "",
        extract: str = "volume",
        array: Optional[Union[GeometryArray, Dict[str, Any]]] = None,
        transformation: Optional[Union[GeometryTransformation, Dict[str, Any]]] = None,
        n_refs: int = 0,
        advanced: Optional[Union[GeometryAdvanced, Dict[str, Any]]] = None,
        enabled: bool = True,
        is_obstacle: bool = False,
    ) -> "GeometryMesh":
        """Construct the most common mesh-backed geometry entry in one call."""
        return cls(
            mesh=str(mesh),
            unit=unit,
            extract=extract,
            array=array,
            transformation=transformation,
            volume_selection=volume_selection,
            surface_selection=surface_selection,
            curve_selection=curve_selection,
            point_selection=point_selection,
            n_refs=n_refs,
            advanced=advanced,
            enabled=enabled,
            is_obstacle=is_obstacle,
        )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GeometryMesh":
        array = d.get("array")
        if isinstance(array, dict):
            array = GeometryArray.from_dict(array)
        transformation = d.get("transformation")
        if isinstance(transformation, dict):
            transformation = GeometryTransformation.from_dict(transformation)
        advanced = d.get("advanced")
        if isinstance(advanced, dict):
            advanced = GeometryAdvanced.from_dict(advanced)
        return cls(
            mesh=d["mesh"],
            type=d.get("type", "mesh"),
            extract=d.get("extract", "volume"),
            unit=d.get("unit", ""),
            array=array,
            transformation=transformation,
            volume_selection=d.get("volume_selection"),
            surface_selection=d.get("surface_selection"),
            curve_selection=d.get("curve_selection"),
            point_selection=d.get("point_selection"),
            n_refs=int(d.get("n_refs", 0)),
            advanced=advanced,
            enabled=bool(d.get("enabled", True)),
            is_obstacle=bool(d.get("is_obstacle", False)),
        )


@dataclass
class GeometryMeshArray(GeometryMesh):
    """Arrayed mesh geometry entry."""

    array: Optional[Union[GeometryArray, Dict[str, Any]]] = None


@dataclass
class GeometryPlane:
    """Plane geometry object."""

    point: List[float] = field(default_factory=list)
    normal: List[float] = field(default_factory=list)
    type: str = "plane"
    enabled: bool = True
    is_obstacle: bool = False

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "point": list(self.point),
            "normal": list(self.normal),
        }
        if self.type != "plane":
            result["type"] = self.type
        if not self.enabled:
            result["enabled"] = False
        if self.is_obstacle:
            result["is_obstacle"] = True
        return result

    @classmethod
    def obstacle(
        cls,
        *,
        point: List[float],
        normal: List[float],
        enabled: bool = True,
    ) -> "GeometryPlane":
        """Construct an obstacle plane."""
        return cls(point=list(point), normal=list(normal), enabled=enabled, is_obstacle=True)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GeometryPlane":
        return cls(
            point=[float(v) for v in d.get("point", [])],
            normal=[float(v) for v in d.get("normal", [])],
            type=d.get("type", "plane"),
            enabled=bool(d.get("enabled", True)),
            is_obstacle=bool(d.get("is_obstacle", False)),
        )


@dataclass
class GeometryGround:
    """Ground plane orthogonal to gravity."""

    height: float = 0.0
    type: str = "ground"
    enabled: bool = True
    is_obstacle: bool = False

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"height": self.height}
        if self.type != "ground":
            result["type"] = self.type
        if not self.enabled:
            result["enabled"] = False
        if self.is_obstacle:
            result["is_obstacle"] = True
        return result

    @classmethod
    def obstacle(cls, *, height: float = 0.0, enabled: bool = True) -> "GeometryGround":
        """Construct an obstacle ground plane."""
        return cls(height=height, enabled=enabled, is_obstacle=True)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GeometryGround":
        return cls(
            height=float(d.get("height", 0.0)),
            type=d.get("type", "ground"),
            enabled=bool(d.get("enabled", True)),
            is_obstacle=bool(d.get("is_obstacle", False)),
        )


@dataclass
class GeometryMeshSequence:
    """Animated mesh-sequence geometry."""

    mesh_sequence: Union[str, List[str]] = field(default_factory=list)
    fps: int = 1
    type: str = "mesh"
    extract: str = "volume"
    unit: str = ""
    transformation: Optional[Union[GeometryTransformation, Dict[str, Any]]] = None
    n_refs: int = 0
    advanced: Optional[Union[GeometryAdvanced, Dict[str, Any]]] = None
    enabled: bool = True
    is_obstacle: bool = False

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "mesh_sequence": _to_plain_value(self.mesh_sequence),
            "fps": self.fps,
        }
        if self.type != "mesh":
            result["type"] = self.type
        if self.extract != "volume":
            result["extract"] = self.extract
        if self.unit:
            result["unit"] = self.unit
        if self.transformation is not None:
            result["transformation"] = _to_plain_value(self.transformation)
        if self.n_refs != 0:
            result["n_refs"] = self.n_refs
        if self.advanced is not None:
            advanced = _to_plain_value(self.advanced)
            if advanced:
                result["advanced"] = advanced
        if not self.enabled:
            result["enabled"] = False
        if self.is_obstacle:
            result["is_obstacle"] = True
        return result

    @classmethod
    def from_files(
        cls,
        mesh_sequence: Union[str, List[str]],
        *,
        fps: int = 1,
        unit: str = "",
        extract: str = "volume",
        transformation: Optional[Union[GeometryTransformation, Dict[str, Any]]] = None,
        n_refs: int = 0,
        advanced: Optional[Union[GeometryAdvanced, Dict[str, Any]]] = None,
        enabled: bool = True,
        is_obstacle: bool = False,
    ) -> "GeometryMeshSequence":
        """Construct an animated mesh-sequence geometry entry."""
        return cls(
            mesh_sequence=mesh_sequence,
            fps=fps,
            unit=unit,
            extract=extract,
            transformation=transformation,
            n_refs=n_refs,
            advanced=advanced,
            enabled=enabled,
            is_obstacle=is_obstacle,
        )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GeometryMeshSequence":
        transformation = d.get("transformation")
        if isinstance(transformation, dict):
            transformation = GeometryTransformation.from_dict(transformation)
        advanced = d.get("advanced")
        if isinstance(advanced, dict):
            advanced = GeometryAdvanced.from_dict(advanced)
        return cls(
            mesh_sequence=d.get("mesh_sequence", []),
            fps=int(d.get("fps", 1)),
            type=d.get("type", "mesh"),
            extract=d.get("extract", "volume"),
            unit=d.get("unit", ""),
            transformation=transformation,
            n_refs=int(d.get("n_refs", 0)),
            advanced=advanced,
            enabled=bool(d.get("enabled", True)),
            is_obstacle=bool(d.get("is_obstacle", False)),
        )


@dataclass
class Geometry:
    """Geometry configuration - provides IDE autocomplete support.

    This class allows users to configure geometry (mesh files, transformations, etc.)
    with IDE autocomplete, instead of using dictionaries.

    Attributes:
        meshes: List of GeometryMesh objects or mesh file paths (strings).
        transformations: **Deprecated**. Prefer ``GeometryMesh.transformation``
            for per-mesh transformations — that path matches the PolyFEM JSON
            schema exactly (``geometry[i].transformation``) and makes it
            impossible to accidentally broadcast or misalign. When still used,
            the accepted shapes are:

            * ``None`` or empty list — no-op, each mesh keeps its own
              ``transformation`` (if any).
            * Length equal to ``len(meshes)`` — zip-assigned in order, emits
              a ``DeprecationWarning``.
            * Length 1 with multiple meshes — broadcast to all meshes, emits a
              ``DeprecationWarning`` (legacy behavior preserved so old scripts
              don't silently break).

            Any other length now raises ``ValueError`` — previously the excess
            entries were silently dropped. Any mesh that already sets
            ``transformation`` via ``GeometryMesh.transformation`` also raises
            ``ValueError`` rather than getting quietly overwritten.
        selections: Selection configuration (optional).

    Example:
        >>> geom = Geometry(meshes=[GeometryMesh(mesh="mesh.obj")])
        >>> cfg = SimulationConfig(geometry=geom)
    """
    meshes: Union[
        List[Union[GeometryMesh, GeometryMeshArray, GeometryPlane, GeometryGround, GeometryMeshSequence, str, Dict[str, Any]]],
        GeometryMesh,
        GeometryMeshArray,
        GeometryPlane,
        GeometryGround,
        GeometryMeshSequence,
        List[str],
        str,
    ] = field(default_factory=list)
    items: Optional[List[Union[GeometryMesh, GeometryMeshArray, GeometryPlane, GeometryGround, GeometryMeshSequence, str, Dict[str, Any]]]] = None
    transformations: Optional[List[Dict[str, Any]]] = None
    selections: Optional[Dict[str, Any]] = None

    def add(self, item: Union[GeometryMesh, GeometryMeshArray, GeometryPlane, GeometryGround, GeometryMeshSequence, str, Dict[str, Any]]) -> "Geometry":
        if self.items is None:
            if self.meshes and not isinstance(self.meshes, list):
                self.items = [self.meshes]
            elif isinstance(self.meshes, list):
                self.items = list(self.meshes)
            else:
                self.items = []
        self.items.append(item)
        return self

    def _normalized_item_list(self):
        source = self.items if self.items is not None else self.meshes
        if isinstance(source, (str, GeometryMesh, GeometryMeshArray, GeometryPlane, GeometryGround, GeometryMeshSequence)):
            return [source]
        return list(source)

    def _resolve_per_mesh_transformations(self, n_meshes: int) -> Optional[List[Optional[Dict[str, Any]]]]:
        """Return a list of per-mesh transformations (or ``None`` entries) to
        apply via the top-level ``Geometry.transformations`` path.

        Returns ``None`` when the top-level list should not participate (no
        value set / empty). Emits ``DeprecationWarning`` for both legal shapes
        that still use the deprecated broadcast/zip behavior.
        """
        ts = self.transformations
        if ts is None or len(ts) == 0:
            return None

        if len(ts) == n_meshes:
            warnings.warn(
                "Geometry.transformations is deprecated; pass transformation "
                "per mesh via GeometryMesh.transformation instead. The current "
                "length-matched list is being zip-assigned to each mesh.",
                DeprecationWarning,
                stacklevel=3,
            )
            return list(ts)

        if len(ts) == 1 and n_meshes > 1:
            warnings.warn(
                "Geometry.transformations with a single entry broadcasts to "
                "all meshes; this is a deprecated legacy shortcut. Prefer "
                "GeometryMesh.transformation per mesh for explicit intent.",
                DeprecationWarning,
                stacklevel=3,
            )
            return [ts[0]] * n_meshes

        raise ValueError(
            f"Geometry.transformations must be None, a list of length 1, or a "
            f"list of length len(meshes)={n_meshes}; got length {len(ts)}. "
            f"Use GeometryMesh.transformation for per-mesh transformations."
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format (for backend compatibility).

        Returns:
            A list of per-mesh dicts in the PolyFEM JSON geometry format.
        """
        item_list = self._normalized_item_list()
        if not item_list:
            return []

        per_mesh_xform = self._resolve_per_mesh_transformations(len(item_list))

        result: List[Dict[str, Any]] = []
        for idx, geom_item in enumerate(item_list):
            if hasattr(geom_item, "to_dict") and callable(getattr(geom_item, "to_dict")):
                mesh_dict = geom_item.to_dict()
            elif isinstance(geom_item, str):
                mesh_dict = {"mesh": geom_item}
            elif isinstance(geom_item, dict):
                mesh_dict = dict(geom_item)
            else:
                mesh_dict = {"mesh": str(geom_item)}

            if per_mesh_xform is not None:
                top_xform = per_mesh_xform[idx]
                if top_xform is not None:
                    supports_transformation = "mesh" in mesh_dict or "mesh_sequence" in mesh_dict
                    if not supports_transformation:
                        raise ValueError(
                            "Geometry.transformations only applies to mesh-like "
                            "entries (mesh / mesh_array / mesh_sequence). "
                            f"Entry {idx} does not support transformations."
                        )
                    if "transformation" in mesh_dict and mesh_dict["transformation"] is not None:
                        raise ValueError(
                            f"items[{idx}] already has a transformation from "
                            f"GeometryMesh.transformation; refusing to silently "
                            f"overwrite it with Geometry.transformations[{idx}]. "
                            f"Set only one of the two."
                        )
                    mesh_dict["transformation"] = top_xform

            result.append(mesh_dict)

        return result
    
    @classmethod
    def from_dict(cls, d: Union[List[Dict[str, Any]], Dict[str, Any]]) -> "Geometry":
        """Create Geometry from dictionary (backward compatibility).
        
        Args:
            d: Dictionary or list with geometry configuration.
            
        Returns:
            Geometry instance.
        """
        if isinstance(d, list):
            items = []
            for item in d:
                if isinstance(item, dict):
                    if "mesh_sequence" in item:
                        items.append(GeometryMeshSequence.from_dict(item))
                    elif "point" in item and "normal" in item and "mesh" not in item:
                        items.append(GeometryPlane.from_dict(item))
                    elif "height" in item and "mesh" not in item:
                        items.append(GeometryGround.from_dict(item))
                    elif "mesh" in item and "array" in item:
                        items.append(GeometryMeshArray.from_dict(item))
                    elif "mesh" in item:
                        items.append(GeometryMesh.from_dict(item))
                    else:
                        items.append(item)
                else:
                    items.append(item)
            return cls(items=items)
        elif isinstance(d, dict):
            return cls(items=[d])
        else:
            return cls(items=[])


# ============================================================================
# Solver Configuration Classes
# ============================================================================

try:
    from .config_solver import (
        AugmentedLagrangian,
        LineSearch,
        LinearSolver,
        NonlinearSolver,
        RayleighDamping,
        Solver,
        SolverAdvanced,
        SolverContactOptions,
    )
except ImportError:
    import importlib.util as _importlib_util
    import sys as _sys

    _config_solver_path = Path(__file__).resolve().with_name("config_solver.py")
    _config_solver_spec = _importlib_util.spec_from_file_location(
        "_polyfempy_api_config_solver",
        _config_solver_path,
    )
    if _config_solver_spec is None or _config_solver_spec.loader is None:
        raise
    _config_solver_module = _importlib_util.module_from_spec(_config_solver_spec)
    _sys.modules[_config_solver_spec.name] = _config_solver_module
    _config_solver_spec.loader.exec_module(_config_solver_module)

    AugmentedLagrangian = _config_solver_module.AugmentedLagrangian
    LineSearch = _config_solver_module.LineSearch
    LinearSolver = _config_solver_module.LinearSolver
    NonlinearSolver = _config_solver_module.NonlinearSolver
    RayleighDamping = _config_solver_module.RayleighDamping
    Solver = _config_solver_module.Solver
    SolverAdvanced = _config_solver_module.SolverAdvanced
    SolverContactOptions = _config_solver_module.SolverContactOptions


# ============================================================================
# Time Configuration Classes
# ============================================================================

try:
    from .config_time import BDFIntegrator, ImplicitNewmarkIntegrator, Time
except ImportError:
    import importlib.util as _importlib_util
    import sys as _sys

    _config_time_path = Path(__file__).resolve().with_name("config_time.py")
    _config_time_spec = _importlib_util.spec_from_file_location(
        "_polyfempy_api_config_time",
        _config_time_path,
    )
    if _config_time_spec is None or _config_time_spec.loader is None:
        raise
    _config_time_module = _importlib_util.module_from_spec(_config_time_spec)
    _sys.modules[_config_time_spec.name] = _config_time_module
    _config_time_spec.loader.exec_module(_config_time_module)

    BDFIntegrator = _config_time_module.BDFIntegrator
    ImplicitNewmarkIntegrator = _config_time_module.ImplicitNewmarkIntegrator
    Time = _config_time_module.Time


# ============================================================================
# Output Configuration Classes
# ============================================================================

try:
    from .config_output import (
        FallbackOutput,
        Output,
        OutputAdvanced,
        OutputData,
        OutputDataAdvanced,
        OutputLog,
        OutputParaviewOptions,
        OutputReference,
        ParaviewOutput,
        ResultOutput,
    )
except ImportError:
    import importlib.util as _importlib_util
    import sys as _sys

    _config_output_path = Path(__file__).resolve().with_name("config_output.py")
    _config_output_spec = _importlib_util.spec_from_file_location(
        "_polyfempy_api_config_output",
        _config_output_path,
    )
    if _config_output_spec is None or _config_output_spec.loader is None:
        raise
    _config_output_module = _importlib_util.module_from_spec(_config_output_spec)
    _sys.modules[_config_output_spec.name] = _config_output_module
    _config_output_spec.loader.exec_module(_config_output_module)

    FallbackOutput = _config_output_module.FallbackOutput
    Output = _config_output_module.Output
    OutputAdvanced = _config_output_module.OutputAdvanced
    OutputData = _config_output_module.OutputData
    OutputDataAdvanced = _config_output_module.OutputDataAdvanced
    OutputLog = _config_output_module.OutputLog
    OutputParaviewOptions = _config_output_module.OutputParaviewOptions
    OutputReference = _config_output_module.OutputReference
    ParaviewOutput = _config_output_module.ParaviewOutput
    ResultOutput = _config_output_module.ResultOutput


# ============================================================================
# Contact Configuration Classes
# ============================================================================

@dataclass
class CollisionMesh:
    """Collision mesh options for contact."""

    enabled: bool = True
    tessellation_type: str = "regular"
    mesh: Optional[str] = None
    linear_map: Optional[str] = None
    max_edge_length: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        if not self.enabled:
            result["enabled"] = False
        if self.tessellation_type != "regular":
            result["tessellation_type"] = self.tessellation_type
        if self.mesh:
            result["mesh"] = self.mesh
        if self.linear_map:
            result["linear_map"] = self.linear_map
        if self.max_edge_length is not None:
            result["max_edge_length"] = self.max_edge_length
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CollisionMesh":
        return cls(
            enabled=bool(d.get("enabled", True)),
            tessellation_type=str(d.get("tessellation_type", "regular")),
            mesh=d.get("mesh"),
            linear_map=d.get("linear_map"),
            max_edge_length=d.get("max_edge_length"),
        )


@dataclass
class Adhesion:
    """Adhesion options for contact."""

    adhesion_enabled: bool = False
    dhat_p: float = 0.001
    dhat_a: float = 0.01
    adhesion_strength: float = 0.001
    tangential_adhesion_coefficient: float = 0.0
    epsa: float = 0.001

    def to_dict(self) -> Dict[str, Any]:
        defaults = type(self)()
        result: Dict[str, Any] = {}
        for key in self.__dataclass_fields__:
            value = getattr(self, key)
            if value != getattr(defaults, key):
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Adhesion":
        defaults = cls()
        kwargs = {key: d.get(key, getattr(defaults, key)) for key in cls.__dataclass_fields__}
        return cls(**kwargs)


@dataclass
class Contact:
    """Contact configuration - provides IDE autocomplete support.
    
    Attributes:
        enabled: Enable contact. Defaults to False.
        dhat: Contact distance threshold. Defaults to 0.01.
        mu: Friction coefficient. Defaults to 0.0.
        epsv: Viscosity parameter. Defaults to 0.0.
        barrier_stiffness: Barrier stiffness. Defaults to 1e3.
    
    Example:
        >>> contact = Contact(enabled=True, dhat=0.01, mu=0.5)
    """
    enabled: bool = False
    dhat: float = 0.001
    dhat_percentage: float = 0.8
    epsv: float = 0.001
    friction_coefficient: float = 0.0
    mu: Optional[float] = None
    use_convergent_formulation: bool = False
    use_area_weighting: bool = True
    use_improved_max_operator: bool = True
    use_physical_barrier: bool = True
    collision_mesh: Optional[Union[CollisionMesh, Dict[str, Any]]] = None
    use_gcp_formulation: bool = False
    alpha_n: float = 0.5
    alpha_t: float = 0.5
    min_distance_ratio: float = 0.5
    use_adaptive_dhat: bool = False
    periodic: bool = False
    adhesion: Optional[Union[Adhesion, Dict[str, Any]]] = None
    barrier_stiffness: Optional[Any] = None

    @classmethod
    def frictionless(
        cls,
        *,
        dhat: float = 0.001,
        barrier_stiffness: Optional[Any] = None,
        use_adaptive_dhat: bool = False,
        periodic: bool = False,
        collision_mesh: Optional[Union[CollisionMesh, Dict[str, Any]]] = None,
        adhesion: Optional[Union[Adhesion, Dict[str, Any]]] = None,
    ) -> "Contact":
        """Construct an enabled frictionless contact model."""
        return cls(
            enabled=True,
            dhat=dhat,
            friction_coefficient=0.0,
            mu=0.0,
            barrier_stiffness=barrier_stiffness,
            use_adaptive_dhat=use_adaptive_dhat,
            periodic=periodic,
            collision_mesh=collision_mesh,
            adhesion=adhesion,
        )

    @classmethod
    def coulomb(
        cls,
        *,
        mu: float,
        dhat: float = 0.001,
        barrier_stiffness: Optional[Any] = None,
        use_adaptive_dhat: bool = False,
        periodic: bool = False,
        collision_mesh: Optional[Union[CollisionMesh, Dict[str, Any]]] = None,
        adhesion: Optional[Union[Adhesion, Dict[str, Any]]] = None,
    ) -> "Contact":
        """Construct an enabled Coulomb-friction contact model."""
        return cls(
            enabled=True,
            dhat=dhat,
            friction_coefficient=mu,
            mu=mu,
            barrier_stiffness=barrier_stiffness,
            use_adaptive_dhat=use_adaptive_dhat,
            periodic=periodic,
            collision_mesh=collision_mesh,
            adhesion=adhesion,
        )

    @classmethod
    def adhesive(
        cls,
        *,
        adhesion_strength: float = 0.001,
        mu: float = 0.0,
        dhat: float = 0.001,
        dhat_p: float = 0.001,
        dhat_a: float = 0.01,
        tangential_adhesion_coefficient: float = 0.0,
        epsa: float = 0.001,
        barrier_stiffness: Optional[Any] = None,
        use_adaptive_dhat: bool = False,
        periodic: bool = False,
        collision_mesh: Optional[Union[CollisionMesh, Dict[str, Any]]] = None,
    ) -> "Contact":
        """Construct an enabled adhesive contact model."""
        return cls(
            enabled=True,
            dhat=dhat,
            friction_coefficient=mu,
            mu=mu,
            barrier_stiffness=barrier_stiffness,
            use_adaptive_dhat=use_adaptive_dhat,
            periodic=periodic,
            collision_mesh=collision_mesh,
            adhesion=Adhesion(
                adhesion_enabled=True,
                dhat_p=dhat_p,
                dhat_a=dhat_a,
                adhesion_strength=adhesion_strength,
                tangential_adhesion_coefficient=tangential_adhesion_coefficient,
                epsa=epsa,
            ),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result: Dict[str, Any] = {}
        defaults = type(self)()
        alias_mu = self.friction_coefficient if self.mu is None else self.mu
        if self.enabled:
            result["enabled"] = True
        if self.dhat != defaults.dhat:
            result["dhat"] = self.dhat
        if self.dhat_percentage != defaults.dhat_percentage:
            result["dhat_percentage"] = self.dhat_percentage
        if self.epsv != defaults.epsv:
            result["epsv"] = self.epsv
        if alias_mu != 0.0:
            result["friction_coefficient"] = alias_mu
        if self.use_convergent_formulation:
            result["use_convergent_formulation"] = True
        if self.use_area_weighting != defaults.use_area_weighting:
            result["use_area_weighting"] = self.use_area_weighting
        if self.use_improved_max_operator != defaults.use_improved_max_operator:
            result["use_improved_max_operator"] = self.use_improved_max_operator
        if self.use_physical_barrier != defaults.use_physical_barrier:
            result["use_physical_barrier"] = self.use_physical_barrier
        if self.collision_mesh is not None:
            collision = _to_plain_value(self.collision_mesh)
            if collision:
                result["collision_mesh"] = collision
        if self.use_gcp_formulation:
            result["use_gcp_formulation"] = True
        if self.alpha_n != defaults.alpha_n:
            result["alpha_n"] = self.alpha_n
        if self.alpha_t != defaults.alpha_t:
            result["alpha_t"] = self.alpha_t
        if self.min_distance_ratio != defaults.min_distance_ratio:
            result["min_distance_ratio"] = self.min_distance_ratio
        if self.use_adaptive_dhat:
            result["use_adaptive_dhat"] = True
        if self.periodic:
            result["periodic"] = True
        if self.adhesion is not None:
            adhesion = _to_plain_value(self.adhesion)
            if adhesion:
                result["adhesion"] = adhesion
        if self.barrier_stiffness is not None:
            result["barrier_stiffness"] = _to_plain_value(self.barrier_stiffness)
        return result
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Contact":
        """Create Contact from dictionary (backward compatibility)."""
        collision_mesh = d.get("collision_mesh")
        if isinstance(collision_mesh, dict):
            collision_mesh = CollisionMesh.from_dict(collision_mesh)

        adhesion = d.get("adhesion")
        if isinstance(adhesion, dict):
            adhesion = Adhesion.from_dict(adhesion)

        return cls(
            enabled=bool(d.get("enabled", False)),
            dhat=float(d.get("dhat", 0.001)),
            dhat_percentage=float(d.get("dhat_percentage", 0.8)),
            epsv=float(d.get("epsv", 0.001)),
            friction_coefficient=float(d.get("friction_coefficient", d.get("mu", 0.0))),
            mu=d.get("mu"),
            use_convergent_formulation=bool(d.get("use_convergent_formulation", False)),
            use_area_weighting=bool(d.get("use_area_weighting", True)),
            use_improved_max_operator=bool(d.get("use_improved_max_operator", True)),
            use_physical_barrier=bool(d.get("use_physical_barrier", True)),
            collision_mesh=collision_mesh,
            use_gcp_formulation=bool(d.get("use_gcp_formulation", False)),
            alpha_n=float(d.get("alpha_n", 0.5)),
            alpha_t=float(d.get("alpha_t", 0.5)),
            min_distance_ratio=float(d.get("min_distance_ratio", 0.5)),
            use_adaptive_dhat=bool(d.get("use_adaptive_dhat", False)),
            periodic=bool(d.get("periodic", False)),
            adhesion=adhesion,
            barrier_stiffness=d.get("barrier_stiffness"),
        )


      
