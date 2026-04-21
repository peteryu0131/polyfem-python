from dataclasses import dataclass, field
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
            result["E"] = self.E
        if self.nu is not None:
            result["nu"] = self.nu
        if self.rho is not None:
            result["rho"] = self.rho
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
    """NeoHookean. Use (E, nu) or (lambda_, mu)."""
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
    
    def __post_init__(self):
        has_e_nu = self.E is not None and self.nu is not None
        has_lambda_mu = self.lambda_ is not None and self.mu is not None
        
        if not (has_e_nu or has_lambda_mu):
            raise ValueError("NeoHookean requires either (E, nu) or (lambda_, mu)")
        if has_e_nu and has_lambda_mu:
            raise ValueError("NeoHookean cannot have both (E, nu) and (lambda_, mu)")
    
    def to_dict(self) -> Dict[str, Any]:
        result = {"type": self.type}
        if self.E is not None and self.nu is not None:
            result["E"] = self.E
            result["nu"] = self.nu
        elif self.lambda_ is not None and self.mu is not None:
            result["lambda"] = self.lambda_
            result["mu"] = self.mu
        if self.id != 0:
            result["id"] = self.id
        if self.rho != 1:
            result["rho"] = self.rho
        if self.phi != 0:
            result["phi"] = self.phi
        if self.psi != 0:
            result["psi"] = self.psi
        return result


@dataclass
class IsochoricNeoHookean:
    """IsochoricNeoHookean. Use (E, nu) or (lambda_, mu)."""
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
    
    def __post_init__(self):
        has_e_nu = self.E is not None and self.nu is not None
        has_lambda_mu = self.lambda_ is not None and self.mu is not None
        
        if not (has_e_nu or has_lambda_mu):
            raise ValueError("IsochoricNeoHookean requires either (E, nu) or (lambda_, mu)")
        if has_e_nu and has_lambda_mu:
            raise ValueError("IsochoricNeoHookean cannot have both (E, nu) and (lambda_, mu)")
    
    def to_dict(self) -> Dict[str, Any]:
        result = {"type": self.type}
        if self.E is not None and self.nu is not None:
            result["E"] = self.E
            result["nu"] = self.nu
        elif self.lambda_ is not None and self.mu is not None:
            result["lambda"] = self.lambda_
            result["mu"] = self.mu
        if self.id != 0:
            result["id"] = self.id
        if self.rho != 1:
            result["rho"] = self.rho
        if self.phi != 0:
            result["phi"] = self.phi
        if self.psi != 0:
            result["psi"] = self.psi
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
    
    Example:
        >>> # E-nu input
        >>> material = LinearElasticity(E=2100, nu=0.3)
        >>> 
        >>> # lambda-mu input
        >>> material = LinearElasticity(lambda_=1000, mu=800)
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
    
    def __post_init__(self):
        has_e_nu = self.E is not None and self.nu is not None
        has_lambda_mu = self.lambda_ is not None and self.mu is not None
        
        if not (has_e_nu or has_lambda_mu):
            raise ValueError("LinearElasticity requires either (E, nu) or (lambda_, mu)")
        if has_e_nu and has_lambda_mu:
            raise ValueError("LinearElasticity cannot have both (E, nu) and (lambda_, mu)")
    
    def to_dict(self) -> Dict[str, Any]:
        result = {"type": self.type}
        if self.E is not None and self.nu is not None:
            result["E"] = self.E
            result["nu"] = self.nu
            if self.phi != 0:
                result["phi"] = self.phi
            if self.psi != 0:
                result["psi"] = self.psi
        elif self.lambda_ is not None and self.mu is not None:
            result["lambda"] = self.lambda_
            result["mu"] = self.mu
        if self.id != 0:
            result["id"] = self.id
        if self.rho != 1:
            result["rho"] = self.rho
        return result


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
    
    Example:
        >>> # E-nu input
        >>> material = HookeLinearElasticity(E=2100, nu=0.3)
        >>> 
        >>> # elasticity_tensor input
        >>> material = HookeLinearElasticity(elasticity_tensor=[...])
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
    
    def __post_init__(self):
        """Validate that either E-nu or elasticity_tensor is provided."""
        has_e_nu = self.E is not None and self.nu is not None
        has_tensor = self.elasticity_tensor is not None
        
        if not (has_e_nu or has_tensor):
            raise ValueError("HookeLinearElasticity requires either (E, nu) or elasticity_tensor")
        if has_e_nu and has_tensor:
            raise ValueError("HookeLinearElasticity cannot have both (E, nu) and elasticity_tensor")
    
    def to_dict(self) -> Dict[str, Any]:
        result = {"type": self.type}
        if self.E is not None and self.nu is not None:
            result["E"] = self.E
            result["nu"] = self.nu
        elif self.elasticity_tensor is not None:
            result["elasticity_tensor"] = self.elasticity_tensor
        if self.id != 0:
            result["id"] = self.id
        if self.rho != 1:
            result["rho"] = self.rho
        if self.fiber_direction != [0, 0, 0]:
            result["fiber_direction"] = self.fiber_direction
        return result


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
    
    Example:
        >>> # E-nu input
        >>> material = SaintVenant(E=2100, nu=0.3)
        >>> 
        >>> # elasticity_tensor input
        >>> material = SaintVenant(elasticity_tensor=[...])
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
    
    def __post_init__(self):
        """Validate that either E-nu or elasticity_tensor is provided."""
        has_e_nu = self.E is not None and self.nu is not None
        has_tensor = self.elasticity_tensor is not None
        
        if not (has_e_nu or has_tensor):
            raise ValueError("SaintVenant requires either (E, nu) or elasticity_tensor")
        if has_e_nu and has_tensor:
            raise ValueError("SaintVenant cannot have both (E, nu) and elasticity_tensor")
    
    def to_dict(self) -> Dict[str, Any]:
        result = {"type": self.type}
        if self.E is not None and self.nu is not None:
            result["E"] = self.E
            result["nu"] = self.nu
            if self.phi != 0:
                result["phi"] = self.phi
            if self.psi != 0:
                result["psi"] = self.psi
        elif self.elasticity_tensor is not None:
            result["elasticity_tensor"] = self.elasticity_tensor
            if self.phi != 0:
                result["phi"] = self.phi
            if self.psi != 0:
                result["psi"] = self.psi
        if self.id != 0:
            result["id"] = self.id
        if self.rho != 1:
            result["rho"] = self.rho
        if self.fiber_direction != [0, 0, 0]:
            result["fiber_direction"] = self.fiber_direction
        return result


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
    rhs: Optional[List[float]] = None
    pressure: Optional[List[Dict[str, Any]]] = None
    
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
            result["rhs"] = self.rhs
        
        if self.pressure is not None:
            result["pressure"] = self.pressure
        
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
        
        return bc


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


# Union type for all problem params
ProblemParams = Union[GravityParams, TorsionParams, FlowParams, FlowWithObstacleParams, Dict[str, Any]]


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
    solver: Optional[Union["Solver", Dict[str, Any]]] = None
    time: Optional[Union["Time", Dict[str, Any]]] = None
    output: Optional[Union["Output", Dict[str, Any]]] = None
    contact: Optional[Union["Contact", Dict[str, Any]]] = None
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
            solver=self.solver,  # Preserve solver
            time=self.time,  # Preserve time
            output=self.output,  # Preserve output
            contact=self.contact,  # Preserve contact
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
            solver=solver,
            time=time,
            output=output,
            contact=contact,
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
        for key in ("E", "nu"):
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
    
    # ---------------- Convenience methods for setting parameters ----------------
    
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
                # Try legacy Problems module
                try:
                    from polyfempy.legacy import Problems
                    problem_class = getattr(Problems, problem_name, None)
                except ImportError:
                    pass
            
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
class GeometryMesh:
    """Mesh geometry configuration - provides IDE autocomplete support.
    
    Attributes:
        mesh: Path to mesh file (required).
        volume_selection: Volume selection ID (optional).
        surface_selection: Surface selection ID (optional).
        transformation: Transformation matrix or parameters (optional).
        is_obstacle: Whether this mesh is an obstacle (optional).
    
    Example:
        >>> geom = GeometryMesh(mesh="mesh.obj", volume_selection=1)
        >>> obstacle = GeometryMesh(mesh="plane.obj", is_obstacle=True)
    """
    mesh: str = field()
    volume_selection: Optional[int] = None
    surface_selection: Optional[int] = None
    transformation: Optional[Dict[str, Any]] = None
    is_obstacle: Optional[bool] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {"mesh": self.mesh}
        if self.volume_selection is not None:
            result["volume_selection"] = self.volume_selection
        if self.surface_selection is not None:
            result["surface_selection"] = self.surface_selection
        if self.transformation is not None:
            result["transformation"] = self.transformation
        if self.is_obstacle is not None:
            result["is_obstacle"] = self.is_obstacle
        return result


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
    meshes: Union[List[GeometryMesh], List[str], GeometryMesh, str] = field(default_factory=list)
    transformations: Optional[List[Dict[str, Any]]] = None
    selections: Optional[Dict[str, Any]] = None

    def _normalized_mesh_list(self):
        if isinstance(self.meshes, (str, GeometryMesh)):
            return [self.meshes]
        return list(self.meshes)

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
        mesh_list = self._normalized_mesh_list()
        if not mesh_list:
            return []

        per_mesh_xform = self._resolve_per_mesh_transformations(len(mesh_list))

        result: List[Dict[str, Any]] = []
        for idx, mesh_item in enumerate(mesh_list):
            if isinstance(mesh_item, GeometryMesh):
                mesh_dict = mesh_item.to_dict()
            elif isinstance(mesh_item, str):
                mesh_dict = {"mesh": mesh_item}
            elif isinstance(mesh_item, dict):
                mesh_dict = dict(mesh_item)
            else:
                mesh_dict = {"mesh": str(mesh_item)}

            if per_mesh_xform is not None:
                top_xform = per_mesh_xform[idx]
                if top_xform is not None:
                    if "transformation" in mesh_dict and mesh_dict["transformation"] is not None:
                        raise ValueError(
                            f"meshes[{idx}] already has a transformation from "
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
            meshes = []
            for item in d:
                if isinstance(item, dict):
                    if "mesh" in item:
                        meshes.append(GeometryMesh(
                            mesh=item["mesh"],
                            volume_selection=item.get("volume_selection"),
                            surface_selection=item.get("surface_selection"),
                            transformation=item.get("transformation")
                        ))
                    else:
                        meshes.append(item)
                else:
                    meshes.append(item)
            return cls(meshes=meshes)
        elif isinstance(d, dict):
            return cls(meshes=[d])
        else:
            return cls(meshes=[])


# ============================================================================
# Solver Configuration Classes
# ============================================================================

@dataclass
class LinearSolver:
    """Linear solver configuration - provides IDE autocomplete support.
    
    Attributes:
        solver_type: Solver type (e.g., "Eigen::SparseLU", "Eigen::SimplicialLDLT", "Eigen::SparseQR").
        solver_priority: List of solver types to try in order (C++ tries until one is available).
        precond: Preconditioner type (optional).
        max_iterations: Maximum iterations (optional).
        tolerance: Tolerance (optional).
    
    Example:
        >>> solver = LinearSolver(solver_type="Eigen::SparseLU")
        >>> solver = LinearSolver(solver_priority=["Eigen::PardisoLDLT", "Eigen::SimplicialLDLT"])
    """
    solver_type: str = "Eigen::SparseLU"
    solver_priority: Optional[List[str]] = None  # If set, used instead of solver_type (list format)
    precond: Optional[str] = None
    max_iterations: Optional[int] = None
    tolerance: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format.
        
        Note: C++ backend expects "solver" key (not "solver_type") and can accept
        either a string or a list of strings (for solver priority).
        """
        # C++ backend expects "solver" key, and can accept list for priority
        solver_val = self.solver_priority if self.solver_priority is not None else self.solver_type
        result = {"solver": solver_val}
        if self.precond is not None:
            result["precond"] = self.precond
        if self.max_iterations is not None:
            result["max_iterations"] = self.max_iterations
        if self.tolerance is not None:
            result["tolerance"] = self.tolerance
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LinearSolver":
        """Create LinearSolver from dictionary (handles solver list for priority)."""
        ld = dict(d)
        solver_val = ld.pop("solver", ld.pop("solver_type", None))
        extra = {k: v for k, v in ld.items() if k in ("precond", "max_iterations", "tolerance")}
        if isinstance(solver_val, list):
            return cls(solver_priority=solver_val, **extra)
        if solver_val is not None:
            return cls(solver_type=solver_val, **extra)
        return cls(**extra)


@dataclass
class NonlinearSolver:
    """Nonlinear solver configuration - provides IDE autocomplete support.

    Attributes:
        solver_type: Solver type (e.g., "newton", "newton_armijo", "newton_ls", "Newton").
        max_iterations: Maximum iterations. Defaults to 100.
        tolerance: Tolerance. Defaults to 1e-6.
        grad_norm: Gradient norm tolerance (optional).
        x_delta: Solution delta tolerance (optional).
        iterations_per_strategy: Iterations per strategy (optional).
        line_search: Line search configuration (can be string or dict with method).
        method_blocks: PolyFEM JSON lets callers tune the active solver via a
            method-specific sub-dict whose key is the solver name, e.g.
            ``"Newton": {"residual_tolerance": 100}`` or ``"ADAM": {...}``.
            These blocks are preserved here so they round-trip through
            ``Solver.from_dict`` / ``Solver.to_dict`` without being silently
            filtered. Construct directly as
            ``NonlinearSolver(..., method_blocks={"Newton": {"residual_tolerance": 100}})``.

    Example:
        >>> solver = NonlinearSolver(solver_type="newton", max_iterations=50)
        >>> solver = NonlinearSolver(
        ...     solver_type="Newton",
        ...     method_blocks={"Newton": {"residual_tolerance": 100}},
        ... )
    """
    solver_type: str = "newton"
    max_iterations: int = 100
    tolerance: float = 1e-6
    grad_norm: Optional[float] = None
    x_delta: Optional[float] = None
    iterations_per_strategy: Optional[int] = None
    line_search: Optional[Union[str, Dict[str, Any]]] = None
    method_blocks: Optional[Dict[str, Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format. Minimal structure for C++ solver.

        Only include fields we explicitly set - omit max_iterations/tolerance when default,
        so C++ inject_defaults adds them. This avoids schema validation failures that
        occur when Python sends fields not in the root optional list.

        Any ``method_blocks`` entries (``Newton`` / ``ADAM`` / ``L-BFGS`` /
        ``L-BFGS-B`` / ``StochasticADAM`` / ``StochasticGradientDescent``) are
        written back at the ``nonlinear`` level — that's what the PolyFEM JSON
        schema expects for per-method tuning such as ``residual_tolerance``.
        """
        result: Dict[str, Any] = {"solver": self.solver_type}
        if self.max_iterations != 100:
            result["max_iterations"] = self.max_iterations
        if self.tolerance != 1e-6:
            result["tolerance"] = self.tolerance
        if self.grad_norm is not None:
            result["grad_norm"] = self.grad_norm
        if self.x_delta is not None:
            result["x_delta"] = self.x_delta
        if self.iterations_per_strategy is not None:
            result["iterations_per_strategy"] = self.iterations_per_strategy
        if self.line_search is not None:
            if isinstance(self.line_search, dict):
                # Schema oneOf fails when method-specific blocks (RobustArmijo, Armijo, etc.) are present.
                # Only include "method" and shared line_search fields; no method-specific blocks.
                line_search_cleaned = {}
                if "method" in self.line_search:
                    line_search_cleaned["method"] = self.line_search["method"]
                    # Do NOT include method-specific keys (RobustArmijo, Armijo, etc.) - they cause schema errors
                    for key in ["default_init_step_size", "max_step_size_iter", "max_step_size_iter_final",
                               "min_step_size", "min_step_size_final", "step_ratio", "use_grad_norm_tol"]:
                        if key in self.line_search:
                            line_search_cleaned[key] = self.line_search[key]
                else:
                    # If no method specified, copy as-is (backward compatibility)
                    line_search_cleaned = self.line_search
                result["line_search"] = line_search_cleaned
            else:
                result["line_search"] = {"method": self.line_search}

        # Per-method tuning blocks (Newton / ADAM / L-BFGS / ...). They must be
        # emitted at the ``nonlinear`` level — PolyFEM expects them right next
        # to ``solver`` / ``max_iterations`` etc., not nested anywhere deeper.
        if self.method_blocks:
            for block_name, block_value in self.method_blocks.items():
                if isinstance(block_value, dict):
                    result[block_name] = dict(block_value)
                else:
                    result[block_name] = block_value
        return result


@dataclass
class Solver:
    """Solver configuration - provides IDE autocomplete support.
    
    This class allows users to configure solver parameters with IDE autocomplete.
    
    Attributes:
        linear: Linear solver configuration (optional).
        nonlinear: Nonlinear solver configuration (optional).
        max_threads: Maximum threads. Defaults to 1.
        advanced: Advanced solver options (optional dict).
    
    Example:
        >>> solver = Solver(
        ...     linear=LinearSolver(solver_type="Eigen::SparseLU"),
        ...     nonlinear=NonlinearSolver(max_iterations=50)
        ... )
    """
    linear: Optional[LinearSolver] = None
    nonlinear: Optional[NonlinearSolver] = None
    max_threads: int = 1
    advanced: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format (for backend compatibility)."""
        result = {}
        if self.linear is not None:
            linear_dict = self.linear.to_dict()
            # C++ backend expects "solver" key in linear config
            # If we have "solver_type", convert it to "solver"
            if "solver_type" in linear_dict:
                linear_dict["solver"] = linear_dict.pop("solver_type")
            result["linear"] = linear_dict
        if self.nonlinear is not None:
            result["nonlinear"] = self.nonlinear.to_dict()
        if self.max_threads != 1:
            result["max_threads"] = self.max_threads
        if self.advanced is not None:
            # Support both: advanced as sub-object content {"lump_mass_matrix": ...}
            # or as from_dict style {"advanced": {...}}
            if "advanced" in self.advanced:
                result.update(self.advanced)
            else:
                result["advanced"] = self.advanced
        return result
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Solver":
        """Create Solver from dictionary (backward compatibility)."""
        linear = None
        if "linear" in d:
            linear = LinearSolver.from_dict(d["linear"]) if isinstance(d["linear"], dict) else d["linear"]
        
        nonlinear = None
        if "nonlinear" in d:
            if isinstance(d["nonlinear"], dict):
                # PolyFEM JSON lets users override per-method settings with a
                # sub-dict keyed by the solver name (e.g.
                # ``"Newton": {"residual_tolerance": 100}``). These blocks
                # used to be filtered out entirely to dodge schema errors,
                # which silently discarded the user's intent. Instead,
                # preserve them on ``NonlinearSolver.method_blocks`` so
                # ``to_dict()`` can reinstate them for the C++ solver.
                _METHOD_BLOCK_KEYS = (
                    "ADAM", "L-BFGS", "L-BFGS-B", "Newton",
                    "StochasticADAM", "StochasticGradientDescent",
                )
                raw_nl = dict(d["nonlinear"])
                method_blocks: Dict[str, Any] = {}
                for k in _METHOD_BLOCK_KEYS:
                    if k in raw_nl:
                        method_blocks[k] = raw_nl.pop(k)

                # Map JSON "solver" key to NonlinearSolver.solver_type
                if "solver" in raw_nl and "solver_type" not in raw_nl:
                    raw_nl["solver_type"] = raw_nl.pop("solver")

                if method_blocks:
                    raw_nl["method_blocks"] = method_blocks
                nonlinear = NonlinearSolver(**raw_nl)
            else:
                nonlinear = d["nonlinear"]
        
        return cls(
            linear=linear,
            nonlinear=nonlinear,
            max_threads=d.get("max_threads", 1),
            advanced={k: v for k, v in d.items() if k not in ["linear", "nonlinear", "max_threads"]}
        )


# ============================================================================
# Time Configuration Classes
# ============================================================================

@dataclass
class Time:
    """Time configuration for transient problems - provides IDE autocomplete support.
    
    Attributes:
        t0: Initial time. Defaults to 0.0.
        tend: End time (required).
        dt: Time step size (required).
        time_steps: Number of time steps (optional, alternative to dt).
        integrator: Time integrator type (e.g., "ImplicitEuler", "ImplicitNewmark").
                   Defaults to "ImplicitEuler".
    
    Example:
        >>> time = Time(t0=0.0, tend=1.0, dt=0.01, integrator="ImplicitEuler")
    """
    tend: float = field()
    dt: Optional[float] = None
    t0: float = 0.0
    time_steps: Optional[int] = None
    integrator: str = "ImplicitEuler"
    
    def __post_init__(self):
        """Validate that either dt or time_steps is provided."""
        if self.dt is None and self.time_steps is None:
            raise ValueError("Time requires either dt or time_steps")
        if self.dt is not None and self.time_steps is not None:
            raise ValueError("Time cannot have both dt and time_steps")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {
            "t0": self.t0,
            "tend": self.tend,
            "integrator": self.integrator,
        }
        if self.dt is not None:
            result["dt"] = self.dt
        if self.time_steps is not None:
            result["time_steps"] = self.time_steps
        return result
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Time":
        """Create Time from dictionary (backward compatibility)."""
        return cls(
            t0=d.get("t0", 0.0),
            tend=d.get("tend", 1.0),
            dt=d.get("dt"),
            time_steps=d.get("time_steps"),
            integrator=d.get("integrator", "ImplicitEuler")
        )


# ============================================================================
# Output Configuration Classes
# ============================================================================

@dataclass
class ParaviewOutput:
    """Paraview output configuration - provides IDE autocomplete support.
    
    Attributes:
        volume: Export volume data. Defaults to True.
        surface: Export surface data. Defaults to False.
        wireframe: Export wireframe. Defaults to False.
        points: Export points. Defaults to False.
        file_name: Output file name (e.g., "sim.pvd"). Defaults to None.
        options: Additional options (e.g., contact_forces, friction_forces, velocity, acceleration).
        vismesh_rel_area: Visualization mesh relative area. Defaults to None.
    
    Example:
        >>> paraview = ParaviewOutput(volume=True, surface=True, file_name="output.pvd")
    """
    volume: bool = True
    surface: bool = False
    wireframe: bool = False
    points: bool = False
    file_name: Optional[str] = None
    options: Optional[Dict[str, Any]] = None
    vismesh_rel_area: Optional[float] = None
    skip_frame: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {
            "volume": self.volume,
            "surface": self.surface,
            "wireframe": self.wireframe,
            "points": self.points,
        }
        if self.file_name is not None:
            result["file_name"] = self.file_name
        if self.options is not None:
            result["options"] = self.options
        if self.vismesh_rel_area is not None:
            result["vismesh_rel_area"] = self.vismesh_rel_area
        if self.skip_frame is not None:
            result["skip_frame"] = self.skip_frame
        return result


@dataclass
class ResultOutput:
    """Python-side result request for ``solve()``.

    This does **not** go into the PolyFEM JSON schema. It tells the Python API
    which result fields the user cares about and whether missing fields should
    be treated as an error.

    Attributes:
        fields: Requested result fields, e.g. ``["u", "stress", "von_mises"]``.
            ``None`` keeps legacy behavior and lets ``solve()`` return whatever
            is cheaply available.
        strict: If True, ``solve()`` raises when any requested field is still
            unavailable after native extraction and any configured fallbacks.
    """

    fields: Optional[List[str]] = None
    strict: bool = False

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        if self.fields is not None:
            result["fields"] = list(self.fields)
        if self.strict:
            result["strict"] = True
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ResultOutput":
        fields = d.get("fields")
        if fields is not None:
            fields = [str(x) for x in fields]
        return cls(fields=fields, strict=bool(d.get("strict", False)))


@dataclass
class FallbackOutput:
    """Python-side fallback policy for ``solve()`` result extraction.

    Attributes:
        sampled_vtu: Controls the temporary-VTU fallback used to recover
            sampled fields such as ``von_mises`` when the native result bundle
            does not provide them.
            - ``"never"``: never use temporary VTU fallback
            - ``"auto"``: use fallback only when requested fields need it
            - ``"always"``: always attempt fallback
        temp_storage: ``"ram"`` prefers ``/dev/shm`` when available, ``"disk"``
            uses the default temporary directory.
        keep_temp_files: If True, keep the temporary VTU directory for debugging
            instead of deleting it immediately.
    """

    sampled_vtu: str = "never"
    temp_storage: str = "ram"
    keep_temp_files: bool = False

    def __post_init__(self):
        sampled_vtu = str(self.sampled_vtu).strip().lower()
        if sampled_vtu not in ("never", "auto", "always"):
            raise ValueError(f"sampled_vtu must be one of never/auto/always, got {self.sampled_vtu!r}")
        self.sampled_vtu = sampled_vtu

        temp_storage = str(self.temp_storage).strip().lower()
        if temp_storage not in ("ram", "disk"):
            raise ValueError(f"temp_storage must be 'ram' or 'disk', got {self.temp_storage!r}")
        self.temp_storage = temp_storage

    def to_dict(self) -> Dict[str, Any]:
        result = {"sampled_vtu": self.sampled_vtu}
        if self.temp_storage != "ram":
            result["temp_storage"] = self.temp_storage
        if self.keep_temp_files:
            result["keep_temp_files"] = True
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FallbackOutput":
        return cls(
            sampled_vtu=str(d.get("sampled_vtu", "never")),
            temp_storage=str(d.get("temp_storage", "ram")),
            keep_temp_files=bool(d.get("keep_temp_files", False)),
        )


@dataclass
class Output:
    """Output configuration - provides IDE autocomplete support.
    
    Attributes:
        directory: Output directory. Defaults to "output".
        paraview: Paraview output configuration (optional).
        json: Export JSON results (can be bool or string filename). Defaults to True.
        log: Log configuration (level, etc.).
        advanced: Advanced output options (e.g., save_time_sequence, save_solve_sequence_debug).
        save_paraview: Python-side convenience switch. If False, ``to_dict()``
            disables Paraview sequence output without requiring the caller to
            manually touch ``advanced.save_time_sequence`` or clear
            ``paraview.file_name``.
        result: Python-side result request for ``solve()``.
        fallback: Python-side result fallback policy for ``solve()``.
    
    Example:
        >>> output = Output(directory="results", paraview=ParaviewOutput(volume=True))
    """
    directory: str = "output"
    paraview: Optional[ParaviewOutput] = None
    json: Union[bool, str] = True
    log: Optional[Dict[str, Any]] = None
    advanced: Optional[Dict[str, Any]] = None
    save_paraview: Optional[bool] = None
    result: Optional[Union[ResultOutput, Dict[str, Any]]] = None
    fallback: Optional[Union[FallbackOutput, Dict[str, Any]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format (for backend compatibility)."""
        result = {
            "directory": self.directory,
        }
        if isinstance(self.json, str):
            result["json"] = self.json
        elif self.json:
            result["json"] = True
        
        paraview_dict = self.paraview.to_dict() if self.paraview is not None else None
        advanced_dict = dict(self.advanced) if self.advanced is not None else None

        if self.save_paraview is False:
            if paraview_dict is None:
                paraview_dict = {}
            paraview_dict["file_name"] = ""
            if advanced_dict is None:
                advanced_dict = {}
            advanced_dict["save_time_sequence"] = False

        if paraview_dict is not None:
            result["paraview"] = paraview_dict
        if self.log is not None:
            result["log"] = self.log
        if advanced_dict is not None:
            result["advanced"] = advanced_dict
        return result

    def runtime_options(self) -> Dict[str, Any]:
        """Return Python-only runtime output controls for ``solve()``.

        These options are intentionally excluded from ``to_dict()`` because they
        are not part of the PolyFEM JSON schema.
        """
        result_cfg = None
        if isinstance(self.result, ResultOutput):
            result_cfg = self.result.to_dict()
        elif isinstance(self.result, dict):
            result_cfg = dict(self.result)

        fallback_cfg = None
        if isinstance(self.fallback, FallbackOutput):
            fallback_cfg = self.fallback.to_dict()
        elif isinstance(self.fallback, dict):
            fallback_cfg = dict(self.fallback)

        out: Dict[str, Any] = {}
        if result_cfg:
            out["result"] = result_cfg
        if fallback_cfg:
            out["fallback"] = fallback_cfg
        return out

    def request_results(self, fields: List[str], *, strict: bool = False) -> "Output":
        """Convenience helper for ``solve()`` result requests."""
        self.result = ResultOutput(fields=list(fields), strict=bool(strict))
        return self

    def configure_fallback(
        self,
        *,
        sampled_vtu: str = "auto",
        temp_storage: str = "ram",
        keep_temp_files: bool = False,
    ) -> "Output":
        """Convenience helper for ``solve()`` fallback behavior."""
        self.fallback = FallbackOutput(
            sampled_vtu=sampled_vtu,
            temp_storage=temp_storage,
            keep_temp_files=keep_temp_files,
        )
        return self
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Output":
        """Create Output from dictionary (backward compatibility)."""
        paraview = None
        if "paraview" in d:
            if isinstance(d["paraview"], dict):
                paraview = ParaviewOutput(**d["paraview"])
            else:
                paraview = d["paraview"]
        
        result_cfg = None
        if "result" in d and isinstance(d["result"], dict):
            result_cfg = ResultOutput.from_dict(d["result"])

        fallback_cfg = None
        if "fallback" in d and isinstance(d["fallback"], dict):
            fallback_cfg = FallbackOutput.from_dict(d["fallback"])

        return cls(
            directory=d.get("directory", "output"),
            paraview=paraview,
            json=d.get("json", True),
            log=d.get("log"),
            advanced=d.get("advanced"),
            save_paraview=d.get("save_paraview"),
            result=result_cfg,
            fallback=fallback_cfg,
        )


# ============================================================================
# Contact Configuration Classes
# ============================================================================

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
    dhat: float = 0.01
    mu: float = 0.0
    epsv: float = 0.0
    barrier_stiffness: float = 1e3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "enabled": self.enabled,
            "dhat": self.dhat,
            "mu": self.mu,
            "epsv": self.epsv,
            "barrier_stiffness": self.barrier_stiffness,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Contact":
        """Create Contact from dictionary (backward compatibility)."""
        return cls(
            enabled=d.get("enabled", False),
            dhat=d.get("dhat", 0.01),
            mu=d.get("mu", 0.0),
            epsv=d.get("epsv", 0.0),
            barrier_stiffness=d.get("barrier_stiffness", 1e3)
        )


      
