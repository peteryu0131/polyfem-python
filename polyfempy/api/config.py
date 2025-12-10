  
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING, Union, List, Dict, Any, overload
import json

if TYPE_CHECKING:
    from .selection import Selection

# Normalize PDE names to "Poisson" / "LinearElasticity"
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

# Normalize material parameter keys to "E" and "nu"
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

# Parameter promotion configuration: defines which extras parameters should be
# promoted to top level in to_dict(), and how to validate/convert them.
# 
# Format: {param_name: (validator_func, error_msg_template)}
# - validator_func: (value) -> converted_value, raises ValueError if invalid
# - error_msg_template: String template with {value} and {type_name} placeholders
#
# To add a new parameter:
# 1. Add an entry to this dictionary
# 2. Define a validator function that converts/validates the value
# 3. Provide an error message template
#
# Example:
#   "tolerance": (
#       lambda v: float(v) if float(v) > 0 else (_ for _ in ()).throw(ValueError("must be positive")),
#       "extras['tolerance'] must be a positive float, got {value!r} (type: {type_name})"
#   ),
def _validate_positive_int(v):
    """Validate and convert to positive integer."""
    v = int(v)
    if v <= 0:
        raise ValueError("must be positive")
    return v

def _validate_int_or_none(v):
    """Validate and convert to integer or None."""
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
    # Add new parameters here as needed:
    # Example: positive float
    # "tolerance": (
    #     lambda v: (v := float(v)) if v > 0 else (_ for _ in ()).throw(ValueError("must be positive")),
    #     "extras['tolerance'] must be a positive float, got {value!r} (type: {type_name})"
    # ),
    # Or use a function:
    # def _validate_positive_float(v):
    #     v = float(v)
    #     if v <= 0:
    #         raise ValueError("must be positive")
    #     return v
    # "tolerance": (
    #     _validate_positive_float,
    #     "extras['tolerance'] must be a positive float, got {value!r} (type: {type_name})"
    # ),
}


def _canon_pde(name: str) -> str:
    """Normalize a PDE name to 'Poisson' or 'LinearElasticity'.

    If the input is empty/None, defaults to 'LinearElasticity'.
    Aliases such as 'linear elasticity' or 'scalar' are mapped via `_PDE_ALIASES`.

    Args:
        name: Raw PDE name (case/spacing-insensitive).

    Returns:
        Normalized PDE name: 'Poisson' or 'LinearElasticity'.

    Notes:
        - The mapping is conservative: unknown names are returned as-is to allow forward compatibility.
    """
    if not name:
        return "LinearElasticity"
    key = name.replace(" ", "_").lower()
    return _PDE_ALIASES.get(key, name)


def _canon_materials(mat: dict) -> dict:
    """Normalize material dictionary keys (e.g., E/nu) without altering values.

    Aliases like 'youngs_modulus' and 'poisson_ratio' are mapped to 'E' and 'nu'.
    Unknown keys are preserved.

    Args:
        mat: Material parameters dictionary. May be None.

    Returns:
        A new dictionary with normalized keys and original values.

    Notes:
        - Only keys are normalized; values are kept intact.
        - Typical normalized keys include: 'E', 'nu'.
    """
    out = {}
    for k, v in (mat or {}).items():
        out[_MAT_ALIASES.get(k.lower(), k)] = v
    return out


# ============================================================================
# Material and Boundary Condition Classes (for better IDE support)
# ============================================================================

@dataclass
class Material:
    """Material parameters class - provides IDE autocomplete support.
    
    This class allows users to set material parameters with IDE autocomplete,
    instead of using dictionaries where IDE cannot suggest available keys.
    
    Attributes:
        E: Young's modulus (optional).
        nu: Poisson's ratio (optional).
        rho: Density (optional).
        type: Material type. Defaults to "LinearElasticity".
            Supported types include:
            - "LinearElasticity" (default)
            - "HookeLinearElasticity"
            - "SaintVenant"
            - "NeoHookean"
            - "MooneyRivlin"
            - "MooneyRivlin3Param"
            - "MooneyRivlin3ParamSymbolic"
            - "UnconstrainedOgden"
            - "IncompressibleOgden"
            - "IncompressibleLinearElasticity"
            - "Stokes"
            - "NavierStokes"
            - "OperatorSplitting"
            - "Laplacian"
            - "Helmholtz"
            - "Bilaplacian"
            - "AMIPS"
            - "FixedCorotational"
            And other material types supported by PolyFEM.
    
    Example:
        >>> # Linear elasticity (default)
        >>> material = Material(E=2100, nu=0.3)
        >>> cfg = SimulationConfig(materials=material)
        >>> 
        >>> # NeoHookean material
        >>> material = Material(E=2100, nu=0.3, type="NeoHookean")
        >>> 
        >>> # SaintVenant material
        >>> material = Material(E=2100, nu=0.3, type="SaintVenant")
        >>> 
        >>> # material.E  # IDE will autocomplete
        >>> # material.nu  # IDE will autocomplete
        >>> # material.type  # IDE will autocomplete
    """
    E: Optional[float] = None
    nu: Optional[float] = None
    rho: Optional[float] = None
    type: str = "LinearElasticity"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format (for backend compatibility).
        
        Returns:
            Dictionary with material parameters.
        """
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
        """Create Material from dictionary (backward compatibility).
        
        Handles aliases like 'youngs_modulus' -> 'E', 'poisson_ratio' -> 'nu'.
        
        Args:
            d: Dictionary with material parameters.
            
        Returns:
            Material instance.
        """
        # Handle aliases
        E = d.get("E") or d.get("e") or d.get("young") or d.get("youngs") or d.get("youngs_modulus") or d.get("young_modulus")
        nu = d.get("nu") or d.get("poisson") or d.get("poisson_ratio")
        return cls(
            E=E,
            nu=nu,
            rho=d.get("rho"),
            type=d.get("type", "LinearElasticity")
        )


# ============================================================================
# Specific Material Classes (with @overload support for multiple input types)
# ============================================================================

# Type alias for flexible parameter types
_ParamType = Union[float, str, Any]
_IdType = Union[int, List[int]]


@dataclass
class NeoHookean:
    """NeoHookean material - provides IDE autocomplete support.
    
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
        phi: First angle. Defaults to 0.
        psi: Second angle. Defaults to 0.
    
    Example:
        >>> # E-nu input
        >>> material = NeoHookean(E=2100, nu=0.3)
        >>> 
        >>> # lambda-mu input
        >>> material = NeoHookean(lambda_=1000, mu=800)
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
    
    def __post_init__(self):
        """Validate that either E-nu or lambda-mu is provided."""
        has_e_nu = self.E is not None and self.nu is not None
        has_lambda_mu = self.lambda_ is not None and self.mu is not None
        
        if not (has_e_nu or has_lambda_mu):
            raise ValueError("NeoHookean requires either (E, nu) or (lambda_, mu)")
        if has_e_nu and has_lambda_mu:
            raise ValueError("NeoHookean cannot have both (E, nu) and (lambda_, mu)")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
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
    """IsochoricNeoHookean material - provides IDE autocomplete support.
    
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
        phi: First angle. Defaults to 0.
        psi: Second angle. Defaults to 0.
    
    Example:
        >>> # E-nu input
        >>> material = IsochoricNeoHookean(E=2100, nu=0.3)
        >>> 
        >>> # lambda-mu input
        >>> material = IsochoricNeoHookean(lambda_=1000, mu=800)
    """
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
        """Validate that either E-nu or lambda-mu is provided."""
        has_e_nu = self.E is not None and self.nu is not None
        has_lambda_mu = self.lambda_ is not None and self.mu is not None
        
        if not (has_e_nu or has_lambda_mu):
            raise ValueError("IsochoricNeoHookean requires either (E, nu) or (lambda_, mu)")
        if has_e_nu and has_lambda_mu:
            raise ValueError("IsochoricNeoHookean cannot have both (E, nu) and (lambda_, mu)")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
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
    """MooneyRivlin material - provides IDE autocomplete support.
    
    Attributes:
        c1: First Mooney-Rivlin parameter (required).
        c2: Second Mooney-Rivlin parameter (required).
        k: Bulk modulus (required).
        id: Material ID or list of IDs. Defaults to 0.
        rho: Density. Defaults to 1.
    
    Example:
        >>> material = MooneyRivlin(c1=0.5, c2=0.1, k=1000)
    """
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
    """MooneyRivlin3Param material - provides IDE autocomplete support.
    
    Attributes:
        c1: First Mooney-Rivlin parameter (required).
        c2: Second Mooney-Rivlin parameter (required).
        c3: Third Mooney-Rivlin parameter (required).
        d1: First volumetric parameter (required).
        id: Material ID or list of IDs. Defaults to 0.
        rho: Density. Defaults to 1.
    
    Example:
        >>> material = MooneyRivlin3Param(c1=0.5, c2=0.1, c3=0.05, d1=1000)
    """
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
    """MooneyRivlin3ParamSymbolic material - provides IDE autocomplete support.
    
    Attributes:
        c1: First Mooney-Rivlin parameter (required).
        c2: Second Mooney-Rivlin parameter (required).
        c3: Third Mooney-Rivlin parameter (required).
        d1: First volumetric parameter (required).
        id: Material ID or list of IDs. Defaults to 0.
        rho: Density. Defaults to 1.
    
    Example:
        >>> material = MooneyRivlin3ParamSymbolic(c1=0.5, c2=0.1, c3=0.05, d1=1000)
    """
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
        """Validate that either E-nu or lambda-mu is provided."""
        has_e_nu = self.E is not None and self.nu is not None
        has_lambda_mu = self.lambda_ is not None and self.mu is not None
        
        if not (has_e_nu or has_lambda_mu):
            raise ValueError("LinearElasticity requires either (E, nu) or (lambda_, mu)")
        if has_e_nu and has_lambda_mu:
            raise ValueError("LinearElasticity cannot have both (E, nu) and (lambda_, mu)")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
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
        """Convert to dictionary format."""
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
        """Convert to dictionary format."""
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
        extras: Advanced options; passed through if the backend exposes such hooks.
        selection: Optional Selection object for geometric boundary selection.
        problem_type: Optional predefined problem type (e.g., 'Gravity', 'Franke', 'Torsion').
        problem_params: Optional parameters for predefined problems. Can be ProblemParams class
                       (GravityParams, TorsionParams, FlowParams, FlowWithObstacleParams) or dict.
                       Supports IDE autocomplete when using classes.

    Example:
        >>> # Using classes (recommended - IDE autocomplete)
        >>> material = Material(E=2100, nu=0.3)
        >>> bc = BoundaryConditions()
        >>> bc.add_dirichlet(id=4, value=[0.0, 0.0])
        >>> cfg = SimulationConfig(materials=material, boundary_conditions=bc)
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
    extras: dict = field(default_factory=dict)
    selection: Optional["Selection"] = None
    problem_type: Optional[str] = None
    problem_params: ProblemParams = field(default_factory=dict)

    # ---------------- Canonicalization ----------------

    def _get_materials_dict(self) -> Dict[str, Any]:
        """Internal helper: convert materials to dict format."""
        # Check if it's a material class instance (has to_dict method)
        if hasattr(self.materials, 'to_dict') and callable(getattr(self.materials, 'to_dict')):
            return self.materials.to_dict()
        elif isinstance(self.materials, list) and len(self.materials) > 0:
            # Handle list of materials (take first one for simple API)
            first = self.materials[0]
            if hasattr(first, 'to_dict') and callable(getattr(first, 'to_dict')):
                return first.to_dict()
            elif isinstance(first, dict):
                return _canon_materials(first)
            else:
                return {}
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
        # If we have full JSON config stored, return it
        if self.extras and "_full_json_config" in self.extras:
            return dict(self.extras["_full_json_config"])
        
        # Otherwise, construct from fields
        c = self.canonicalized()
        # Handle materials: if it's a material class, use to_dict(); otherwise use dict
        materials_dict = c._get_materials_dict() if hasattr(c, '_get_materials_dict') else (
            c.materials.to_dict() if hasattr(c.materials, 'to_dict') else (
                c.materials if isinstance(c.materials, dict) else dict(c.materials)
            )
        )
        result = {
            "pde": c.pde,
            "discr_order": c.discr_order,
            "materials": materials_dict,
            "boundary_conditions": c.boundary_conditions if isinstance(c.boundary_conditions, dict) else dict(c.boundary_conditions),
        }
        
        # Extract common solver parameters from extras to top level for backend compatibility
        if c.extras:
            # Copy extras but also promote common keys to top level
            result["extras"] = dict(c.extras)
            
            # Promote parameters according to _EXTRAS_PROMOTION_RULES
            for param_name, (validator, error_template) in _EXTRAS_PROMOTION_RULES.items():
                if param_name in c.extras:
                    value = c.extras[param_name]
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

    def to_json_str(self) -> str:
        """Serialize the canonical configuration to a compact JSON string.

        The object is first canonicalized to ensure normalized keys and values
        in the serialized output.

        Returns:
            A compact JSON string representing the canonical configuration.

        Example:
            >>> cfg = SimulationConfig.linear_elasticity(2100, 0.3)
            >>> cfg.to_json_str()
            '{"pde":"LinearElasticity","discr_order":1,"materials":{"E":2100,"nu":0.3},"boundary_conditions":{}}'
        """
        c = self.canonicalized()
        obj = {
            "pde": c.pde,
            "discr_order": c.discr_order,
            "materials": c.materials if isinstance(c.materials, dict) else c.materials,
            "boundary_conditions": c.boundary_conditions if isinstance(c.boundary_conditions, dict) else c.boundary_conditions,
        }
        if c.extras:
            obj["extras"] = c.extras
        return json.dumps(obj, separators=(",", ":"))

    @classmethod
    def from_json_str(cls, s: str) -> "SimulationConfig":
        """Deserialize a configuration from a JSON string.

        This expects the format produced by `to_json_str()`.

        Args:
            s: JSON string produced by `to_json_str()`.

        Returns:
            A `SimulationConfig` instance reconstructed from the JSON.

        Notes:
            - Unknown keys beyond the known fields are ignored.
            - Canonicalization is not performed here; call `canonicalized()` if needed.
        """
        d = json.loads(s)
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
        - common (external JSON references)
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
        
        # Extract materials - handle both dict and array formats
        materials = full_config.get("materials", {})
        materials_dict = {}
        if isinstance(materials, list) and len(materials) > 0:
            # Convert array format to dict (take first material for simple API)
            first_mat = materials[0]
            if isinstance(first_mat, dict):
                materials_dict = {k: v for k, v in first_mat.items() if k != "id"}
            else:
                materials_dict = {}
        elif isinstance(materials, dict):
            materials_dict = dict(materials)
        
        # Extract boundary_conditions - convert to BoundaryConditions if dict
        boundary_conditions_raw = full_config.get("boundary_conditions", {})
        if isinstance(boundary_conditions_raw, dict):
            boundary_conditions = BoundaryConditions.from_dict(boundary_conditions_raw)
        else:
            boundary_conditions = boundary_conditions_raw
        
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
        with open(filepath, "r") as f:
            config_dict = json.load(f)
        return cls.from_json_dict(config_dict)

    # ---------------- Validation ----------------

    def validate(self) -> None:
        """Perform lightweight sanity checks on key fields.

        Checks include:
        - `discr_order` must be a positive integer.
        - If present, `materials['E']` and `materials['nu']` must be numeric.

        Raises:
            ValueError: If any check fails.

        Notes:
            - Physics-range checks (e.g., 0 < nu < 0.5) can be added here if desired.
            - Call `canonicalized()` beforehand to normalize aliases.
        """
        if not isinstance(self.discr_order, int) or self.discr_order <= 0:
            raise ValueError(f"discr_order must be a positive integer, got {self.discr_order!r}")
        mats = self._get_materials_dict()
        for key in ("E", "nu"):
            if key in mats and not isinstance(mats[key], (int, float)):
                raise ValueError(f"materials['{key}'] must be a number, got {type(mats[key]).__name__}")
    
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
      
      