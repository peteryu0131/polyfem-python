  
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING
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


@dataclass
class SimulationConfig:
    """Human-friendly configuration → canonical form → PolyFEM Settings/Problem.

    This class stores intuitive configuration fields, provides normalization and
    lightweight validation, and (optionally) constructs a backend Settings/Problem.

    Attributes:
        pde: PDE name. Aliases auto-normalized to 'Poisson' or 'LinearElasticity'.
        discr_order: Polynomial order of the discretization (1, 2, ...).
        materials: Material parameters (e.g., {'E': 2100, 'nu': 0.3}). Aliases accepted.
        boundary_conditions: High-level BC container; applied later in `solve()`.
        extras: Advanced options; passed through if the backend exposes such hooks.
        selection: Optional Selection object for geometric boundary selection.
        problem_type: Optional predefined problem type (e.g., 'Gravity', 'Franke', 'Torsion').
        problem_params: Optional parameters for predefined problems (e.g., {'force': 0.1} for Gravity).

    Example:
        >>> cfg = SimulationConfig(pde="LinearElasticity",
        ...                        discr_order=1,
        ...                        materials={"E": 2100, "nu": 0.3})
        >>> s = cfg.to_settings()  # may raise if backend is missing
    """

    pde: str = "LinearElasticity"
    discr_order: int = 1
    materials: dict = field(default_factory=dict)
    boundary_conditions: dict = field(default_factory=dict)
    extras: dict = field(default_factory=dict)
    selection: Optional["Selection"] = None
    problem_type: Optional[str] = None
    problem_params: dict = field(default_factory=dict)

    # ---------------- Canonicalization ----------------

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
            materials=_canon_materials(self.materials),
            boundary_conditions=dict(self.boundary_conditions or {}),
            extras=dict(self.extras or {}),
            selection=self.selection,  # Selection objects are not copied
            problem_type=self.problem_type,
            problem_params=dict(self.problem_params or {}),
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
        result = {
            "pde": c.pde,
            "discr_order": c.discr_order,
            "materials": dict(c.materials),
            "boundary_conditions": dict(c.boundary_conditions),
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
        if c.problem_params:
            result["problem_params"] = dict(c.problem_params)
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
            "materials": c.materials,
            "boundary_conditions": c.boundary_conditions,
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
            materials_dict = {k: v for k, v in first_mat.items() if k != "id" and k != "type"}
        elif isinstance(materials, dict):
            materials_dict = dict(materials)
        
        # Extract boundary_conditions
        boundary_conditions = dict(full_config.get("boundary_conditions", {}))
        
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
        mats = _canon_materials(self.materials)
        for key in ("E", "nu"):
            if key in mats and not isinstance(mats[key], (int, float)):
                raise ValueError(f"materials['{key}'] must be a number, got {type(mats[key]).__name__}")

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
            if c.problem_params:
                problem = problem_class(**c.problem_params)
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
        mats = _canon_materials(c.materials)
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
            problem_params={"force": force},
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
            problem_params={
                "axis_coordinate": axis_coordinate,  # Legacy handles both spellings
                "n_turns": n_turns,
                "fixed_boundary": fixed_boundary,
                "turning_boundary": turning_boundary,
            },
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
            problem_params={
                "inflow": inflow,
                "outflow": outflow,
                "inflow_amout": inflow_amount,  # Note: typo in original API
                "outflow_amout": outflow_amount,  # Note: typo in original API
                "direction": direction,
                "obstacle": obstacle,
            },
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
            problem_params={"U": U, "time_dependent": time_dependent},
        )
      
      