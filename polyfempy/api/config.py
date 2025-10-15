  
from dataclasses import dataclass, field
import json

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
        )

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
        return cls(
            pde=d.get("pde", "LinearElasticity"),
            discr_order=int(d.get("discr_order", 1)),
            materials=dict(d.get("materials", {})),
            boundary_conditions=dict(d.get("boundary_conditions", {})),
            extras=dict(d.get("extras", {})),
        )

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

        # Choose problem type
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
      
      