"""Solver-related typed configuration blocks.

This module keeps linear/nonlinear solver configuration separate from the larger
``polyfempy.api.config`` facade. ``config.py`` re-exports these names for
backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union


def _to_plain_value(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(getattr(value, "to_dict")):
        return value.to_dict()
    if isinstance(value, dict):
        return {k: _to_plain_value(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_to_plain_value(v) for v in value]
    if isinstance(value, list):
        return [_to_plain_value(v) for v in value]
    return value


# ============================================================================
# Solver Configuration Classes
# ============================================================================

@dataclass
class LineSearch:
    """Line-search settings for nonlinear solvers."""

    method: str = "RobustArmijo"
    default_init_step_size: Optional[float] = None
    max_step_size_iter: Optional[int] = None
    max_step_size_iter_final: Optional[int] = None
    min_step_size: Optional[float] = None
    min_step_size_final: Optional[float] = None
    step_ratio: Optional[float] = None
    use_grad_norm_tol: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"method": self.method}
        for key in (
            "default_init_step_size",
            "max_step_size_iter",
            "max_step_size_iter_final",
            "min_step_size",
            "min_step_size_final",
            "step_ratio",
            "use_grad_norm_tol",
        ):
            value = getattr(self, key)
            if value is not None:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LineSearch":
        kwargs = {key: d.get(key) for key in cls.__dataclass_fields__ if key != "method"}
        return cls(method=str(d.get("method", "RobustArmijo")), **kwargs)


@dataclass
class AugmentedLagrangian:
    """Augmented Lagrangian options for Dirichlet/contact handling."""

    initial_weight: float = 1e6
    scaling: float = 2.0
    max_weight: Optional[float] = None
    eta: Optional[float] = None
    error: float = 1e-2

    def to_dict(self) -> Dict[str, Any]:
        defaults = type(self)()
        result: Dict[str, Any] = {}
        for key in self.__dataclass_fields__:
            value = getattr(self, key)
            if value is not None and value != getattr(defaults, key):
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AugmentedLagrangian":
        defaults = cls()
        kwargs = {key: d.get(key, getattr(defaults, key)) for key in cls.__dataclass_fields__}
        return cls(**kwargs)


@dataclass
class SolverContactOptions:
    """Contact-related solver inner-loop settings."""

    CCD: Optional[Any] = None
    friction_iterations: Optional[int] = None
    tangential_adhesion_iterations: Optional[int] = None
    friction_convergence_tol: Optional[float] = None
    barrier_stiffness: Optional[Any] = None
    initial_barrier_stiffness: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key in self.__dataclass_fields__:
            value = getattr(self, key)
            if value is not None:
                result[key] = _to_plain_value(value)
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SolverContactOptions":
        kwargs = {key: d.get(key) for key in cls.__dataclass_fields__}
        return cls(**kwargs)


@dataclass
class RayleighDamping:
    """Rayleigh damping coefficients."""

    mass: Optional[float] = None
    stiffness: Optional[float] = None
    lagging_iterations: int = 1

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        if self.mass is not None:
            result["mass"] = self.mass
        if self.stiffness is not None:
            result["stiffness"] = self.stiffness
        if self.lagging_iterations != 1:
            result["lagging_iterations"] = self.lagging_iterations
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RayleighDamping":
        return cls(
            mass=d.get("mass"),
            stiffness=d.get("stiffness"),
            lagging_iterations=int(d.get("lagging_iterations", 1)),
        )


@dataclass
class SolverAdvanced:
    """Advanced solver settings."""

    cache_size: int = 900000
    lump_mass_matrix: bool = False
    lagged_regularization_weight: float = 0.0
    lagged_regularization_iterations: int = 1
    check_inversion: str = "Discrete"
    jacobian_threshold: float = 0.0
    characteristic_length: float = -1.0
    characteristic_force_density: float = 10000.0

    def to_dict(self) -> Dict[str, Any]:
        defaults = type(self)()
        result: Dict[str, Any] = {}
        for key in self.__dataclass_fields__:
            value = getattr(self, key)
            if value != getattr(defaults, key):
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SolverAdvanced":
        defaults = cls()
        kwargs = {key: d.get(key, getattr(defaults, key)) for key in cls.__dataclass_fields__}
        return cls(**kwargs)


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
    def pardiso_ldlt(
        cls,
        *,
        precond: Optional[str] = None,
        max_iterations: Optional[int] = None,
        tolerance: Optional[float] = None,
    ) -> "LinearSolver":
        """Construct the common Pardiso LDLT linear-solver setup."""
        return cls(
            solver_type="Eigen::PardisoLDLT",
            precond=precond,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )

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
    line_search: Optional[Union[str, LineSearch, Dict[str, Any]]] = None
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
            if isinstance(self.line_search, LineSearch):
                result["line_search"] = self.line_search.to_dict()
            elif isinstance(self.line_search, dict):
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

    @classmethod
    def newton(
        cls,
        *,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        grad_norm: Optional[float] = None,
        x_delta: Optional[float] = None,
        iterations_per_strategy: Optional[int] = None,
        line_search: Optional[Union[str, LineSearch, Dict[str, Any]]] = None,
        residual_tolerance: Optional[float] = None,
    ) -> "NonlinearSolver":
        """Construct the common Newton nonlinear-solver setup."""
        method_blocks = None
        if residual_tolerance is not None:
            method_blocks = {"Newton": {"residual_tolerance": residual_tolerance}}
        return cls(
            solver_type="Newton",
            max_iterations=max_iterations,
            tolerance=tolerance,
            grad_norm=grad_norm,
            x_delta=x_delta,
            iterations_per_strategy=iterations_per_strategy,
            line_search=line_search,
            method_blocks=method_blocks,
        )


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
    augmented_lagrangian: Optional[Union[AugmentedLagrangian, Dict[str, Any]]] = None
    contact: Optional[Union[SolverContactOptions, Dict[str, Any]]] = None
    rayleigh_damping: Optional[Union[RayleighDamping, Dict[str, Any]]] = None
    advanced: Optional[Union[SolverAdvanced, Dict[str, Any]]] = None
    
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
        if self.augmented_lagrangian is not None:
            al_dict = _to_plain_value(self.augmented_lagrangian)
            if al_dict:
                result["augmented_lagrangian"] = al_dict
        if self.contact is not None:
            contact_dict = _to_plain_value(self.contact)
            if contact_dict:
                result["contact"] = contact_dict
        if self.rayleigh_damping is not None:
            damping_dict = _to_plain_value(self.rayleigh_damping)
            if damping_dict:
                result["rayleigh_damping"] = damping_dict
        if self.advanced is not None:
            advanced_dict = _to_plain_value(self.advanced)
            if isinstance(advanced_dict, dict) and "advanced" in advanced_dict:
                result.update(advanced_dict)
            elif advanced_dict:
                result["advanced"] = advanced_dict
        return result

    @classmethod
    def newton_contact(
        cls,
        *,
        linear: Optional[LinearSolver] = None,
        linear_solver: str = "Eigen::PardisoLDLT",
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        grad_norm: Optional[float] = None,
        x_delta: Optional[float] = None,
        iterations_per_strategy: Optional[int] = None,
        line_search: Optional[Union[str, LineSearch, Dict[str, Any]]] = None,
        residual_tolerance: Optional[float] = None,
        barrier_stiffness: Optional[Any] = None,
        max_threads: int = 1,
        augmented_lagrangian: Optional[Union[AugmentedLagrangian, Dict[str, Any]]] = None,
        rayleigh_damping: Optional[Union[RayleighDamping, Dict[str, Any]]] = None,
        advanced: Optional[Union[SolverAdvanced, Dict[str, Any]]] = None,
    ) -> "Solver":
        """Construct the common Newton + contact solver stack in one call."""
        linear_cfg = linear
        if linear_cfg is None:
            linear_cfg = LinearSolver(solver_type=linear_solver)
        return cls(
            linear=linear_cfg,
            nonlinear=NonlinearSolver.newton(
                max_iterations=max_iterations,
                tolerance=tolerance,
                grad_norm=grad_norm,
                x_delta=x_delta,
                iterations_per_strategy=iterations_per_strategy,
                line_search=line_search,
                residual_tolerance=residual_tolerance,
            ),
            max_threads=max_threads,
            augmented_lagrangian=augmented_lagrangian,
            contact=SolverContactOptions(barrier_stiffness=barrier_stiffness),
            rayleigh_damping=rayleigh_damping,
            advanced=advanced,
        )
    
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
                if "line_search" in raw_nl and isinstance(raw_nl["line_search"], dict):
                    raw_nl["line_search"] = LineSearch.from_dict(raw_nl["line_search"])

                if method_blocks:
                    raw_nl["method_blocks"] = method_blocks
                nonlinear = NonlinearSolver(**raw_nl)
            else:
                nonlinear = d["nonlinear"]

        augmented_lagrangian = None
        if "augmented_lagrangian" in d:
            if isinstance(d["augmented_lagrangian"], dict):
                augmented_lagrangian = AugmentedLagrangian.from_dict(d["augmented_lagrangian"])
            else:
                augmented_lagrangian = d["augmented_lagrangian"]

        solver_contact = None
        if "contact" in d:
            if isinstance(d["contact"], dict):
                solver_contact = SolverContactOptions.from_dict(d["contact"])
            else:
                solver_contact = d["contact"]

        rayleigh_damping = None
        if "rayleigh_damping" in d:
            if isinstance(d["rayleigh_damping"], dict):
                rayleigh_damping = RayleighDamping.from_dict(d["rayleigh_damping"])
            else:
                rayleigh_damping = d["rayleigh_damping"]

        advanced = None
        if "advanced" in d and isinstance(d["advanced"], dict):
            advanced = SolverAdvanced.from_dict(d["advanced"])
        else:
            advanced = {
                k: v for k, v in d.items()
                if k not in [
                    "linear", "nonlinear", "max_threads",
                    "augmented_lagrangian", "contact", "rayleigh_damping"
                ]
            }
            if not advanced:
                advanced = None
        
        return cls(
            linear=linear,
            nonlinear=nonlinear,
            max_threads=d.get("max_threads", 1),
            augmented_lagrangian=augmented_lagrangian,
            contact=solver_contact,
            rayleigh_damping=rayleigh_damping,
            advanced=advanced,
        )


__all__ = [
    "LineSearch",
    "AugmentedLagrangian",
    "SolverContactOptions",
    "RayleighDamping",
    "SolverAdvanced",
    "LinearSolver",
    "NonlinearSolver",
    "Solver",
]
