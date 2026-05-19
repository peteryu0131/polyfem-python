"""Time-related typed configuration blocks.

This module keeps transient/time-step configuration separate from the larger
``polyfempy.api.config`` facade. ``config.py`` re-exports these names for
backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union


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
# Time Configuration Classes
# ============================================================================

@dataclass
class BDFIntegrator:
    """Backwards differentiation formula integrator options."""

    type: str = "BDF"
    steps: int = 1

    def to_dict(self) -> Dict[str, Any]:
        result = {"type": self.type}
        if self.steps != 1:
            result["steps"] = self.steps
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BDFIntegrator":
        return cls(type=str(d.get("type", "BDF")), steps=int(d.get("steps", 1)))


@dataclass
class ImplicitNewmarkIntegrator:
    """Implicit Newmark integrator options."""

    type: str = "ImplicitNewmark"
    gamma: float = 0.5
    beta: float = 0.25

    def to_dict(self) -> Dict[str, Any]:
        result = {"type": self.type}
        if self.gamma != 0.5:
            result["gamma"] = self.gamma
        if self.beta != 0.25:
            result["beta"] = self.beta
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ImplicitNewmarkIntegrator":
        return cls(
            type=str(d.get("type", "ImplicitNewmark")),
            gamma=float(d.get("gamma", 0.5)),
            beta=float(d.get("beta", 0.25)),
        )


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
    tend: Optional[float] = None
    dt: Optional[float] = None
    t0: float = 0.0
    time_steps: Optional[int] = None
    integrator: Union[str, BDFIntegrator, ImplicitNewmarkIntegrator, Dict[str, Any]] = "ImplicitEuler"
    quasistatic: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {"t0": self.t0}
        if self.tend is not None:
            result["tend"] = self.tend
        if self.dt is not None:
            result["dt"] = self.dt
        if self.time_steps is not None:
            result["time_steps"] = self.time_steps
        if isinstance(self.integrator, (BDFIntegrator, ImplicitNewmarkIntegrator)):
            result["integrator"] = self.integrator.to_dict()
        else:
            result["integrator"] = _to_plain_value(self.integrator)
        if self.quasistatic:
            result["quasistatic"] = True
        return result

    @classmethod
    def transient(
        cls,
        *,
        tend: float,
        dt: Optional[float] = None,
        time_steps: Optional[int] = None,
        t0: float = 0.0,
        integrator: Union[str, BDFIntegrator, ImplicitNewmarkIntegrator, Dict[str, Any]] = "ImplicitEuler",
        quasistatic: bool = False,
    ) -> "Time":
        """Construct the common transient-time block."""
        return cls(
            t0=t0,
            tend=tend,
            dt=dt,
            time_steps=time_steps,
            integrator=integrator,
            quasistatic=quasistatic,
        )

    @classmethod
    def bdf(
        cls,
        *,
        tend: float,
        dt: Optional[float] = None,
        time_steps: Optional[int] = None,
        t0: float = 0.0,
        steps: int = 1,
        quasistatic: bool = False,
    ) -> "Time":
        """Construct a transient time block with a BDF integrator."""
        return cls.transient(
            t0=t0,
            tend=tend,
            dt=dt,
            time_steps=time_steps,
            integrator=BDFIntegrator(steps=steps),
            quasistatic=quasistatic,
        )

    @classmethod
    def implicit_newmark(
        cls,
        *,
        tend: float,
        dt: Optional[float] = None,
        time_steps: Optional[int] = None,
        t0: float = 0.0,
        gamma: float = 0.5,
        beta: float = 0.25,
        quasistatic: bool = False,
    ) -> "Time":
        """Construct a transient time block with an implicit Newmark integrator."""
        return cls.transient(
            t0=t0,
            tend=tend,
            dt=dt,
            time_steps=time_steps,
            integrator=ImplicitNewmarkIntegrator(gamma=gamma, beta=beta),
            quasistatic=quasistatic,
        )
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Time":
        """Create Time from dictionary (backward compatibility)."""
        integrator = d.get("integrator", "ImplicitEuler")
        if isinstance(integrator, dict):
            type_name = str(integrator.get("type", "")).strip()
            if type_name == "BDF":
                integrator = BDFIntegrator.from_dict(integrator)
            elif type_name == "ImplicitNewmark":
                integrator = ImplicitNewmarkIntegrator.from_dict(integrator)
        return cls(
            t0=d.get("t0", 0.0),
            tend=d.get("tend"),
            dt=d.get("dt"),
            time_steps=d.get("time_steps"),
            integrator=integrator,
            quasistatic=bool(d.get("quasistatic", False)),
        )


__all__ = [
    "BDFIntegrator",
    "ImplicitNewmarkIntegrator",
    "Time",
]
