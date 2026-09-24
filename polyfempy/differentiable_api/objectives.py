"""Objective spec helpers for differentiable PolyFEM losses."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


_RESERVED_EXTRA_KEYS = {
    "type",
    "state",
    "volume_selection",
    "power",
    "weight",
    "print_energy",
}

_DEFAULT_WEIGHT = 1.0
_DEFAULT_PRINT_ENERGY = ""
_DEFAULT_STRESS_NORM_POWER = 2


class Objective(str, Enum):
    """Supported differentiable backend objective types."""

    MAX_STRESS = "max_stress"
    STRESS_NORM = "stress_norm"
    COMPLIANCE = "compliance"
    VOLUME = "volume"


@dataclass(frozen=True)
class ObjectiveSpec:
    """JSON-shaped objective description for future backend objective losses."""

    type_name: str
    state: Any = 0
    volume_selection: tuple[int, ...] | None = None
    power: Any = None
    weight: Any = _DEFAULT_WEIGHT
    print_energy: Any = _DEFAULT_PRINT_ENERGY
    extra: tuple[tuple[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Return a backend objective JSON payload copy."""

        payload: dict[str, Any] = {
            "type": self.type_name,
            "state": copy.deepcopy(self.state),
        }
        if self.volume_selection is not None:
            payload["volume_selection"] = list(self.volume_selection)
        if self.power is not None:
            payload["power"] = copy.deepcopy(self.power)
        if self.weight is not None:
            payload["weight"] = copy.deepcopy(self.weight)
        if self.print_energy is not None:
            payload["print_energy"] = copy.deepcopy(self.print_energy)
        for key, value in self.extra:
            payload[key] = copy.deepcopy(value)
        return payload


class ObjectiveNamespace:
    """Factory namespace for differentiable objective specs."""

    def stress_norm(
        self,
        *,
        selection: Any = None,
        state: Any = 0,
        power: Any = _DEFAULT_STRESS_NORM_POWER,
        weight: Any = None,
        print_energy: Any = None,
        extra: Mapping[str, Any] | None = None,
    ) -> ObjectiveSpec:
        """Create a stress_norm objective spec."""

        return _objective_spec(
            "stress_norm",
            selection=selection,
            state=state,
            power=power,
            weight=weight,
            print_energy=print_energy,
            extra=extra,
        )

    def max_stress(
        self,
        *,
        selection: Any = None,
        state: Any = 0,
        weight: Any = None,
        print_energy: Any = None,
        extra: Mapping[str, Any] | None = None,
    ) -> ObjectiveSpec:
        """Create a max_stress objective spec."""

        return _objective_spec(
            "max_stress",
            selection=selection,
            state=state,
            weight=weight,
            print_energy=print_energy,
            extra=extra,
        )

    def compliance(
        self,
        *,
        selection: Any = None,
        state: Any = 0,
        weight: Any = None,
        print_energy: Any = None,
        extra: Mapping[str, Any] | None = None,
    ) -> ObjectiveSpec:
        """Create a compliance objective spec."""

        return _objective_spec(
            "compliance",
            selection=selection,
            state=state,
            weight=weight,
            print_energy=print_energy,
            extra=extra,
        )

    def volume(
        self,
        *,
        selection: Any = None,
        state: Any = 0,
        weight: Any = None,
        print_energy: Any = None,
        extra: Mapping[str, Any] | None = None,
    ) -> ObjectiveSpec:
        """Create a volume objective spec."""

        return _objective_spec(
            "volume",
            selection=selection,
            state=state,
            weight=weight,
            print_energy=print_energy,
            extra=extra,
        )


def _objective_spec(
    type_name: str,
    *,
    selection: Any,
    state: Any,
    power: Any = None,
    weight: Any = None,
    print_energy: Any = None,
    extra: Mapping[str, Any] | None = None,
) -> ObjectiveSpec:
    return ObjectiveSpec(
        type_name=type_name,
        state=copy.deepcopy(state),
        volume_selection=_volume_selection(selection),
        power=copy.deepcopy(power),
        weight=_defaulted(weight, _DEFAULT_WEIGHT),
        print_energy=_defaulted(print_energy, _DEFAULT_PRINT_ENERGY),
        extra=_extra_items(extra),
    )


def _volume_selection(selection: Any) -> tuple[int, ...] | None:
    if selection is None:
        return None
    if isinstance(selection, int):
        return (_positive_selection_id(selection),)
    if isinstance(selection, (str, bytes)):
        raise TypeError("backend selection id must be an int or model selection handle")

    try:
        values = tuple(selection)
    except TypeError:
        return (_selection_id(selection),)

    return tuple(_selection_id(value) for value in values)


def _selection_id(value: Any) -> int:
    if isinstance(value, int):
        return _positive_selection_id(value)

    direct_id = getattr(value, "backend_id", None)
    if direct_id is not None:
        return _positive_selection_id(direct_id)

    selection_ref = getattr(value, "selection_ref", None)
    if selection_ref is not None:
        return _selection_id(selection_ref)

    volume_ref = getattr(value, "volume_ref", None)
    if volume_ref is not None:
        return _selection_id(volume_ref)

    raise TypeError("backend selection id must be an int or model selection handle")


def _positive_selection_id(value: Any) -> int:
    if not isinstance(value, int):
        raise TypeError("backend selection id must be an int")
    if value < 1:
        raise ValueError("backend selection id must be positive")
    return value


def _extra_items(extra: Mapping[str, Any] | None) -> tuple[tuple[str, Any], ...]:
    if extra is None:
        return ()
    if not isinstance(extra, Mapping):
        raise TypeError("extra must be a mapping")

    overlap = sorted(set(extra) & _RESERVED_EXTRA_KEYS)
    if overlap:
        joined = ", ".join(overlap)
        raise ValueError(f"extra cannot override objective field(s): {joined}")

    return tuple((str(key), copy.deepcopy(value)) for key, value in extra.items())


def _defaulted(value: Any, default: Any) -> Any:
    if value is None:
        return copy.deepcopy(default)
    return copy.deepcopy(value)


objectives = ObjectiveNamespace()


def _build_objective(
    objective: Objective,
    *,
    params: Mapping[str, Any] | None = None,
    default_selection: Any = None,
) -> ObjectiveSpec:
    """Build a backend objective spec from the public objective enum."""

    kind = _objective_kind(objective)
    objective_params = _objective_params(params, default_selection)
    builders = {
        Objective.MAX_STRESS: objectives.max_stress,
        Objective.STRESS_NORM: objectives.stress_norm,
        Objective.COMPLIANCE: objectives.compliance,
        Objective.VOLUME: objectives.volume,
    }
    return builders[kind](**objective_params)


def _objective_kind(objective: Objective) -> Objective:
    if isinstance(objective, Objective):
        return objective
    raise TypeError("objective must be a diff.Objective enum value")


def _objective_params(
    params: Mapping[str, Any] | None,
    default_selection: Any,
) -> dict[str, Any]:
    if params is None:
        objective_params: dict[str, Any] = {}
    elif isinstance(params, Mapping):
        objective_params = copy.deepcopy(dict(params))
    else:
        raise TypeError("objective_params must be a mapping")

    if "selection" not in objective_params and default_selection is not None:
        if default_selection == "all":
            objective_params["selection"] = []
        elif not isinstance(default_selection, (str, bytes)):
            objective_params["selection"] = default_selection
    return objective_params


def MaxStress(
    *,
    selection: Any = None,
    state: Any = 0,
    weight: Any = None,
    print_energy: Any = None,
    extra: Mapping[str, Any] | None = None,
) -> ObjectiveSpec:
    """Create a max_stress objective using the meeting-facing API name."""

    return objectives.max_stress(
        selection=selection,
        state=state,
        weight=weight,
        print_energy=print_energy,
        extra=extra,
    )


def StressNorm(
    *,
    selection: Any = None,
    state: Any = 0,
    power: Any = _DEFAULT_STRESS_NORM_POWER,
    weight: Any = None,
    print_energy: Any = None,
    extra: Mapping[str, Any] | None = None,
) -> ObjectiveSpec:
    """Create a stress_norm objective using the meeting-facing API name."""

    return objectives.stress_norm(
        selection=selection,
        state=state,
        power=power,
        weight=weight,
        print_energy=print_energy,
        extra=extra,
    )


__all__ = [
    "Objective",
    "MaxStress",
    "ObjectiveNamespace",
    "ObjectiveSpec",
    "StressNorm",
    "objectives",
]
