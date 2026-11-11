"""Material differentiable objective wrapper."""

from __future__ import annotations

from collections.abc import Mapping
import json
import math
from numbers import Real
from typing import Any

from . import _backend
from .model import DifferentiableModel, ModelMetadata
from .shape import _load_default_backend, _objective_payload


class MaterialOptimization:
    """Material objective with optional diagnostics after each evaluation.

    ``diagnostics`` contains native peak time/location for von_mises_peak,
    obtained from the existing solve rather than another simulation.
    """

    def __init__(
        self,
        model: DifferentiableModel,
        lame: Any = None,
        *,
        materials: Mapping | None = None,
        objective: Any,
        objective_params: Mapping[str, Any] | None = None,
        backend: Any | None = None,
    ) -> None:
        if objective is None:
            raise ValueError("objective must not be None")
        if materials is not None and lame is not None:
            raise ValueError("Provide either lame or materials, not both")
        payload = _single_material_payload(model=model, lame=materials if materials is not None else lame)
        self.model = model
        self.lame = lame
        self.objective = objective
        self.objective_params = (
            None if objective_params is None else dict(objective_params)
        )
        self.backend = backend
        self.diagnostics: dict[str, Any] = {}
        self._enu = None if materials is None else _ENuMaterials(payload, materials, backend)
        self.parameters = (lame,) if self._enu is None else self._enu.parameters

    def optimizer(self, optimizer: Any, **kwargs: Any) -> Any:
        """Optimize trainable inputs with the shared loop (E then nu per body)."""
        from .optimization import optimization

        return optimization(self, parameters=self.parameters).optimizer(optimizer, **kwargs)

    def __call__(self) -> Any:
        """Return one differentiable material objective evaluation."""

        from .torch_ops import MaterialOpt

        kwargs = {"objective": self.objective, "diagnostics": self.diagnostics}
        if self.objective_params is not None:
            kwargs["objective_params"] = self.objective_params
        if self.backend is not None:
            kwargs["backend"] = self.backend
        return MaterialOpt.apply(
            self.model,
            self.lame if self._enu is None else self._enu.lame(),
            **kwargs,
        )


def material_opt(
    model: DifferentiableModel,
    lame: Any = None,
    *,
    materials: Mapping | None = None,
    objective: Any,
    objective_params: Mapping[str, Any] | None = None,
    backend: Any | None = None,
) -> MaterialOptimization:
    """Create a material objective from direct Lame tensors or body E/nu overrides.

    ``materials={body: {"E": E_tensor, "nu": nu_tensor, "unit": "MPa"}}``
    overrides only the supplied properties; all other properties/bodies inherit
    the model. A supplied E requires a stress unit; nu is dimensionless. This
    entry supports uniform isotropic LinearElasticity/NeoHookean in 2D and 3D.
    Values may be scalars or scalar floating tensors. Trainable leaves appear
    in ``parameters`` in mapping order (E before nu), with duplicates removed.
    Mappings are reevaluated on every call. Only first derivatives are supported.
    The direct ``lame`` interface continues to use backend internal units.
    """

    return MaterialOptimization(
        model=model,
        lame=lame,
        materials=materials,
        objective=objective,
        objective_params=objective_params,
        backend=backend,
    )


class _ENuMaterials:
    """Resolve constant model data once; rebuild differentiable mappings per call."""

    def __init__(self, payload, overrides, backend):
        self.overrides, self.parameters, self.like = self._inputs(overrides)
        self.backend = backend if backend is not None else _load_default_backend()
        self._load_model(payload)

    @classmethod
    def _inputs(cls, overrides, *, internal_units=False):
        import torch
        from .objectives import _selection_id

        if overrides is None:
            return {}, (), torch.tensor(0., dtype=torch.float64)
        if not isinstance(overrides, Mapping) or not overrides:
            raise ValueError("materials must be a nonempty body-to-E/nu mapping")
        normalized = {}
        tensors = []
        parameters = []
        for body, fields in overrides.items():
            body_id = 0 if type(body) is int and body == 0 else _selection_id(body)
            if body_id in normalized:
                raise ValueError(f"Duplicate material override for body {body_id}")
            if (not isinstance(fields, Mapping) or not set(fields) & {"E", "nu"}
                    or set(fields) - {"E", "nu", "unit"}):
                raise ValueError("Material overrides accept only E, nu and unit")
            if "E" in fields and not (internal_units and "unit" not in fields) and (
                    not isinstance(fields.get("unit"), str) or not fields["unit"]):
                raise ValueError("A supplied E requires an explicit stress unit")
            if "unit" in fields and "E" not in fields:
                raise ValueError("unit applies to E, not dimensionless nu")
            normalized[body_id] = dict(fields)
            for name in ("E", "nu"):
                if name not in fields:
                    continue
                value = fields[name]
                cls._validate_scalar(value, name)
                if isinstance(value, torch.Tensor):
                    tensors.append(value)
                    if value.requires_grad and all(value is not p for p in parameters):
                        parameters.append(value)
        like = tensors[0] if tensors else torch.tensor(0., dtype=torch.float64)
        if any(t.dtype != like.dtype or t.device != like.device for t in tensors):
            raise ValueError("All material tensors must have the same dtype and device")
        return normalized, tuple(parameters), like

    def _load_model(self, payload):
        # Reject expressions/heterogeneous models instead of silently homogenizing them.
        definitions = payload.get("materials")
        if isinstance(definitions, Mapping):
            definitions = [definitions]
        if not isinstance(definitions, list) or not definitions:
            raise ValueError("E/nu materials require explicit model material definitions")
        for definition in definitions:
            if not isinstance(definition, Mapping):
                raise ValueError("Model material definitions must be objects")
            if definition.get("type") not in ("LinearElasticity", "NeoHookean"):
                raise ValueError("E/nu overrides support only LinearElasticity and NeoHookean")
            if any(key in definition for key in ("lambda", "mu", "__polyfem_material_element_index")):
                raise ValueError("E/nu overrides require uniform E/nu model materials")
            if "E" in definition and "young" in definition:
                raise ValueError("Use one Young's modulus field: E or young, not both")
            self._quantity(definition.get("E", definition.get("young")), "E")
            self._validate_scalar(definition.get("nu"), "nu")

        session_type = _backend.require_material_enu_backend(self.backend)
        session = session_type()
        session.set_settings(payload)
        self.metadata = ModelMetadata(**session.model_metadata())
        resolved = json.loads(session.settings())
        self.units = resolved.get("units", {})
        body_ids = set(self.metadata.element_body_ids)
        if self.metadata.dimension not in (2, 3) or not body_ids:
            raise ValueError("E/nu materials require 2D or 3D mesh body metadata")
        if not set(self.overrides) <= body_ids:
            raise ValueError("Unknown body in material overrides")
        self.base = {}
        for definition in definitions:
            ids = definition.get("id", list(body_ids) if len(definitions) == 1 else None)
            ids = ids if isinstance(ids, list) else [ids]
            for body_id in ids:
                if type(body_id) is not int or body_id in self.base or body_id not in body_ids:
                    raise ValueError("Material definitions must uniquely cover mesh body IDs")
                E, unit = self._quantity(definition.get("E", definition.get("young")), "E")
                self.base[body_id] = (E, definition["nu"], self._scale(unit))
        if set(self.base) != body_ids:
            raise ValueError("Material definitions must cover every mesh body")
        self.scales = {body: self._scale(fields["unit"]) for body, fields in self.overrides.items()
                       if "E" in fields}

    @staticmethod
    def _validate_scalar(value, name):
        import torch
        if isinstance(value, torch.Tensor):
            if value.ndim != 0 or not value.is_floating_point():
                raise ValueError(f"{name} must be a scalar floating tensor")
            number = float(value.detach())
        elif isinstance(value, Real) and not isinstance(value, bool):
            number = float(value)
        else:
            raise ValueError(f"{name} must be uniform numeric data, not an expression or field")
        if not math.isfinite(number) or (name == "E" and number <= 0) or (
                name == "nu" and not -1 < number < .5):
            raise ValueError("Isotropic materials require finite E > 0 and -1 < nu < 0.5")

    @classmethod
    def _quantity(cls, value, name):
        unit = None
        if isinstance(value, Mapping):
            if set(value) != {"value", "unit"} or not isinstance(value["unit"], str) or not value["unit"]:
                raise ValueError("Material quantities require numeric value and explicit unit")
            value, unit = value["value"], value["unit"]
        cls._validate_scalar(value, name)
        return value, unit

    def _scale(self, unit):
        # Bare model values already use simulation units, exactly as in ordinary forward.
        return 1. if unit is None else self.backend._material_stress_scale(self.units, unit)

    def lame(self, overrides=None):
        import torch
        from .torch_ops import _MaterialENu

        if overrides is None:
            fields_by_body, like, scales = self.overrides, self.like, self.scales
        else:
            fields_by_body, _, like = self._inputs(overrides, internal_units=True)
            if not set(fields_by_body) <= set(self.base):
                raise ValueError("Unknown body in material overrides")
            scales = {body: self._scale(fields.get("unit"))
                      for body, fields in fields_by_body.items() if "E" in fields}
        body_ids, pairs = [], []
        for body, (base_E, base_nu, base_scale) in self.base.items():
            fields = fields_by_body.get(body, {})
            E, nu = fields.get("E", base_E), fields.get("nu", base_nu)
            self._validate_scalar(E, "E")
            self._validate_scalar(nu, "nu")
            E = E if isinstance(E, torch.Tensor) else like.new_tensor(E)
            nu = nu if isinstance(nu, torch.Tensor) else like.new_tensor(nu)
            pairs.append(torch.stack((E * scales.get(body, base_scale), nu)))
            body_ids.append(body)
        converted = _MaterialENu.apply(torch.stack(pairs), self.metadata.dimension, self.backend)
        return self.metadata.material_by_body(dict(zip(body_ids, converted)))


def _single_material_payload(
    *,
    model: DifferentiableModel,
    lame: Any,
) -> dict[str, Any]:
    if not isinstance(model, DifferentiableModel):
        raise TypeError("model must be a polyfempy.differentiable_api.DifferentiableModel")
    if lame is None:
        raise ValueError("lame must not be None")

    payloads = model.as_dicts()
    if len(payloads) != 1:
        raise ValueError("material_opt MVP supports exactly one model")
    return payloads[0]


def _run_material_session(
    *,
    payload: dict[str, Any],
    lame: Any,
    objective: Any,
    objective_params: Mapping[str, Any] | None = None,
    backend: Any | None = None,
    target: dict[str, Any] | None = None,
) -> tuple[Any, Any]:
    backend_module = backend if backend is not None else _load_default_backend()
    session_type = _backend.require_material_opt_backend(backend_module)
    session = session_type()

    session.set_settings(payload)
    session.set_material_lame_parameters(lame)
    if target is not None:
        if not callable(getattr(session, "set_target_settings", None)):
            raise _backend.BackendContractError("Rebuild the backend for fixed target matching: missing set_target_settings")
        session.set_target_settings(target)
    objective_payload = _objective_payload(
        objective,
        objective_params=objective_params,
        default_selection="all",
    )
    if objective_payload is None:
        raise ValueError("material_opt requires an objective")
    session.set_objective(objective_payload)
    return session.solve_material_objective(), session


__all__ = [
    "MaterialOptimization",
    "material_opt",
]
