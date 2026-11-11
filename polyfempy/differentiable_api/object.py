"""Callable simulation outputs for user-defined PyTorch loss functions."""

from __future__ import annotations

from .material import _ENuMaterials
from .model import DifferentiableModel, model as make_model
from .objectives import Objective, _selection_id
from .shape import _objective_payload


def _model(config):
    wrapped = config if isinstance(config, DifferentiableModel) else make_model(config)
    if len(wrapped.as_dicts()) != 1:
        raise ValueError("Differentiable objects support exactly one model")
    return wrapped


def _objective(objective, selection):
    if objective is None:
        raise ValueError("objective must not be None")
    if selection is not None and not isinstance(objective, Objective):
        raise ValueError("For an objective spec, put selection in the spec itself")
    return _objective_payload(objective, default_selection="all" if selection is None else selection)


class MaterialObject:
    """Evaluate a scalar output with E/nu overrides.

    With body, accept scalar or per-element E/nu for that body. Without body,
    accept materials mapping body IDs/handles to scalar or per-element overrides.

    E uses simulation internal stress units unless unit is supplied at creation.
    Omitted properties inherit the config. The output keeps the objective's
    units/weight; unit conversion and outer losses can be ordinary torch code.
    Model metadata is prepared once; each call builds a fresh autograd graph.
    Single-body fields have shape (n_elements,) or (n_elements, 1),
    in element_indices order.
    Without body, element_indices and n_elements describe the whole mesh.
    """

    def __init__(self, config, *, body=None, objective, selection=None, unit=None, target=None, backend=None):
        self.model = _model(config)
        self.objective = _objective(objective, selection)
        self._target = None if target is None else _model(target).as_dicts()[0]
        spatial = self.objective.get("static_objective", self.objective)
        if self._target is not None:
            if (self.objective.get("type") not in ("target", "transient_integral")
                    or spatial.get("type") != "target"
                    or self.objective.get("state", 0) != 0
                    or spatial.get("state", 0) != 0
                    or spatial.get("target_state") != 1):
                raise ValueError("target requires a target objective with state=0, target_state=1 (optionally transient_integral)")
        elif spatial.get("type") == "target":
            raise ValueError("A target objective requires target=config")
        self.body = None if body is None else (0 if type(body) is int and body == 0 else _selection_id(body))
        if body is None and unit is not None:
            raise ValueError("Without body, put each stress unit in the materials mapping")
        if unit is not None and (not isinstance(unit, str) or not unit):
            raise ValueError("unit must be a nonempty stress unit or None for internal units")
        self.unit = unit
        self._enu = _ENuMaterials(self.model.as_dicts()[0], None, backend)
        if self.body is not None and self.body not in self._enu.base:
            raise ValueError("Unknown body in material overrides")
        self.metadata = self._enu.metadata
        self.element_indices = tuple(i for i, body_id in enumerate(self.metadata.element_body_ids)
                                     if self.body is None or body_id == self.body)
        self.n_elements = len(self.element_indices)
        self.element_indices_by_body = {
            body_id: tuple(i for i, value in enumerate(self.metadata.element_body_ids)
                           if value == body_id)
            for body_id in self._enu.base
        }
        if unit is not None:
            self._enu._scale(unit)
        self.diagnostics = {}

    def __call__(self, *, E=None, nu=None, materials=None):
        import torch
        from .torch_ops import MaterialOpt

        if self.body is None:
            if E is not None or nu is not None or materials is None:
                raise ValueError("Without body, supply materials={body: {'E': ..., 'nu': ...}}")
            lame = self._body_materials(materials)
            kwargs = {} if self._target is None else {"target": self._target}
            return MaterialOpt.apply(self.model, lame, objective=self.objective,
                                     backend=self._enu.backend, diagnostics=self.diagnostics, **kwargs)
        if materials is not None:
            raise ValueError("Use materials without body, or E/nu with body; do not mix them")

        fields = {}
        if E is not None:
            fields["E"] = E
            if self.unit is not None:
                fields["unit"] = self.unit
        if nu is not None:
            fields["nu"] = nu
        if not fields:
            raise ValueError("Supply E, nu, or both")
        if any(isinstance(v, torch.Tensor) and v.ndim > 0 for v in (E, nu)):
            lame = self._element_material(E, nu)
        else:
            lame = self._enu.lame({self.body: fields})
        kwargs = {} if self._target is None else {"target": self._target}
        return MaterialOpt.apply(self.model, lame, objective=self.objective,
                                 backend=self._enu.backend, diagnostics=self.diagnostics, **kwargs)

    def _body_materials(self, materials):
        from collections.abc import Mapping
        import torch

        if not isinstance(materials, Mapping) or not materials:
            raise ValueError("materials must be a nonempty body-to-E/nu mapping")
        normalized = {}
        for body, fields in materials.items():
            body_id = 0 if type(body) is int and body == 0 else _selection_id(body)
            if body_id in normalized:
                raise ValueError(f"Duplicate material override for body {body_id}")
            if body_id not in self._enu.base:
                raise ValueError("Unknown body in material overrides")
            if (not isinstance(fields, Mapping) or not set(fields) & {"E", "nu"}
                    or set(fields) - {"E", "nu", "unit"}):
                raise ValueError("Material overrides accept only E, nu and unit")
            if any(fields[key] is None for key in ("E", "nu") if key in fields):
                raise ValueError("Omit inherited E/nu properties instead of supplying None")
            if "unit" in fields and ("E" not in fields or not isinstance(fields["unit"], str)
                                     or not fields["unit"]):
                raise ValueError("unit requires E and a nonempty stress unit")
            normalized[body_id] = fields
        tensors = [value for fields in normalized.values() for key, value in fields.items()
                   if key in ("E", "nu") and isinstance(value, torch.Tensor)]
        if not any(value.ndim > 0 for value in tensors):
            return self._enu.lame(materials)
        like = tensors[0]
        if any(value.dtype != like.dtype or value.device != like.device for value in tensors):
            raise ValueError("All material tensors must have the same dtype and device")
        lame = self._enu.lame().to(dtype=like.dtype, device=like.device)
        for body_id, fields in normalized.items():
            lame = self._element_material(fields.get("E"), fields.get("nu"),
                                          body=body_id, unit=fields.get("unit"),
                                          baseline=lame, like=like)
        return lame

    def _element_material(self, E, nu, *, body=None, unit=None, baseline=None, like=None):
        import torch
        from .torch_ops import _MaterialENu

        tensors = [v for v in (E, nu) if isinstance(v, torch.Tensor)]
        if body is None:
            body, unit = self.body, self.unit
        element_indices = self.element_indices_by_body[body]
        n_elements = len(element_indices)
        if like is None:
            like = tensors[0]
        if any(v.dtype != like.dtype or v.device != like.device for v in tensors):
            raise ValueError("All material tensors must have the same dtype and device")
        base_E, base_nu, base_scale = self._enu.base[body]
        values = []
        for name, value, default in (("E", E, base_E * base_scale), ("nu", nu, base_nu)):
            if value is None:
                value = default
            elif not isinstance(value, torch.Tensor):
                self._enu._validate_scalar(value, name)
            if isinstance(value, torch.Tensor) and (
                    not value.is_floating_point()
                    or tuple(value.shape) not in ((), (n_elements,), (n_elements, 1))):
                raise ValueError(f"{name} must be a scalar or floating tensor with shape ({n_elements},) or ({n_elements}, 1)")
            tensor = torch.as_tensor(value, dtype=like.dtype, device=like.device)
            if not tensor.is_floating_point():
                raise ValueError("Material tensors must be floating-point")
            valid = tensor > 0 if name == "E" else (tensor > -1) & (tensor < .5)
            if not bool((torch.isfinite(tensor) & valid).all()):
                raise ValueError("Isotropic materials require finite E > 0 and -1 < nu < 0.5")
            if name == "E" and E is not None and unit is not None:
                tensor = tensor * self._enu._scale(unit)
            if tensor.ndim == 2:
                tensor = tensor.reshape(n_elements)
            values.append(tensor.expand(n_elements))
        pairs = torch.stack(values, dim=1)
        updated = _MaterialENu.apply(pairs, self.metadata.dimension, self._enu.backend)
        # Fixed bodies retain their configured properties and internal units.
        if baseline is None:
            baseline = self._enu.lame().to(dtype=like.dtype, device=like.device)
        indices = torch.tensor(element_indices, dtype=torch.long, device=like.device)
        return baseline.index_copy(0, indices, updated)


class ShapeObject:
    """Evaluate a scalar output for a full-model vertex tensor V.

    V has shape (n_vertices, dimension), in backend vertex order and internal
    length units. Connectivity comes from config and stays fixed. selection
    chooses the output region, not the vertices to move. Keep other vertices
    fixed in the user's torch mapping. Only first derivatives are supported.
    """

    def __init__(self, config, *, objective, selection=None, target=None, backend=None):
        self.model = _model(config)
        self.objective = _objective(objective, selection)
        self.target = None if target is None else _model(target).as_dicts()[0]
        if self.target is not None:
            spatial = self.objective.get("static_objective", {})
            if (self.objective.get("type") != "transient_integral" or spatial.get("type") != "target"
                    or spatial.get("target_state") != 1 or not spatial.get("reference_cached_body_ids")
                    or not spatial.get("surface_selection")):
                raise ValueError("Shape target requires a transient surface target on explicit reference bodies")
        self.backend = backend

    def __call__(self, *, V):
        import torch
        from .torch_ops import ShapeOpt

        if (not isinstance(V, torch.Tensor) or not V.is_floating_point()
                or V.ndim != 2 or V.shape[1] not in (2, 3)
                or V.shape[0] == 0 or not bool(torch.isfinite(V).all())):
            raise ValueError("V must be a finite floating tensor with shape (N, 2) or (N, 3)")
        if self.target is not None:
            return ShapeOpt.apply(self.model, "all", V, self.backend, self.objective, None, self.target)
        return ShapeOpt.apply(self.model, "all", V, objective=self.objective,
                              backend=self.backend)


class InitialObject:
    """Evaluate a scalar output with new initial displacement and velocity.

    Component vectors apply uniformly to the entire model, limited to
    unconstrained P1 simplex displacement spaces without periodic boundaries.
    Omitted displacement/velocity is zero, not inherited from config.
    Alternatively pass initial_condition=(n_dofs, 2), columns [u0, v0], in
    backend FE order. Inputs use internal length and length/time units.
    selection measures the output; it does not select which body moves.
    Only first derivatives are supported.
    target is a fixed reference simulation on the same mesh and time grid.
    """

    def __init__(self, config, *, objective, selection=None, target=None, backend=None):
        self.model = _model(config)
        self.objective = _objective(objective, selection)
        self._target = None if target is None else _model(target).as_dicts()[0]
        spatial = self.objective.get("static_objective", self.objective)
        if self._target is not None:
            if (self.objective.get("type") != "transient_integral"
                    or spatial.get("type") not in ("target", "center-target")
                    or self.objective.get("state", 0) != 0
                    or spatial.get("state", 0) != 0
                    or spatial.get("target_state") != 1):
                raise ValueError("target requires a transient_integral target or center-target objective with state=0, target_state=1")
        elif spatial.get("type") in ("target", "center-target"):
            raise ValueError("A target objective requires target=config")
        self.backend = backend
        self.metadata = self.model.metadata(backend=backend)

    def __call__(self, *, velocity=None, displacement=None, initial_condition=None):
        import torch
        from .torch_ops import InitialConditionOpt

        if initial_condition is not None:
            if velocity is not None or displacement is not None:
                raise ValueError("Pass initial_condition or velocity/displacement, not both")
            if (not isinstance(initial_condition, torch.Tensor)
                    or not initial_condition.is_floating_point()
                    or tuple(initial_condition.shape) != (self.metadata.n_dofs, 2)
                    or not bool(torch.isfinite(initial_condition).all())):
                raise ValueError(
                    f"initial_condition must be a finite floating tensor with shape "
                    f"({self.metadata.n_dofs}, 2), columns [displacement, velocity]"
                )
        else:
            if velocity is None and displacement is None:
                raise ValueError("Supply velocity, displacement, or initial_condition")
            if velocity is None:
                self.metadata._check_vector(displacement, self.metadata.value_dim, "displacement")
                velocity = torch.zeros_like(displacement)
            initial_condition = self.metadata.uniform_initial_condition(
                velocity, displacement=displacement,
            )
        kwargs = {} if self._target is None else {"target": self._target}
        return InitialConditionOpt.apply(self.model, initial_condition,
                                         objective=self.objective, backend=self.backend, **kwargs)


def material(config, *, body=None, objective, selection=None, unit=None, target=None, backend=None):
    """Create obj(E=..., nu=...) for one body, or obj(materials=...) for many.

    Without body, materials maps body IDs/handles to scalar or element-vector
    E/nu overrides. Vectors follow element_indices_by_body[body_id] ordering.
    Each E uses internal units unless its mapping supplies unit. Omitted
    properties and bodies inherit config on every call. This is one solve;
    different parameter categories and general multi-state targets are not added.
    """
    return MaterialObject(config, body=body, objective=objective, selection=selection,
                          unit=unit, target=target, backend=backend)


def shape(config, *, objective, selection=None, target=None, backend=None):
    """Create obj(V=...) with fixed mesh connectivity and full-model vertices."""
    return ShapeObject(config, objective=objective, selection=selection, target=target, backend=backend)


def initial(config, *, objective, selection=None, target=None, backend=None):
    """Create obj(velocity=..., displacement=...) or obj(initial_condition=...)."""
    return InitialObject(config, objective=objective, selection=selection, target=target, backend=backend)


__all__ = ["material", "shape", "initial", "damping", "friction", "dirichlet", "pressure",
           "MaterialObject", "ShapeObject", "InitialObject"]


def damping(config, *, objective, selection=None, target=None, backend=None):
    """Create obj(psi=..., phi=...) for global viscous damping (transient only).

    Optional target is a fixed reference simulation with the same mesh and time
    settings. Matching uses a transient surface target objective. Only first derivatives are supported.
    """
    from .control import DampingObject
    return DampingObject(config, kind='damping', objective=objective,
                         selection=selection, target=target, backend=backend)


def friction(config, *, objective, selection=None, target=None, backend=None):
    """Create obj(coefficient=...) for global contact friction (transient only).

    Optional target is a fixed reference with the same mesh and time settings.
    Requires fixed positive barrier stiffness. Target matching supports first derivatives.
    """
    from .control import FrictionObject
    return FrictionObject(config, kind='friction', objective=objective,
                          selection=selection, target=target, backend=backend)


def dirichlet(config, *, objective, boundary_ids=None, node_ids=None, time_slices=None, selection=None, backend=None):
    """Create obj(values=...) for boundary or nodal displacement control.

    Choose boundary_ids (transient) or node_ids (quasistatic). Nodal values have
    shape (node, dimension), in the supplied input-mesh vertex ID order, and
    replace existing nodal constraints. All components must be constrained.
    Boundary values have shape (time, boundary, dimension); time slice 0 controls
    step 1. Omitted slices select all steps. Values use internal length units.
    """
    from .control import ControlObject
    if (boundary_ids is None) == (node_ids is None):
        raise ValueError('Provide exactly one of boundary_ids or node_ids')
    if node_ids is not None:
        if time_slices is not None:
            raise ValueError('time_slices do not apply to quasistatic node_ids')
        return ControlObject(config, kind='dirichlet-nodes', objective=objective,
                             node_ids=node_ids, selection=selection, backend=backend)
    return ControlObject(config, kind='dirichlet-boundary', objective=objective,
                         boundary_ids=boundary_ids, time_slices=time_slices,
                         selection=selection, backend=backend)


def pressure(config, *, boundary_ids, objective, time_slices=None, selection=None, backend=None):
    """Create obj(values=...) with shape (boundary,) or (time, boundary).

    Values use internal pressure units and upstream's pressure sign convention.
    Time slice 0 controls step 1. Omitted slices select all steps.
    """
    from .control import ControlObject
    return ControlObject(config, kind='pressure', objective=objective,
                         boundary_ids=boundary_ids, time_slices=time_slices,
                         selection=selection, backend=backend)
