"""User-design adapters for differentiable PolyFEM optimization."""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import torch


DesignValueMap = Callable[[Sequence[torch.nn.Parameter]], Any]
VertexMap = Callable[..., torch.Tensor]
Projection = Callable[[Sequence[torch.nn.Parameter]], None]
ParameterBounds = Tuple[Optional[float], Optional[float]]


def _accepts_context(fn: VertexMap) -> bool:
    """Return True when ``fn`` should receive the optional context argument."""
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return True

    parameters = list(signature.parameters.values())
    if any(parameter.kind == inspect.Parameter.VAR_POSITIONAL for parameter in parameters):
        return True

    positional = [
        parameter
        for parameter in parameters
        if parameter.kind
        in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }
    ]
    return len(positional) >= 3


def _call_vertex_map(
    vertex_map: VertexMap,
    design_value: Any,
    base_vertices: torch.Tensor,
    context: Any,
) -> torch.Tensor:
    """Call a user vertex map with or without ``context``.

    New simple examples can use ``vertex_map(params, base_vertices)``. Advanced
    maps that cache masks or metadata can use
    ``vertex_map(params, base_vertices, context)``.
    """
    if _accepts_context(vertex_map):
        return vertex_map(design_value, base_vertices, context)
    return vertex_map(design_value, base_vertices)


def _normalize_bounds(bounds: Any) -> Optional[ParameterBounds]:
    if bounds is None:
        return None
    if not isinstance(bounds, Sequence) or isinstance(bounds, (str, bytes)):
        raise TypeError("bounds must be a (lower, upper) pair")
    if len(bounds) != 2:
        raise ValueError(f"bounds must contain exactly two entries, got {len(bounds)}")
    lower = None if bounds[0] is None else float(bounds[0])
    upper = None if bounds[1] is None else float(bounds[1])
    if lower is not None and upper is not None and lower > upper:
        raise ValueError(f"bounds lower value {lower} is greater than upper value {upper}")
    return (lower, upper)


def make_parameter(
    name: str,
    value: Any,
    *,
    bounds: Optional[Sequence[Optional[float]]] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = True,
) -> torch.nn.Parameter:
    """Create a named PyTorch design parameter.

    The returned object is still a normal ``torch.nn.Parameter``. The name and
    optional bounds are metadata used by the user-friendly design builders.
    """
    if isinstance(value, torch.Tensor):
        tensor = value.detach().clone()
        if dtype is not None or device is not None:
            tensor = tensor.to(
                dtype=dtype if dtype is not None else tensor.dtype,
                device=device if device is not None else tensor.device,
            )
        elif not torch.is_floating_point(tensor):
            tensor = tensor.to(dtype=torch.get_default_dtype())
    else:
        tensor = torch.as_tensor(
            value,
            dtype=dtype if dtype is not None else torch.get_default_dtype(),
            device=device,
        )

    parameter = torch.nn.Parameter(tensor, requires_grad=bool(requires_grad))
    parameter._polyfem_design_name = str(name)  # type: ignore[attr-defined]
    parameter._polyfem_bounds = _normalize_bounds(bounds)  # type: ignore[attr-defined]
    return parameter


def parameter_name(parameter: torch.nn.Parameter, default: Optional[str] = None) -> Optional[str]:
    """Return the name stored by ``make_parameter(...)`` if present."""
    return getattr(parameter, "_polyfem_design_name", default)


def parameter_bounds(parameter: torch.nn.Parameter) -> Optional[ParameterBounds]:
    """Return the bounds stored by ``make_parameter(...)`` if present."""
    return getattr(parameter, "_polyfem_bounds", None)


def normalize_design_parameters(
    parameters: Sequence[torch.nn.Parameter],
    *,
    parameter_names: Optional[Sequence[str]] = None,
    bounds: Optional[Mapping[str, Sequence[Optional[float]]]] = None,
) -> tuple[list[torch.nn.Parameter], list[str], list[Optional[ParameterBounds]]]:
    """Normalize user parameters into torch parameters, names, and bounds."""
    params = list(parameters)
    if not params:
        raise ValueError("at least one design parameter is required")

    if parameter_names is not None and len(parameter_names) != len(params):
        raise ValueError(
            "parameter_names length must match parameters length: "
            f"{len(parameter_names)} != {len(params)}"
        )

    names: list[str] = []
    param_bounds: list[Optional[ParameterBounds]] = []
    for index, parameter in enumerate(params):
        if not isinstance(parameter, torch.nn.Parameter):
            raise TypeError(
                "parameters must be torch.nn.Parameter objects; "
                f"entry {index} has type {type(parameter).__name__}"
            )
        if parameter_names is None:
            name = parameter_name(parameter, f"param_{index}")
        else:
            name = str(parameter_names[index])
        if name is None or not str(name):
            raise ValueError(f"parameter name at index {index} is empty")
        names.append(str(name))

        if bounds is not None and str(name) in bounds:
            param_bounds.append(_normalize_bounds(bounds[str(name)]))
        else:
            param_bounds.append(parameter_bounds(parameter))

    if len(set(names)) != len(names):
        raise ValueError(f"parameter names must be unique, got {names!r}")
    return params, names, param_bounds


def make_named_parameter_map(
    parameters: Sequence[torch.nn.Parameter],
    *,
    parameter_names: Optional[Sequence[str]] = None,
) -> DesignValueMap:
    """Build a map that presents parameters to ``vertex_map`` as a name dict."""
    _, names, _ = normalize_design_parameters(
        parameters,
        parameter_names=parameter_names,
    )

    def parameter_map(params: Sequence[torch.nn.Parameter]) -> dict[str, torch.nn.Parameter]:
        if len(params) != len(names):
            raise ValueError(f"expected {len(names)} parameters, got {len(params)}")
        return dict(zip(names, params))

    return parameter_map


def make_bounds_projector(
    parameters: Sequence[torch.nn.Parameter],
    *,
    parameter_names: Optional[Sequence[str]] = None,
    bounds: Optional[Mapping[str, Sequence[Optional[float]]]] = None,
) -> Projection:
    """Build a projection function that clamps named parameters to bounds."""
    _, names, param_bounds = normalize_design_parameters(
        parameters,
        parameter_names=parameter_names,
        bounds=bounds,
    )

    def project(params: Sequence[torch.nn.Parameter]) -> None:
        if len(params) != len(param_bounds):
            raise ValueError(f"expected {len(param_bounds)} parameters, got {len(params)}")
        with torch.no_grad():
            for parameter, name, bound in zip(params, names, param_bounds):
                if bound is None:
                    continue
                lower, upper = bound
                if lower is not None and upper is not None:
                    parameter.clamp_(min=lower, max=upper)
                elif lower is not None:
                    parameter.clamp_(min=lower)
                elif upper is not None:
                    parameter.clamp_(max=upper)

    return project


def validate_parameterized_vertices(
    vertices: Any,
    *,
    base_vertices: Optional[torch.Tensor] = None,
    parameters: Optional[Sequence[torch.nn.Parameter]] = None,
    source: str = "vertex_map",
) -> torch.Tensor:
    """Validate vertices returned by a user parameterized-shape map.

    This catches common user mistakes close to the source instead of letting
    PyTorch or PolyFEM fail later with a lower-level error.
    """
    source_name = str(source)
    if not isinstance(vertices, torch.Tensor):
        raise TypeError(
            f"{source_name} must return a torch.Tensor, got {type(vertices).__name__}. "
            "Use torch operations in the map; do not convert differentiable values to NumPy."
        )

    if base_vertices is not None and tuple(vertices.shape) != tuple(base_vertices.shape):
        raise ValueError(
            f"{source_name} must return vertices with shape {tuple(base_vertices.shape)}, "
            f"got {tuple(vertices.shape)}. Keep mesh topology fixed; do not remesh or "
            "change the vertex count inside the differentiable map."
        )

    if vertices.ndim != 2:
        raise ValueError(
            f"{source_name} must return a 2D vertex tensor with shape (n_vertices, dim), "
            f"got shape {tuple(vertices.shape)}."
        )

    if vertices.shape[-1] not in (2, 3):
        raise ValueError(
            f"{source_name} must return vertices whose last dimension is 2 or 3, "
            f"got shape {tuple(vertices.shape)}."
        )

    if not torch.is_floating_point(vertices):
        raise TypeError(
            f"{source_name} must return floating-point vertices, got dtype {vertices.dtype}."
        )

    if base_vertices is not None:
        if vertices.device != base_vertices.device:
            raise ValueError(
                f"{source_name} returned vertices on device {vertices.device}, but "
                f"base_vertices is on {base_vertices.device}. Return vertices on the same "
                "device as base_vertices."
            )
        if vertices.dtype != base_vertices.dtype:
            raise TypeError(
                f"{source_name} returned vertices with dtype {vertices.dtype}, but "
                f"base_vertices has dtype {base_vertices.dtype}. Return the same dtype as "
                "base_vertices."
            )

    if not bool(torch.isfinite(vertices).all().detach().cpu().item()):
        raise ValueError(
            f"{source_name} returned vertices containing NaN or Inf values. Check the "
            "parameter bounds and the differentiable geometry formula."
        )

    params = [] if parameters is None else list(parameters)
    if (
        torch.is_grad_enabled()
        and any(parameter.requires_grad for parameter in params)
        and not vertices.requires_grad
    ):
        raise ValueError(
            f"{source_name} returned vertices that are not connected to differentiable "
            "parameters. Use torch operations on the parameters and do not call detach(), "
            "item(), numpy(), or torch.no_grad() inside the differentiable map."
        )

    return vertices


@dataclass
class ParameterizedVertexDesign:
    """Map user design parameters to a fixed-topology vertex tensor.

    PolyFEM shape autograd differentiates with respect to vertices ``X``. This
    adapter lets users optimize lower-dimensional parameters by providing a
    differentiable PyTorch map:

    ``parameters -> design_value -> vertices``.

    ``vertex_map`` must use PyTorch operations for any differentiable math and
    must return a vertex tensor with the same shape as ``base_vertices``. It may
    move vertices, but it must not change mesh topology, vertex count, or cell
    connectivity. Simple maps can accept ``(design_value, base_vertices)``.
    Advanced maps can accept ``(design_value, base_vertices, context)`` and
    cache masks or other non-differentiable metadata in ``context`` instead of
    rebuilding them every call.
    """

    parameters: Optional[Sequence[torch.nn.Parameter]] = None
    vertex_map: Optional[VertexMap] = None
    base_vertices: Optional[torch.Tensor] = None
    parameter_map: Optional[DesignValueMap] = None
    project: Optional[Projection] = None
    context: Any = None
    geometry: Any = None
    differentiable_params: Optional[list[str]] = None
    name: str = "parameterized_shape"
    validate_vertices: bool = True

    def torch_parameters(self) -> list[torch.nn.Parameter]:
        if self.parameters is not None:
            return list(self.parameters)
        if self.geometry is not None and hasattr(self.geometry, "parameters"):
            return list(self.geometry.parameters())
        return []

    def design_value(self) -> Any:
        params = self.torch_parameters()
        if self.parameter_map is not None:
            return self.parameter_map(params)
        if len(params) == 1:
            return params[0]
        return tuple(params)

    def vertices(self) -> torch.Tensor:
        params = self.torch_parameters()
        if self.vertex_map is not None:
            if self.base_vertices is None:
                raise ValueError("ParameterizedVertexDesign.vertex_map requires base_vertices")
            out = _call_vertex_map(
                self.vertex_map,
                self.design_value(),
                self.base_vertices,
                self.context,
            )
            source = "vertex_map"
        elif self.geometry is not None:
            out = self.geometry()
            source = "geometry"
        else:
            raise ValueError("ParameterizedVertexDesign requires vertex_map or geometry")

        if self.validate_vertices:
            return validate_parameterized_vertices(
                out,
                base_vertices=self.base_vertices,
                parameters=params,
                source=source,
            )
        if not isinstance(out, torch.Tensor):
            raise TypeError(f"{source} must return a torch.Tensor, got {type(out).__name__}")
        return out

    def project_(self) -> None:
        params = self.torch_parameters()
        if self.project is not None:
            self.project(params)
            return
        if self.geometry is not None and hasattr(self.geometry, "project"):
            self.geometry.project()

    def differentiable_param_names(self) -> list[str]:
        if self.differentiable_params is not None:
            return list(self.differentiable_params)
        return ["parameterized_geometry"]


__all__ = [
    "DesignValueMap",
    "ParameterBounds",
    "ParameterizedVertexDesign",
    "Projection",
    "VertexMap",
    "make_bounds_projector",
    "make_named_parameter_map",
    "make_parameter",
    "normalize_design_parameters",
    "parameter_bounds",
    "parameter_name",
    "validate_parameterized_vertices",
]
