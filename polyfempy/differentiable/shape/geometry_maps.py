"""Reusable PyTorch helpers for parameterized vertex maps.

These helpers do not define an experiment. They only provide common tensor
operations that user-defined maps often need when building

    parameters -> vertices

for parameterized shape differentiation.
"""

from __future__ import annotations

from typing import Union

import torch

TensorLike = Union[float, int, torch.Tensor]


def _as_tensor_like(value: TensorLike, reference: torch.Tensor) -> torch.Tensor:
    """Return ``value`` as a tensor on the same dtype/device as ``reference``."""
    if isinstance(value, torch.Tensor):
        return value.to(dtype=reference.dtype, device=reference.device)
    return torch.as_tensor(value, dtype=reference.dtype, device=reference.device)


def _validate_vertices(vertices: torch.Tensor) -> None:
    if not isinstance(vertices, torch.Tensor):
        raise TypeError(f"vertices must be a torch.Tensor, got {type(vertices).__name__}")
    if vertices.ndim != 2:
        raise ValueError(f"vertices must be a 2D tensor with shape (n_vertices, dim), got {tuple(vertices.shape)}")
    if vertices.shape[1] < 2:
        raise ValueError(f"vertices must have at least x/y coordinates, got shape {tuple(vertices.shape)}")


def _validate_axis(vertices: torch.Tensor, axis: int) -> int:
    if not isinstance(axis, int):
        raise TypeError(f"axis must be an int, got {type(axis).__name__}")
    if axis < 0 or axis >= vertices.shape[1]:
        raise ValueError(
            f"axis must be in [0, {vertices.shape[1] - 1}] for vertices shape {tuple(vertices.shape)}, got {axis}"
        )
    return axis


def _validate_mask(vertices: torch.Tensor, mask: torch.Tensor) -> None:
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"mask must be a torch.Tensor, got {type(mask).__name__}")
    if mask.dtype != torch.bool:
        raise TypeError(f"mask must be a bool tensor, got dtype {mask.dtype}")
    if mask.ndim != 1:
        raise ValueError(f"mask must be 1D, got shape {tuple(mask.shape)}")
    if mask.shape[0] != vertices.shape[0]:
        raise ValueError(
            f"mask length must match number of vertices: {mask.shape[0]} != {vertices.shape[0]}"
        )
    if not bool(torch.any(mask).detach().cpu().item()):
        raise ValueError("mask selects no vertices")


def vertices_axis_le(
    vertices: torch.Tensor,
    *,
    axis: int,
    value: TensorLike,
    eps: TensorLike = 1e-8,
) -> torch.Tensor:
    """Select vertices whose coordinate on ``axis`` is <= ``value``.

    Axis follows normal vertex coordinate order: ``0`` is x, ``1`` is y, and
    ``2`` is z for 3D meshes. This is the generic version behind convenience
    helpers such as ``vertices_y_le``.
    """
    _validate_vertices(vertices)
    coordinate_axis = _validate_axis(vertices, axis)
    limit = _as_tensor_like(value, vertices)
    tolerance = _as_tensor_like(eps, vertices)
    return vertices[:, coordinate_axis] <= limit + tolerance


def vertices_y_le(
    vertices: torch.Tensor,
    y_max: TensorLike,
    *,
    eps: TensorLike = 1e-8,
) -> torch.Tensor:
    """Select vertices whose y coordinate is less than or equal to ``y_max``.

    This is a convenience wrapper for ``vertices_axis_le(vertices, axis=1, ...)``.
    Use ``vertices_axis_le`` directly when the selected coordinate is not y.
    """
    return vertices_axis_le(vertices, axis=1, value=y_max, eps=eps)


def selected_axis_center(vertices: torch.Tensor, mask: torch.Tensor, *, axis: int) -> torch.Tensor:
    """Return the midpoint of the selected vertices along ``axis``.

    The result stays as a torch tensor, so it can be used inside a differentiable
    vertex map without leaving PyTorch.
    """
    _validate_vertices(vertices)
    _validate_mask(vertices, mask)
    coordinate_axis = _validate_axis(vertices, axis)
    selected_coordinate = vertices[mask, coordinate_axis]
    return 0.5 * (torch.min(selected_coordinate) + torch.max(selected_coordinate))


def selected_x_center(vertices: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return the midpoint of the selected vertices in x.

    This is a convenience wrapper for ``selected_axis_center(..., axis=0)``.
    """
    return selected_axis_center(vertices, mask, axis=0)


def tan_half_angle_scale(
    angle_deg: torch.Tensor,
    reference_deg: TensorLike,
) -> torch.Tensor:
    """Return ``tan(angle / 2) / tan(reference / 2)`` for degree inputs.

    PyTorch trigonometric functions use radians, while many geometry parameters
    are easier for users to specify in degrees. This helper keeps that unit
    conversion in one place.
    """
    if not isinstance(angle_deg, torch.Tensor):
        raise TypeError(f"angle_deg must be a torch.Tensor, got {type(angle_deg).__name__}")

    reference = _as_tensor_like(reference_deg, angle_deg)
    angle_rad = angle_deg * torch.pi / 180.0
    reference_rad = reference * torch.pi / 180.0
    return torch.tan(angle_rad / 2.0) / torch.tan(reference_rad / 2.0)


def relative_scale(
    value: torch.Tensor,
    reference: TensorLike,
) -> torch.Tensor:
    """Return ``value / reference`` with ``reference`` matched to ``value``.

    This is the common "the initial design has scale 1" pattern used by many
    parameterized vertex maps.
    """
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"value must be a torch.Tensor, got {type(value).__name__}")
    return value / _as_tensor_like(reference, value)


def scale_selected_vertices(
    vertices: torch.Tensor,
    mask: torch.Tensor,
    *,
    x_scale: TensorLike = 1.0,
    y_scale: TensorLike = 1.0,
    x_center: TensorLike = 0.0,
    y_center: TensorLike = 0.0,
) -> torch.Tensor:
    """Scale selected vertices in x/y while leaving unselected vertices unchanged.

    ``x_center`` and ``y_center`` define the point that stays fixed during the
    scaling. The function returns a cloned tensor and does not mutate
    ``vertices`` in place.
    """
    _validate_vertices(vertices)
    _validate_mask(vertices, mask)

    sx = _as_tensor_like(x_scale, vertices)
    sy = _as_tensor_like(y_scale, vertices)
    cx = _as_tensor_like(x_center, vertices)
    cy = _as_tensor_like(y_center, vertices)

    out = vertices.clone()
    out[mask, 0] = cx + (vertices[mask, 0] - cx) * sx
    out[mask, 1] = cy + (vertices[mask, 1] - cy) * sy
    return out


def scale_selected_vertices_about_x_center(
    vertices: torch.Tensor,
    mask: torch.Tensor,
    *,
    x_scale: TensorLike = 1.0,
    y_scale: TensorLike = 1.0,
    y_center: TensorLike = 0.0,
) -> torch.Tensor:
    """Scale selected vertices around the selected vertices' x midpoint.

    This combines the common pair:

    ``center_x = selected_x_center(vertices, mask)``
    ``scale_selected_vertices(..., x_center=center_x)``

    The y center is still explicit because many examples scale height from the
    bottom boundary ``y=0``, while others may want a different fixed y level.
    """
    center_x = selected_x_center(vertices, mask)
    return scale_selected_vertices(
        vertices,
        mask,
        x_scale=x_scale,
        y_scale=y_scale,
        x_center=center_x,
        y_center=y_center,
    )


def scale_selected_vertices_about_axis_center(
    vertices: torch.Tensor,
    mask: torch.Tensor,
    *,
    center_axis: int,
    x_scale: TensorLike = 1.0,
    y_scale: TensorLike = 1.0,
    other_center: TensorLike = 0.0,
) -> torch.Tensor:
    """Scale selected x/y coordinates around the selected midpoint of one axis.

    ``center_axis=0`` keeps the selected x midpoint fixed while x/y are scaled.
    ``center_axis=1`` keeps the selected y midpoint fixed. This is useful for
    user vertex maps that want an axis-generic template, while still applying
    x/y scaling to a 2D mesh.
    """
    _validate_vertices(vertices)
    _validate_mask(vertices, mask)
    axis = _validate_axis(vertices, center_axis)
    if axis not in {0, 1}:
        raise ValueError("scale_selected_vertices_about_axis_center supports center_axis 0 or 1")

    center = selected_axis_center(vertices, mask, axis=axis)
    if axis == 0:
        return scale_selected_vertices(
            vertices,
            mask,
            x_scale=x_scale,
            y_scale=y_scale,
            x_center=center,
            y_center=other_center,
        )
    return scale_selected_vertices(
        vertices,
        mask,
        x_scale=x_scale,
        y_scale=y_scale,
        x_center=other_center,
        y_center=center,
    )
