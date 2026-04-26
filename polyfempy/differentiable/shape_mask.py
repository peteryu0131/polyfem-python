"""Helpers for selecting shape gradients by body id."""

# pyright: reportMissingImports=false

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _mesh_from_result(result: Any) -> Any:
    solver = getattr(result, "solver", None)
    if solver is None or not hasattr(solver, "mesh"):
        raise ValueError("result must keep its solver to build a body vertex mask")
    return solver.mesh()


def _mesh_cells(mesh: Any) -> np.ndarray:
    if not hasattr(mesh, "elements"):
        raise ValueError("mesh does not expose elements(), so body vertex masks are unavailable")
    cells = np.asarray(mesh.elements(), dtype=np.int64)
    if cells.ndim != 2:
        raise ValueError(f"expected mesh elements to be a 2D array, got shape {cells.shape}")
    return cells


def _mesh_body_ids(mesh: Any) -> np.ndarray:
    if not hasattr(mesh, "get_body_ids"):
        raise ValueError("mesh does not expose get_body_ids(), so body vertex masks are unavailable")
    body_ids = np.asarray(mesh.get_body_ids(), dtype=np.int32).reshape(-1)
    if body_ids.size == 0:
        raise ValueError("mesh body ids are empty")
    return body_ids


def _normalize_vertex_indices(indices: np.ndarray, *, n_vertices: int) -> np.ndarray:
    out = np.asarray(indices, dtype=np.int64).reshape(-1)
    if out.size == 0:
        return out
    if out.min() >= 1 and out.max() <= n_vertices:
        out = out - 1
    return out[(out >= 0) & (out < n_vertices)]


def body_vertex_mask(result: Any, *, body_id: int) -> torch.Tensor:
    """Return a boolean vertex mask for one body.

    ``body_id`` is the same id used by ``volume_selection`` in the config.
    In experiment 02, ``body_id=1`` is the lattice and ``body_id=2`` is the
    falling block.

    Common use:
        ``mask = body_vertex_mask(result, body_id=1)``
    """
    vertices = getattr(result, "vertices", None)
    if vertices is None:
        raise ValueError("result.vertices is missing; cannot build a shape vertex mask")

    n_vertices = int(vertices.shape[0])
    mesh = _mesh_from_result(result)
    body_ids = _mesh_body_ids(mesh)

    mask_np = np.zeros(n_vertices, dtype=bool)
    if body_ids.size == n_vertices:
        mask_np = body_ids == int(body_id)
    else:
        cells = _mesh_cells(mesh)
        if body_ids.size == cells.shape[0]:
            selected_cells = cells[body_ids == int(body_id)]
        elif body_ids.size == cells.shape[1]:
            selected_cells = cells[:, body_ids == int(body_id)].T
        else:
            raise ValueError(
                "mesh body ids do not match vertices or elements: "
                f"body_ids={body_ids.size}, vertices={n_vertices}, elements={cells.shape[0]}"
            )
        if selected_cells.size > 0:
            selected_vertices = _normalize_vertex_indices(
                selected_cells,
                n_vertices=n_vertices,
            )
            mask_np[selected_vertices] = True

    if not mask_np.any():
        raise ValueError(f"body_id={body_id} selected no vertices")
    return torch.as_tensor(mask_np, dtype=torch.bool, device=vertices.device)


def shape_gradient_for_body(result: Any, *, body_id: int) -> torch.Tensor:
    """Return ``result.shape_gradient`` with non-selected body vertices zeroed.

    This is useful when the objective is measured on one body but the design
    variable should be another body. For example, use a block von Mises loss
    while keeping only the gradient with respect to the lattice shape:

        ``shape_gradient_for_body(result, body_id=1)``
    """
    gradient = getattr(result, "shape_gradient", None)
    if gradient is None:
        raise ValueError("shape gradient is missing; call loss.backward() first")

    mask = body_vertex_mask(result, body_id=int(body_id))
    view_shape = (mask.shape[0],) + (1,) * (gradient.ndim - 1)
    return torch.where(mask.reshape(view_shape), gradient, torch.zeros_like(gradient))


__all__ = ["body_vertex_mask", "shape_gradient_for_body"]
