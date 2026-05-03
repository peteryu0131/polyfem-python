"""Array-backed mesh helpers for guided config construction."""

from __future__ import annotations

from typing import Any

import numpy as np


def is_array_backed_body(section: Any) -> bool:
    return section.vertices is not None or section.cells is not None


def _coerce_body_vertices(vertices: Any, *, body_name: str) -> np.ndarray:
    vertices_np = np.asarray(vertices, dtype=np.float64)
    if vertices_np.ndim != 2:
        raise ValueError(
            f"array-backed body '{body_name}' requires vertices with shape (n_vertices, dim), "
            f"got {vertices_np.shape!r}"
        )
    if vertices_np.shape[0] == 0:
        raise ValueError(f"array-backed body '{body_name}' requires at least one vertex")
    return vertices_np


def _coerce_body_cells(cells: Any, *, body_name: str) -> np.ndarray:
    cells_np = np.asarray(cells, dtype=np.int32)
    if cells_np.ndim != 2:
        raise ValueError(
            f"array-backed body '{body_name}' requires cells/faces with shape (n_cells, k), "
            f"got {cells_np.shape!r}"
        )
    if cells_np.shape[0] == 0:
        raise ValueError(f"array-backed body '{body_name}' requires at least one cell/face")
    return cells_np


def build_guided_array_mesh_payload(
    array_bodies: list[tuple[Any, Any]],
) -> dict[str, np.ndarray]:
    merged_vertices: list[np.ndarray] = []
    merged_cells: list[np.ndarray] = []
    merged_body_ids: list[np.ndarray] = []

    expected_dim: int | None = None
    expected_cell_width: int | None = None
    vertex_offset = 0

    for section, body in array_bodies:
        vertices_np = _coerce_body_vertices(section.vertices, body_name=section.name)
        cells_np = _coerce_body_cells(section.cells, body_name=section.name)

        if expected_dim is None:
            expected_dim = int(vertices_np.shape[1])
        elif vertices_np.shape[1] != expected_dim:
            raise ValueError(
                "all array-backed bodies must use the same vertex dimension; "
                f"expected {expected_dim}, got {vertices_np.shape[1]} for body '{section.name}'"
            )

        if expected_cell_width is None:
            expected_cell_width = int(cells_np.shape[1])
        elif cells_np.shape[1] != expected_cell_width:
            raise ValueError(
                "all array-backed bodies must use the same cell width; "
                f"expected {expected_cell_width}, got {cells_np.shape[1]} for body '{section.name}'"
            )

        if np.any(cells_np < 0):
            raise ValueError(f"array-backed body '{section.name}' contains negative cell indices")
        if int(cells_np.max()) >= int(vertices_np.shape[0]):
            raise ValueError(
                f"array-backed body '{section.name}' has cell indices outside its vertex range"
            )

        merged_vertices.append(vertices_np)
        merged_cells.append(cells_np + vertex_offset)
        merged_body_ids.append(np.full(cells_np.shape[0], body.volume_id, dtype=np.int32))
        vertex_offset += int(vertices_np.shape[0])

    return {
        "vertices": np.vstack(merged_vertices),
        "cells": np.vstack(merged_cells),
        "body_ids": np.concatenate(merged_body_ids),
    }
