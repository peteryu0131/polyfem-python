"""Lightweight matplotlib visualization helpers for PolyFEM results.

These helpers intentionally avoid VTU/ParaView.  They render the native or
in-memory rollout mesh directly to PNG files, which is useful for quick paper
figures, notebooks, and remote runs where opening ParaView is inconvenient.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def _require_matplotlib():
    cache_dir = Path(tempfile.gettempdir()) / "polyfempy_matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
    except ImportError as exc:
        raise ImportError(
            "polyfempy.visualization requires matplotlib. "
            "Install it with `pip install matplotlib`."
        ) from exc
    return plt, LineCollection


def _to_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray(value)
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _as_2d_points(points: Any) -> np.ndarray:
    arr = np.asarray(_to_numpy(points), dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"expected points with shape (n, dim>=2), got {arr.shape!r}")
    return np.ascontiguousarray(arr[:, :2])


def _cell_blocks(cells: Any) -> list[np.ndarray]:
    if cells is None:
        return []
    if isinstance(cells, np.ndarray):
        arr = np.asarray(cells, dtype=np.int64)
        return [arr] if arr.ndim == 2 else []
    if isinstance(cells, (list, tuple)):
        blocks = []
        for block in cells:
            if isinstance(block, (list, tuple)) and len(block) == 2 and isinstance(block[0], str):
                arr = np.asarray(_to_numpy(block[1]), dtype=np.int64)
            else:
                arr = np.asarray(_to_numpy(block), dtype=np.int64)
            if arr.ndim == 2 and arr.shape[1] >= 2 and arr.shape[0] > 0:
                blocks.append(arr)
        return blocks
    arr = np.asarray(_to_numpy(cells), dtype=np.int64)
    return [arr] if arr.ndim == 2 else []


def _mesh_edges(cells: Any, n_points: int) -> np.ndarray:
    edges = []
    for block in _cell_blocks(cells):
        if block.size == 0:
            continue
        block = block[:, [i for i in range(block.shape[1])]]
        if np.any(block < 0) or np.any(block >= n_points):
            continue
        for i in range(block.shape[1]):
            edges.append(block[:, [i, (i + 1) % block.shape[1]]])
    if not edges:
        return np.empty((0, 2), dtype=np.int64)
    all_edges = np.vstack(edges)
    all_edges = np.sort(all_edges, axis=1)
    return np.unique(all_edges, axis=0)


def _mesh_boundary_edges(cells: Any, n_points: int) -> np.ndarray:
    edges = []
    for block in _cell_blocks(cells):
        if block.size == 0:
            continue
        if np.any(block < 0) or np.any(block >= n_points):
            continue
        for i in range(block.shape[1]):
            edges.append(block[:, [i, (i + 1) % block.shape[1]]])
    if not edges:
        return np.empty((0, 2), dtype=np.int64)
    all_edges = np.vstack(edges)
    all_edges = np.sort(all_edges, axis=1)
    unique_edges, counts = np.unique(all_edges, axis=0, return_counts=True)
    return unique_edges[counts == 1]


def _selected_indices(n_items: int, selection: Any) -> list[int]:
    if selection is None or selection == "all":
        return list(range(n_items))
    if isinstance(selection, int):
        idx = selection if selection >= 0 else n_items + selection
        if idx < 0 or idx >= n_items:
            raise IndexError(f"frame index {selection} out of range for {n_items} frames")
        return [idx]
    if isinstance(selection, slice):
        return list(range(n_items))[selection]

    out = []
    for raw in selection:
        idx = int(raw)
        idx = idx if idx >= 0 else n_items + idx
        if idx < 0 or idx >= n_items:
            raise IndexError(f"frame index {raw} out of range for {n_items} frames")
        out.append(idx)
    return out


def _format_frame_name(prefix: str, frame_index: int) -> str:
    return f"{prefix}{int(frame_index):04d}.png"


def _plot_wire_mesh(
    vertices: np.ndarray,
    cells: Any,
    path: Path,
    *,
    title: str | None = None,
    reference_vertices: np.ndarray | None = None,
    color: str = "#1f77b4",
    reference_color: str = "0.75",
    linewidth: float = 0.25,
    reference_linewidth: float = 0.18,
    boundary_only: bool = False,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    figsize: tuple[float, float] = (6.0, 6.0),
    dpi: int = 180,
) -> Path:
    plt, LineCollection = _require_matplotlib()
    vertices_2d = _as_2d_points(vertices)
    if boundary_only:
        edges = _mesh_boundary_edges(cells, len(vertices_2d))
    else:
        edges = _mesh_edges(cells, len(vertices_2d))
    if edges.size == 0:
        raise ValueError("cannot plot mesh: no valid cell edges found")

    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize, dpi=int(dpi))
    if reference_vertices is not None:
        ref_2d = _as_2d_points(reference_vertices)
        if ref_2d.shape == vertices_2d.shape:
            ax.add_collection(
                LineCollection(
                    ref_2d[edges],
                    colors=reference_color,
                    linewidths=reference_linewidth,
                    alpha=0.8,
                )
            )
    ax.add_collection(
        LineCollection(
            vertices_2d[edges],
            colors=color,
            linewidths=linewidth,
            alpha=0.95,
        )
    )
    ax.set_aspect("equal", adjustable="box")
    if xlim is None:
        x_pad = max(1e-9, 0.04 * float(np.ptp(vertices_2d[:, 0])))
        ax.set_xlim(float(np.min(vertices_2d[:, 0]) - x_pad), float(np.max(vertices_2d[:, 0]) + x_pad))
    else:
        ax.set_xlim(*xlim)
    if ylim is None:
        y_pad = max(1e-9, 0.04 * float(np.ptp(vertices_2d[:, 1])))
        ax.set_ylim(float(np.min(vertices_2d[:, 1]) - y_pad), float(np.max(vertices_2d[:, 1]) + y_pad))
    else:
        ax.set_ylim(*ylim)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9)
    fig.tight_layout(pad=0.02)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return path


def save_mesh_png(
    vertices: Any,
    cells: Any,
    path: str | Path,
    *,
    title: str | None = None,
    color: str = "#1f77b4",
    linewidth: float = 0.25,
    boundary_only: bool = False,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    figsize: tuple[float, float] = (6.0, 6.0),
    dpi: int = 180,
) -> Path:
    """Save a front-view wireframe PNG for a mesh.

    The first two vertex coordinates are used as the image plane.  This is a
    lightweight alternative to VTU/ParaView for quick inspection.
    """
    return _plot_wire_mesh(
        _as_2d_points(vertices),
        cells,
        Path(path),
        title=title,
        color=color,
        linewidth=linewidth,
        boundary_only=boundary_only,
        xlim=xlim,
        ylim=ylim,
        figsize=figsize,
        dpi=dpi,
    )


def save_mesh_boundary_png(
    vertices: Any,
    cells: Any,
    path: str | Path,
    *,
    title: str | None = None,
    color: str = "#1f77b4",
    linewidth: float = 0.8,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    figsize: tuple[float, float] = (6.0, 6.0),
    dpi: int = 180,
) -> Path:
    """Save a readable boundary-only PNG for a dense 2D mesh.

    This renders exterior and hole boundaries instead of every finite-element
    edge, which is usually the right full-view figure for dense generated
    meshes.
    """
    return save_mesh_png(
        vertices,
        cells,
        path,
        title=title,
        color=color,
        linewidth=linewidth,
        boundary_only=True,
        xlim=xlim,
        ylim=ylim,
        figsize=figsize,
        dpi=dpi,
    )


def _result_rollout(result: Any) -> tuple[np.ndarray, Any, np.ndarray, list[int]]:
    history = getattr(result, "history", None)
    if history is not None and getattr(history, "available", False):
        points = _to_numpy(getattr(history, "points", None))
        u = _to_numpy(getattr(history, "u", None))
        connectivity = _to_numpy(getattr(history, "connectivity", None))
        if points.ndim == 2 and points.shape[1] >= 2 and u.ndim == 3 and u.shape[1] == points.shape[0]:
            return _as_2d_points(points), [("polygon", connectivity)], u[:, :, :2], list(range(u.shape[0]))

    vertices = _as_2d_points(getattr(result, "vertices", None))
    cells = getattr(result, "cells", None)
    u = _to_numpy(getattr(result, "u", None))
    if u.ndim == 1 and u.size == vertices.size:
        displacements = u.reshape(1, *vertices.shape)
    elif u.ndim == 2 and u.shape == vertices.shape:
        displacements = u.reshape(1, *vertices.shape)
    elif u.ndim == 2 and u.shape[0] == vertices.size:
        displacements = np.stack(
            [u[:, i].reshape(vertices.shape) for i in range(u.shape[1])],
            axis=0,
        )
    elif u.ndim == 3 and u.shape[1:] == vertices.shape:
        displacements = u
    else:
        raise ValueError(
            "cannot infer displacement rollout from result: "
            f"vertices shape {vertices.shape!r}, u shape {u.shape!r}"
        )
    return vertices, cells, displacements[:, :, :2], list(range(displacements.shape[0]))


def save_deformed_png(
    result: Any,
    path: str | Path,
    *,
    frame: int = -1,
    displacement_scale: float = 1.0,
    show_reference: bool = True,
    title: str | None = None,
    color: str = "#d62728",
    reference_color: str = "0.75",
    linewidth: float = 0.25,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    figsize: tuple[float, float] = (6.0, 6.0),
    dpi: int = 180,
) -> Path:
    """Save one deformed forward-simulation frame as a PNG.

    The plotted coordinates are ``vertices + displacement_scale * u[frame]``.
    """
    vertices, cells, displacements, _frame_numbers = _result_rollout(result)
    indices = _selected_indices(displacements.shape[0], frame)
    idx = indices[0]
    deformed = vertices + float(displacement_scale) * displacements[idx]
    return _plot_wire_mesh(
        deformed,
        cells,
        Path(path),
        title=title,
        reference_vertices=vertices if show_reference else None,
        color=color,
        reference_color=reference_color,
        linewidth=linewidth,
        xlim=xlim,
        ylim=ylim,
        figsize=figsize,
        dpi=dpi,
    )


def save_forward_rollout_pngs(
    result: Any,
    output_dir: str | Path,
    *,
    frames: Any = "all",
    displacement_scale: float = 1.0,
    prefix: str = "frame_",
    show_reference: bool = True,
    title_prefix: str | None = None,
    color: str = "#d62728",
    reference_color: str = "0.75",
    linewidth: float = 0.25,
    figsize: tuple[float, float] = (6.0, 6.0),
    dpi: int = 180,
) -> list[Path]:
    """Save a forward rollout as a sequence of front-view PNG frames.

    ``frames`` may be ``"all"``, a single integer, a slice, or a list of frame
    indices.  The function prefers ``result.history`` when available, so it can
    render every transient time step without exporting VTU files.
    """
    vertices, cells, displacements, frame_numbers = _result_rollout(result)
    selected = _selected_indices(displacements.shape[0], frames)
    if not selected:
        return []

    all_deformed = vertices[None, :, :] + float(displacement_scale) * displacements[selected]
    stack = np.concatenate([vertices[None, :, :], all_deformed], axis=0)
    x_pad = max(1e-9, 0.04 * float(np.ptp(stack[:, :, 0])))
    y_pad = max(1e-9, 0.04 * float(np.ptp(stack[:, :, 1])))
    xlim = (float(np.min(stack[:, :, 0]) - x_pad), float(np.max(stack[:, :, 0]) + x_pad))
    ylim = (float(np.min(stack[:, :, 1]) - y_pad), float(np.max(stack[:, :, 1]) + y_pad))

    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for idx in selected:
        frame_number = frame_numbers[idx]
        title = None if title_prefix is None else f"{title_prefix} frame {frame_number}"
        written.append(
            _plot_wire_mesh(
                vertices + float(displacement_scale) * displacements[idx],
                cells,
                output_path / _format_frame_name(prefix, frame_number),
                title=title,
                reference_vertices=vertices if show_reference else None,
                color=color,
                reference_color=reference_color,
                linewidth=linewidth,
                xlim=xlim,
                ylim=ylim,
                figsize=figsize,
                dpi=dpi,
            )
        )
    return written


__all__ = [
    "save_deformed_png",
    "save_forward_rollout_pngs",
    "save_mesh_boundary_png",
    "save_mesh_png",
]
