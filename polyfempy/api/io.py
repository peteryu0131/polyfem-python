"""
Mesh I/O layer using meshio.

Provides read_mesh() for loading mesh files and converting to a normalized
in-memory structure. Requires meshio: pip install polyfempy[io] or pip install meshio.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np


@dataclass
class Mesh:
    """Normalized mesh structure for use with solve(vertices, cells, cfg).

    Attributes:
        vertices: (n_vertices, dim) array of point coordinates.
        cells: List of (cell_type, array) e.g. [("tetra", arr), ("triangle", arr)].
        point_data: Per-vertex fields dict (optional).
        cell_data: Per-element fields dict, key -> list of arrays (one per cell block).
    """

    vertices: np.ndarray
    cells: List[Tuple[str, np.ndarray]]
    point_data: dict = field(default_factory=dict)
    cell_data: dict = field(default_factory=dict)

    @property
    def n_vertices(self) -> int:
        return self.vertices.shape[0]

    @property
    def n_cells(self) -> int:
        return sum(arr.shape[0] for _, arr in self.cells)


def read_mesh(path: str | Path) -> Mesh:
    """Read a mesh file via meshio and convert to normalized Mesh.

    Supported formats include VTU, VTK, Gmsh, OBJ, XDMF, etc. (meshio-supported).

    Args:
        path: Path to mesh file.

    Returns:
        Mesh with vertices, cells (as list of (type, array)), point_data, cell_data.

    Raises:
        ImportError: If meshio is not installed (pip install meshio or polyfempy[io]).
        Exception: On read or conversion failure.
    """
    import importlib
    meshio_mod = importlib.import_module("meshio")
    mesh = meshio_mod.read(str(path))

    vertices = np.ascontiguousarray(mesh.points)
    cells = []
    for block in mesh.cells:
        ct = block.type if hasattr(block, "type") else block[0]
        data = block.data if hasattr(block, "data") else block[1]
        arr = np.ascontiguousarray(np.asarray(data), dtype=np.int32)
        cells.append((str(ct), arr))

    point_data = {k: np.ascontiguousarray(np.asarray(v)) for k, v in mesh.point_data.items()}
    # meshio cell_data: {name: [arr_per_block, ...]}
    cell_data = {}
    for name, arr_list in mesh.cell_data.items():
        cell_data[name] = [np.ascontiguousarray(np.asarray(a)) for a in arr_list]

    return Mesh(
        vertices=vertices,
        cells=cells,
        point_data=point_data,
        cell_data=cell_data,
    )
