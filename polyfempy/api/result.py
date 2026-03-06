from typing import Optional

import numpy as np


class _MergedFieldsView:
    """Merges point_data and cell_data for backward compat."""

    def __init__(self, point_data, cell_data):
        self._point_data = point_data
        self._cell_data = cell_data

    def get(self, name, default=None):
        if name in self._point_data:
            return self._point_data[name]
        return self._cell_data.get(name, default)

    def __getitem__(self, name):
        if name in self._point_data:
            return self._point_data[name]
        if name in self._cell_data:
            return self._cell_data[name]
        raise KeyError(name)

    def __contains__(self, name):
        return name in self._point_data or name in self._cell_data

    def keys(self):
        seen = set()
        for k in self._point_data:
            seen.add(k)
            yield k
        for k in self._cell_data:
            if k not in seen:
                yield k

    def items(self):
        for k in self.keys():
            yield k, self[k]

    def __iter__(self):
        return self.keys()

    def __len__(self):
        return len(set(self._point_data) | set(self._cell_data))


class Result:
    """Mesh + solution fields. point_data (per-vertex), cell_data (per-element), meshio/VTK compatible."""

    def __init__(self, backend, vertices, cells, fields=None, point_data=None, cell_data=None, meta=None):
        self.backend = backend
        self.vertices = np.ascontiguousarray(np.asarray(vertices))
        self._cell_blocks = self._normalize_cells(cells, self.vertices)
        self.meta = {} if meta is None else dict(meta)
        if point_data is not None and cell_data is not None:
            self._point_data = {k: np.ascontiguousarray(np.asarray(v)) for k, v in point_data.items()}
            self._cell_data = {k: np.ascontiguousarray(np.asarray(v)) for k, v in cell_data.items()}
        else:
            self._point_data = {}
            self._cell_data = {}
            if fields:
                self._split_fields(dict(fields))
        self.point_data = _PointDataProxy(self)
        self.cell_data = _CellDataProxy(self)
        self.fields = _MergedFieldsView(self._point_data, self._cell_data)  # backward compat

    def _normalize_cells(self, cells, vertices):
        if isinstance(cells, (list, tuple)) and cells and isinstance(cells[0], (tuple, list)):
            return [
                (str(ct), np.ascontiguousarray(np.asarray(arr, dtype=np.int32)))
                for ct, arr in cells
            ]
        arr = np.ascontiguousarray(np.asarray(cells, dtype=np.int32))
        if arr.size == 0:
            return []
        ct = self._guess_cell_type(arr, vertices)
        return [(ct, arr)]

    def _split_fields(self, fields):
        """Split flat fields into point_data and cell_data by shape."""
        nv = self.vertices.shape[0] if self.vertices.ndim >= 1 else 0
        nc = self.n_cells
        for k, v in fields.items():
            a = np.ascontiguousarray(np.asarray(v))
            n = a.shape[0] if a.ndim >= 1 else 0
            if n == nv:
                self._point_data[k] = a
            elif n == nc:
                self._cell_data[k] = a
            else:
                self._point_data[k] = a

    @property
    def n_vertices(self):
        return self.vertices.shape[0]

    @property
    def n_cells(self):
        total = 0
        for _, arr in self._cell_blocks:
            total += arr.shape[0] if arr.ndim >= 1 else 0
        return total

    @property
    def cells(self):
        return self._cell_blocks

    @property
    def V(self):
        return self.vertices

    def _make_contiguous_inplace(self):
        self.vertices = np.ascontiguousarray(self.vertices)
        for i, (ct, arr) in enumerate(self._cell_blocks):
            self._cell_blocks[i] = (ct, np.ascontiguousarray(arr.astype(np.int32, copy=False)))
        for k, v in list(self._point_data.items()):
            self._point_data[k] = np.ascontiguousarray(np.asarray(v))
        for k, v in list(self._cell_data.items()):
            self._cell_data[k] = np.ascontiguousarray(np.asarray(v))

    def field(self, name):
        if name in self._point_data:
            return self._point_data[name]
        return self._cell_data.get(name)

    def set_field(self, name, value):
        arr = np.ascontiguousarray(np.asarray(value))
        nv, nc = self.n_vertices, self.n_cells
        n = arr.shape[0] if arr.ndim >= 1 else 0
        if n == nc and n != nv:
            self._cell_data[name] = arr
        else:
            self._point_data[name] = arr
        return self

    def remove_field(self, name):
        if name in self._point_data:
            del self._point_data[name]
        if name in self._cell_data:
            del self._cell_data[name]
        return self

    def as_numpy(self):
        self._make_contiguous_inplace()
        return self

    def to_backend(self, include_mesh=False):
        if self.backend == "numpy":
            return self
        from . import tensor as T

        for k, v in list(self._point_data.items()):
            self._point_data[k] = T.to_backend(v, self.backend)
        for k, v in list(self._cell_data.items()):
            self._cell_data[k] = T.to_backend(v, self.backend)
        if include_mesh:
            self.vertices = T.to_backend(self.vertices, self.backend)
            for i, (ct, arr) in enumerate(self._cell_blocks):
                self._cell_blocks[i] = (ct, T.to_backend(arr, self.backend))
        return self

    def magnitude(self, name, out_name=None, eps=0.0):
        arr = self.field(name)
        if arr is None:
            return self
        a = np.asarray(arr)
        if a.ndim == 1:
            mag = np.abs(a)
        else:
            mag = np.sqrt((a * a).sum(axis=-1) + float(eps))
        self.set_field(out_name or (name + "_norm"), mag)
        return self

    def write(self, path: str, file_format: Optional[str] = None):
        """Export via meshio. Needs meshio (pip install polyfempy[io])."""
        import importlib
        meshio_mod = importlib.import_module("meshio")
        mesh = self.to_meshio()
        meshio_mod.write(str(path), mesh, file_format=file_format)

    def to_vtk(self, path: str):
        try:
            self.write(path)
        except Exception:
            np.savez(
                path if path.endswith(".npz") else (path + ".npz"),
                vertices=self.vertices,
                **{f"point_{k}": v for k, v in self._point_data.items()},
                **{f"cell_{k}": v for k, v in self._cell_data.items()},
            )

    @classmethod
    def from_meshio(cls, mesh, backend: str = "numpy", meta: Optional[dict] = None):
        vertices = np.ascontiguousarray(mesh.points)
        cells = []
        for block in mesh.cells:
            ct = block.type if hasattr(block, "type") else block[0]
            arr = np.ascontiguousarray(
                block.data if hasattr(block, "data") else block[1],
                dtype=np.int32,
            )
            cells.append((str(ct), arr))
        point_data = {k: np.ascontiguousarray(np.asarray(v)) for k, v in mesh.point_data.items()}
        cell_data = {}
        for name, arr_list in mesh.cell_data.items():
            cell_data[name] = np.ascontiguousarray(np.asarray(arr_list[0]))
        return cls(
            backend=backend,
            vertices=vertices,
            cells=cells,
            point_data=point_data,
            cell_data=cell_data,
            meta=meta or {},
        )

    @classmethod
    def read(cls, path, backend: str = "numpy", meta: Optional[dict] = None):
        """Read mesh file via meshio. Supports VTU, VTK, Gmsh, OBJ, XDMF, etc."""
        import importlib
        meshio_mod = importlib.import_module("meshio")
        mesh = meshio_mod.read(str(path))
        return cls.from_meshio(mesh, backend=backend, meta=meta)

    def to_meshio(self):
        import importlib
        meshio_mod = importlib.import_module("meshio")
        Mesh = getattr(meshio_mod, "Mesh", None)
        if Mesh is None:
            meshio_mesh = importlib.import_module("meshio._mesh")
            Mesh = meshio_mesh.Mesh
        nv = self.n_vertices
        nc = self.n_cells
        point_data_out = {
            k: v for k, v in self._point_data.items()
            if v.shape[0] == nv
        }
        cell_data_out = {}
        if len(self._cell_blocks) == 1:
            for name, arr in self._cell_data.items():
                if arr.shape[0] == nc:
                    cell_data_out[name] = [arr]
        return Mesh(
            self.vertices,
            self._cell_blocks,
            point_data=point_data_out,
            cell_data=cell_data_out,
        )

    def field_names(self):
        return list(set(self._point_data) | set(self._cell_data))

    def summary(self):
        dim = self.vertices.shape[1] if self.vertices.ndim == 2 else "?"
        cells_shape = [(ct, tuple(arr.shape)) for ct, arr in self._cell_blocks]
        return {
            "backend": self.backend,
            "vertices": tuple(self.vertices.shape),
            "cells": cells_shape,
            "dim": dim,
            "point_data": {k: tuple(v.shape) for k, v in self._point_data.items()},
            "cell_data": {k: tuple(v.shape) for k, v in self._cell_data.items()},
        }

    @staticmethod
    def _guess_cell_type(cells, vertices):
        k = cells.shape[1] if cells.ndim == 2 else None
        dim = vertices.shape[1] if vertices.ndim == 2 else None
        if k == 2:
            return "line"
        if k == 3:
            return "triangle"
        if k == 4:
            return "quad" if dim == 2 else "tetra"
        if k == 8:
            return "hexahedron"
        if dim == 2:
            return "triangle" if k and k <= 3 else "quad"
        return "tetra"


class _PointDataProxy:

    def __init__(self, result):
        self._result = result

    def __getitem__(self, name):
        return self._result._point_data[name]

    def __setitem__(self, name, value):
        self._result._point_data[name] = np.ascontiguousarray(np.asarray(value))

    def __delitem__(self, name):
        del self._result._point_data[name]

    def get(self, name, default=None):
        return self._result._point_data.get(name, default)

    def __contains__(self, name):
        return name in self._result._point_data

    def keys(self):
        return self._result._point_data.keys()

    def items(self):
        return self._result._point_data.items()

    def __iter__(self):
        return iter(self._result._point_data)


class _CellDataProxy:

    def __init__(self, result):
        self._result = result

    def __getitem__(self, name):
        return self._result._cell_data[name]

    def __setitem__(self, name, value):
        self._result._cell_data[name] = np.ascontiguousarray(np.asarray(value))

    def __delitem__(self, name):
        del self._result._cell_data[name]

    def get(self, name, default=None):
        return self._result._cell_data.get(name, default)

    def __contains__(self, name):
        return name in self._result._cell_data

    def keys(self):
        return self._result._cell_data.keys()

    def items(self):
        return self._result._cell_data.items()

    def __iter__(self):
        return iter(self._result._cell_data)
