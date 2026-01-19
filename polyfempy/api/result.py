import numpy as np

class Result:
    """Container for mesh and solution fields.

    Stores NumPy arrays and can convert back to the original backend.
    """

    def __init__(self, backend, vertices, cells, fields=None, meta=None):
        """Initialize Result and normalize to NumPy/C-contiguous."""
        self.backend = backend
        self.vertices = np.asarray(vertices)
        self.cells = np.asarray(cells, dtype=np.int32)
        self.fields = {} if fields is None else dict(fields)
        self.meta = {} if meta is None else dict(meta)
        self._make_contiguous_inplace()

    def _make_contiguous_inplace(self):
        """Ensure arrays are C-contiguous and cells are int32."""
        self.vertices = np.ascontiguousarray(self.vertices)
        self.cells = np.ascontiguousarray(self.cells.astype(np.int32, copy=False))
        for k, v in list(self.fields.items()):
            self.fields[k] = np.ascontiguousarray(np.asarray(v))

    def field(self, name):
        """Get field by name, or None if not present."""
        return self.fields.get(name)

    def set_field(self, name, value):
        """Set field (normalized to NumPy, C-contiguous)."""
        self.fields[name] = np.ascontiguousarray(np.asarray(value))
        return self

    def remove_field(self, name):
        """Remove field if it exists."""
        if name in self.fields:
            del self.fields[name]
        return self

    def as_numpy(self):
        """Normalize to NumPy (idempotent)."""
        self._make_contiguous_inplace()
        return self

    def to_backend(self, include_mesh=False):
        """Convert fields (and optionally mesh) back to the original backend.

        This implements a �return-what-was-fed� behavior:
        if `backend` is 'torch' or 'jax', fields are converted via
        `polyfempy.api.tensor.to_backend`. If `backend` is 'numpy', this is a no-op.

        Args:
            include_mesh: If True, also convert `vertices` and `cells` back to the
                original backend. Defaults to False.

        Returns:
            self.

        Raises:
            ImportError: If the original backend requires optional deps (e.g. torch)
                that are not installed.
        """
        if self.backend == "numpy":
            return self
        from . import tensor as T  # lazy import, optional dependency

        for k, v in list(self.fields.items()):
            self.fields[k] = T.to_backend(v, self.backend)

        if include_mesh:
            self.vertices = T.to_backend(self.vertices, self.backend)
            self.cells = T.to_backend(self.cells, self.backend)

        return self

    def magnitude(self, name, out_name=None, eps=0.0):
        """Compute Euclidean norm of vector field (L2 norm, or abs for 1D)."""
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

    def to_vtk(self, path):
        """Export to VTK via meshio, fallback to NPZ."""
        cell_type = self._guess_cell_type(self.cells, self.vertices)
        try:
            import meshio
            mesh = meshio.Mesh(
                self.vertices,
                [(cell_type, self.cells)],
                point_data=self._point_fields(),
            )
            meshio.write(path, mesh)
        except Exception:
            np.savez(
                path if path.endswith(".npz") else (path + ".npz"),
                vertices=self.vertices,
                cells=self.cells,
                **self.fields,
            )

    def field_names(self):
        """List field names."""
        return list(self.fields.keys())

    def summary(self):
        """Return summary dict with shapes and metadata."""
        dim = self.vertices.shape[1] if self.vertices.ndim == 2 else "?"
        info = {
            "backend": self.backend,
            "vertices": tuple(self.vertices.shape),
            "cells": tuple(self.cells.shape),
            "dim": dim,
            "fields": {k: tuple(v.shape) for k, v in self.fields.items()},
        }
        return info

    def _point_fields(self):
        """Collect fields matching vertex count for point data."""
        out = {}
        n = self.vertices.shape[0] if self.vertices.ndim == 2 else None
        for k, v in self.fields.items():
            a = np.asarray(v)
            if n is not None and a.shape[0] == n:
                out[k] = a
        return out

    @staticmethod
    def _guess_cell_type(cells, vertices):
        """Guess meshio cell type from connectivity size and spatial dim."""
        k = cells.shape[1] if cells.ndim == 2 else None
        dim = vertices.shape[1] if vertices.ndim == 2 else None
        if k == 2:
            return "line"
        if k == 3:
            return "triangle"
        if k == 4:
            # 2D -> quad, 3D -> tetra
            return "quad" if dim == 2 else "tetra"
        if k == 8:
            return "hexahedron"
        # Fallback: prefer triangle/quad in 2D, otherwise tetra
        if dim == 2:
            return "triangle" if k and k <= 3 else "quad"
        return "tetra"
 