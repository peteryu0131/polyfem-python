import numpy as np


class Result:
    """A unified container for mesh + point fields across numeric backends.

    Internally, all arrays are stored as NumPy (C-contiguous). When needed,
    fields (and optionally the mesh) can be converted back to the user's
    original backend (e.g., 'numpy' / 'torch' / 'jax').

    Attributes:
        backend: Name of the original backend ('numpy'|'torch'|'jax').
        vertices: Vertex coordinates, shape (N, dim), float32/float64.
        cells: Element connectivity, shape (M, k), int32.
        fields: Mapping from field name to array, e.g., {'u': (N, dim)} or {'p': (N,)}.
        meta: Free-form metadata (logs/timing/diagnostics).

    Notes:
        - Arrays are normalized to C-contiguous for robustness.
        - `cells` are cast to int32.
        - Only fields that match the number of vertices are exported as point data.
    """

    def __init__(self, backend, vertices, cells, fields=None, meta=None):
        """Initialize a Result and normalize data to NumPy/C-contiguous.

        Args:
            backend: Original backend name ('numpy', 'torch', or 'jax').
            vertices: Vertex array-like of shape (N, dim).
            cells: Connectivity array-like of shape (M, k). Will be int32.
            fields: Optional dict[str, array-like] of point fields.
            meta: Optional dict with miscellaneous metadata.

        Returns:
            None. The instance stores NumPy views/copies internally.
        """
        self.backend = backend
        self.vertices = np.asarray(vertices)
        self.cells = np.asarray(cells, dtype=np.int32)
        self.fields = {} if fields is None else dict(fields)
        self.meta = {} if meta is None else dict(meta)
        self._make_contiguous_inplace()

    def _make_contiguous_inplace(self):
        """Ensure internal arrays are C-contiguous (and `cells` are int32).

        This is a no-op if arrays are already in the desired layout/dtype.

        Args:
            None.

        Returns:
            None. Operates in-place on `vertices`, `cells`, and `fields`.
        """
        self.vertices = np.ascontiguousarray(self.vertices)
        self.cells = np.ascontiguousarray(self.cells.astype(np.int32, copy=False))
        for k, v in list(self.fields.items()):
            self.fields[k] = np.ascontiguousarray(np.asarray(v))

    def field(self, name):
        """Retrieve a field by name.

        Args:
            name: Field name.

        Returns:
            The stored NumPy array for the field, or None if not present.
        """
        return self.fields.get(name)

    def set_field(self, name, value):
        """Set/replace a field (normalized to NumPy, C-contiguous).

        Args:
            name: Field name.
            value: Array-like to store.

        Returns:
            self, to allow chaining.
        """
        self.fields[name] = np.ascontiguousarray(np.asarray(value))
        return self

    def remove_field(self, name):
        """Remove a field if it exists.

        Args:
            name: Field name.

        Returns:
            self, to allow chaining.
        """
        if name in self.fields:
            del self.fields[name]
        return self

    def as_numpy(self):
        """Normalize internal storage to NumPy (idempotent).

        Ensures all arrays are NumPy/C-contiguous. Useful as a no-op in
        NumPy-only pipelines to make intent explicit.

        Args:
            None.

        Returns:
            self.
        """
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
        """Create a scalar field as the Euclidean norm of a vector field.

        If the input field is 1D (shape (N,)), the absolute value is used.
        Otherwise, the L2 norm across the last axis is computed.

        Args:
            name: Input field name.
            out_name: Output field name. Defaults to "<name>_norm".
            eps: Optional small non-negative value added inside sqrt for
                numerical stability.

        Returns:
            self.

        Notes:
            - Expects per-vertex fields. Shapes like (N, d) are typical.
            - If the field does not exist, this is a no-op.
        """
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
        """Export to VTK via meshio if available; otherwise save as NPZ.

        The function attempts to guess a suitable `meshio` cell type from
        `(cells.shape[1], vertices.shape[1])`. If `meshio` is missing or an
        error occurs (e.g., unsupported topology), it falls back to saving
        an `.npz` that includes `vertices`, `cells`, and all fields.

        Supported topologies (by default):
            2D: k=3 -> triangle, k=4 -> quad
            3D: k=4 -> tetra,     k=8 -> hexahedron

        Args:
            path: Output path. If it ends with ".npz", NPZ is always written.

        Returns:
            None. Writes a file to `path` (or `path + ".npz"` on fallback).
        """
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
        """List stored field names.

        Args:
            None.

        Returns:
            A list of field names (list[str]).
        """
        return list(self.fields.keys())

    def summary(self):
        """Return a lightweight summary for debugging.

        Args:
            None.

        Returns:
            A dict with shapes and metadata, e.g.:
            {
              "backend": str,
              "vertices": (N, dim),
              "cells": (M, k),
              "dim": dim or "?",
              "fields": {name: shape_tuple, ...},
            }
        """
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
        """Collect fields that can be written as per-vertex point data.

        A field qualifies if `field.shape[0] == num_vertices`.

        Args:
            None.

        Returns:
            A dict mapping field names to NumPy arrays suitable for
            `meshio.Mesh(..., point_data=...)`.
        """
        out = {}
        n = self.vertices.shape[0] if self.vertices.ndim == 2 else None
        for k, v in self.fields.items():
            a = np.asarray(v)
            if n is not None and a.shape[0] == n:
                out[k] = a
        return out

    @staticmethod
    def _guess_cell_type(cells, vertices):
        """Guess a meshio cell type from connectivity size and spatial dim.

        Args:
            cells: Connectivity array, shape (M, k).
            vertices: Vertex array, shape (N, dim).

        Returns:
            A meshio cell type string, e.g. 'triangle', 'quad', 'tetra', 'hexahedron'.

        Notes:
            This is a heuristic; extend as needed for other topologies.
        """
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
 