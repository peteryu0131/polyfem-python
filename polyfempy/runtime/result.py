import numpy as np


class Result:
    """Minimal mesh plus backend solution container."""

    def __init__(self, vertices, cells, u, *, p=None, meta=None):
        self.vertices = np.ascontiguousarray(np.asarray(vertices))
        self.cells = np.ascontiguousarray(np.asarray(cells, dtype=np.int32))
        self.u = self._normalize_vector_field(u)
        self.p = None if p is None else np.ascontiguousarray(np.asarray(p))
        self.meta = {} if meta is None else dict(meta)

    def _normalize_vector_field(self, value):
        arr = np.ascontiguousarray(np.asarray(value))
        if self.vertices.ndim != 2:
            return arr

        vertices_count, dim = self.vertices.shape
        if vertices_count <= 0 or dim <= 0:
            return arr
        if arr.ndim == 1 and arr.size == vertices_count * dim:
            return np.ascontiguousarray(arr.reshape(vertices_count, dim))
        if arr.ndim == 2 and arr.shape == (vertices_count * dim, 1):
            return np.ascontiguousarray(arr.reshape(vertices_count, dim))
        return arr
