from typing import Optional

import numpy as np


class _MergedFieldsView:
    """Priority-ordered read-only view over multiple field dicts.

    Earlier stores shadow later ones. Used to expose ``result.fields`` as a
    single-namespace dict-like that merges point_data, cell_data, and sampled
    data in that priority order.
    """

    def __init__(self, *stores):
        self._stores = tuple(stores)

    def get(self, name, default=None):
        for store in self._stores:
            if name in store:
                return store[name]
        return default

    def __getitem__(self, name):
        for store in self._stores:
            if name in store:
                return store[name]
        raise KeyError(name)

    def __contains__(self, name):
        return any(name in store for store in self._stores)

    def keys(self):
        seen = set()
        for store in self._stores:
            for k in store:
                if k not in seen:
                    seen.add(k)
                    yield k

    def items(self):
        for k in self.keys():
            yield k, self[k]

    def __iter__(self):
        return self.keys()

    def __len__(self):
        seen = set()
        for store in self._stores:
            seen.update(store)
        return len(seen)


class Result:
    """Mesh + solution fields. meshio/VTK compatible.

    Three field namespaces, in lookup priority order:

    - ``point_data`` (per-vertex, aligned with ``vertices``)
    - ``cell_data`` (per-element, aligned with the ``cells`` blocks)
    - ``sampled_data`` (populated by the sampled-VTU fallback; **not** aligned
      with ``vertices`` / ``cells``, it is on a different probe mesh written
      out by the solver). ``to_meshio()`` does **not** emit ``sampled_data``
      because it would attach the array to the wrong mesh.

    Shape contract (native path):
        ``vertices`` ``(n_vertices, dim)``, ``u`` ``(n_vertices, dim)``,
        ``stress`` / ``strain`` ``(n_vertices, 6)`` if present.

    Use ``.u / .p / .stress / .strain / .von_mises`` for common fields. Those
    properties look up point_data → cell_data → sampled_data, so user code
    keeps working even when the value came via fallback (check ``meta`` for a
    ``stress_source`` entry to tell the two apart).
    """

    def __init__(
        self,
        backend,
        vertices,
        cells,
        fields=None,
        point_data=None,
        cell_data=None,
        sampled_data=None,
        meta=None,
    ):
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
        self._sampled_data = (
            {k: np.ascontiguousarray(np.asarray(v)) for k, v in sampled_data.items()}
            if sampled_data
            else {}
        )
        self.point_data = _PointDataProxy(self)
        self.cell_data = _CellDataProxy(self)
        self.sampled_data = _SampledDataProxy(self)
        self.fields = _MergedFieldsView(self._point_data, self._cell_data, self._sampled_data)

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

    @property
    def u(self):
        """Displacement field, shape (n_vertices, dim). Same convention as DifferentiableResult.u."""
        return self.field("u")

    @property
    def p(self):
        """Pressure field if present (e.g. (n_vertices,) or pressure DOFs)."""
        return self.field("p")

    @property
    def stress(self):
        """Stress per-vertex (n_vertices, 6) in Voigt order, if present."""
        return self.field("stress")

    @property
    def strain(self):
        """Strain per-vertex (n_vertices, 6) in Voigt order, if present."""
        return self.field("strain")

    @staticmethod
    def _von_mises_from_stress_voigt(stress):
        """Compute von Mises from stress in Voigt form.

        Supported shapes:
        - (n, 6): [sxx, syy, szz, sxy, syz, szx]
        - (n, 3): [sxx, syy, sxy] (assume szz=syz=szx=0)
        """
        s = np.asarray(stress, dtype=np.float64)
        if s.ndim != 2 or s.shape[1] not in (3, 6):
            raise ValueError(f"Expected stress shape (n,3) or (n,6), got {s.shape}")

        if s.shape[1] == 3:
            sxx = s[:, 0]
            syy = s[:, 1]
            szz = np.zeros_like(sxx)
            sxy = s[:, 2]
            syz = np.zeros_like(sxx)
            szx = np.zeros_like(sxx)
        else:
            sxx = s[:, 0]
            syy = s[:, 1]
            szz = s[:, 2]
            sxy = s[:, 3]
            syz = s[:, 4]
            szx = s[:, 5]

        vm2 = 0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) + 3.0 * (
            sxy**2 + syz**2 + szx**2
        )
        vm2 = np.maximum(vm2, 0.0)
        return np.sqrt(vm2)

    def get_von_mises_numpy(self):
        """Return von Mises as a numpy array, without VTU I/O.

        Priority:
        1. Reuse an existing ``von_mises`` field if already present.
        2. Fall back to computing from ``stress`` when stress is available.
        """
        for name in ("von_mises", "von_mises_avg"):
            arr = self.field(name)
            if arr is not None:
                out = np.asarray(arr)
                if out.size > 0:
                    return out

        stress = self.stress
        if stress is None:
            return None
        return self._von_mises_from_stress_voigt(stress)

    def get_percentile_from_von_mises(self, q=95.0, *, method="linear"):
        """Compute a percentile of von Mises, if available."""
        vm = self.get_von_mises_numpy()
        if vm is None:
            return None
        if vm.size == 0:
            return float("nan")
        return float(np.percentile(vm, q, method=method))

    @property
    def von_mises(self):
        """von Mises stress if available, else ``None``."""
        return self.get_von_mises_numpy()

    def _make_contiguous_inplace(self):
        self.vertices = np.ascontiguousarray(self.vertices)
        for i, (ct, arr) in enumerate(self._cell_blocks):
            self._cell_blocks[i] = (ct, np.ascontiguousarray(arr.astype(np.int32, copy=False)))
        for k, v in list(self._point_data.items()):
            self._point_data[k] = np.ascontiguousarray(np.asarray(v))
        for k, v in list(self._cell_data.items()):
            self._cell_data[k] = np.ascontiguousarray(np.asarray(v))
        for k, v in list(self._sampled_data.items()):
            self._sampled_data[k] = np.ascontiguousarray(np.asarray(v))

    def field(self, name):
        """Look up a field by name across point / cell / sampled namespaces.

        Priority: point_data → cell_data → sampled_data. ``None`` if missing.
        """
        if name in self._point_data:
            return self._point_data[name]
        if name in self._cell_data:
            return self._cell_data[name]
        return self._sampled_data.get(name)

    def set_field(self, name, value):
        """Store a field aligned with the native mesh.

        Length == n_cells (and != n_vertices) → cell_data; otherwise point_data.
        Use ``set_sampled_field`` for fields that live on a different mesh
        (e.g. probe VTU output); this method is only for native-mesh data.
        """
        arr = np.ascontiguousarray(np.asarray(value))
        nv, nc = self.n_vertices, self.n_cells
        n = arr.shape[0] if arr.ndim >= 1 else 0
        if n == nc and n != nv:
            self._cell_data[name] = arr
        else:
            self._point_data[name] = arr
        return self

    def set_sampled_field(self, name, value):
        """Store a field that lives on a different (sampled / probe) mesh.

        Unlike ``set_field``, this never writes into ``point_data`` or
        ``cell_data`` regardless of array length. ``to_meshio()`` / ``write()``
        intentionally ignore ``sampled_data`` because attaching those values to
        the native mesh would be a lie.

        The field is still discoverable via ``result.field(name)``, so existing
        consumers of ``result.stress`` / ``result.von_mises`` keep working.
        """
        arr = np.ascontiguousarray(np.asarray(value))
        self._sampled_data[name] = arr
        return self

    def remove_field(self, name):
        if name in self._point_data:
            del self._point_data[name]
        if name in self._cell_data:
            del self._cell_data[name]
        if name in self._sampled_data:
            del self._sampled_data[name]
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
        for k, v in list(self._sampled_data.items()):
            self._sampled_data[k] = T.to_backend(v, self.backend)
        if include_mesh:
            self.vertices = T.to_backend(self.vertices, self.backend)
            for i, (ct, arr) in enumerate(self._cell_blocks):
                self._cell_blocks[i] = (ct, T.to_backend(arr, self.backend))
        return self

    def to_torch(self, include_mesh=True):
        """Convert all fields (and optionally mesh) to PyTorch and return self.

        Recommended usage for ML:
            r = solve(...)
            r.to_torch(include_mesh=True)
            u = r.u   # (n_vertices, dim)
        """
        from . import tensor as T
        self.backend = "torch"
        for k, v in list(self._point_data.items()):
            self._point_data[k] = T.to_backend(v, "torch")
        for k, v in list(self._cell_data.items()):
            self._cell_data[k] = T.to_backend(v, "torch")
        for k, v in list(self._sampled_data.items()):
            self._sampled_data[k] = T.to_backend(v, "torch")
        if include_mesh:
            self.vertices = T.to_backend(self.vertices, "torch")
            for i, (ct, arr) in enumerate(self._cell_blocks):
                self._cell_blocks[i] = (ct, T.to_backend(arr, "torch"))
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
                **{f"sampled_{k}": v for k, v in self._sampled_data.items()},
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

        # meshio stores cell_data as {name: [arr_per_block]}. We concatenate the
        # per-block arrays into one flat array so the internal representation
        # matches the rest of Result (a single ndarray per field name). The
        # reverse split happens in ``to_meshio()``.
        cell_data = {}
        for name, arr_list in mesh.cell_data.items():
            if not isinstance(arr_list, (list, tuple)) or len(arr_list) == 0:
                continue
            if len(arr_list) == 1:
                cell_data[name] = np.ascontiguousarray(np.asarray(arr_list[0]))
                continue
            try:
                concat = np.concatenate([np.asarray(a) for a in arr_list], axis=0)
                cell_data[name] = np.ascontiguousarray(concat)
            except (ValueError, TypeError):
                # Per-block arrays had incompatible shapes along the non-cell
                # axes. Fall back to the legacy behavior of keeping the first
                # block only so read() at least doesn't crash; round-trip
                # fidelity is lost in this rare case.
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

        # meshio requires cell_data as {name: [arr_per_block]}. For single-block
        # meshes that degenerates to ``[arr]``. For multi-block meshes we split
        # the flat per-field array (stored internally) along axis 0 back into
        # one sub-array per cell block, preserving the ordering established in
        # ``from_meshio``.
        cell_data_out = {}
        n_blocks = len(self._cell_blocks)
        if n_blocks == 1:
            for name, arr in self._cell_data.items():
                if arr.shape[0] == nc:
                    cell_data_out[name] = [arr]
        elif n_blocks > 1:
            block_sizes = [arr.shape[0] for _, arr in self._cell_blocks]
            offsets = np.cumsum([0] + block_sizes)
            for name, arr in self._cell_data.items():
                if arr.shape[0] != nc:
                    continue
                cell_data_out[name] = [
                    np.ascontiguousarray(arr[offsets[i]:offsets[i + 1]])
                    for i in range(n_blocks)
                ]

        return Mesh(
            self.vertices,
            self._cell_blocks,
            point_data=point_data_out,
            cell_data=cell_data_out,
        )

    def field_names(self):
        return list(
            set(self._point_data)
            | set(self._cell_data)
            | set(self._sampled_data)
        )

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
            "sampled_data": {k: tuple(v.shape) for k, v in self._sampled_data.items()},
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


class _SampledDataProxy:
    """Dict-like view over ``Result._sampled_data``.

    ``sampled_data`` holds fields that come from a mesh *other than* the
    native one (typically the sampled-VTU fallback probe mesh). It is kept
    separate from ``point_data`` / ``cell_data`` because ``to_meshio()``
    would otherwise attach the arrays to the wrong vertices / cells.
    """

    def __init__(self, result):
        self._result = result

    def __getitem__(self, name):
        return self._result._sampled_data[name]

    def __setitem__(self, name, value):
        self._result._sampled_data[name] = np.ascontiguousarray(np.asarray(value))

    def __delitem__(self, name):
        del self._result._sampled_data[name]

    def get(self, name, default=None):
        return self._result._sampled_data.get(name, default)

    def __contains__(self, name):
        return name in self._result._sampled_data

    def keys(self):
        return self._result._sampled_data.keys()

    def items(self):
        return self._result._sampled_data.items()

    def __iter__(self):
        return iter(self._result._sampled_data)
