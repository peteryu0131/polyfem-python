from typing import Dict, Optional

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


class HistoryView:
    """Per-timestep view over the solution history populated by the C++ binding.

    Each field here is a **stacked** ndarray whose leading axis is time:

        result.history.u           # (n_steps, n_sampled, dim)
        result.history.vm          # (n_steps, n_sampled)           — von Mises per point
        result.history.vm_avg      # (n_steps, n_sampled)           — node-averaged vM
        result.history.stress      # (n_steps, n_sampled, dim*dim)  — tensor field
        result.history.pressure    # (n_steps, n_sampled, 1) or (n_steps, 0, 0)
        result.history.points      # (n_sampled, dim) — static; sampled mesh is rebuilt once
        result.history.connectivity  # (n_sampled_cells, k) — static
        result.history.body_ids    # (n_sampled,) int — static
        result.history.times       # (n_steps,) best-effort wall/simulation times if known
        result.history.names       # list of raw PolyFEM frame names
        len(result.history)        # number of frames

    Empty when the C++ backend didn't populate any frames (static solve, or
    ``save_time_sequence=False``). ``result.history.available`` is False in that
    case so callers can branch.
    """

    _VON_MISES_IS_EMPTY_NONE = "_vm_none"

    def __init__(self, frames=None, times=None):
        frames = list(frames or [])
        self._frames = frames
        self.names = [str(f.get("name", "")) for f in frames]

        # Static (time-invariant) sampled-mesh geometry: use the first frame.
        if frames:
            self.points = np.asarray(frames[0].get("points", np.empty((0, 0))))
            self.connectivity = np.asarray(
                frames[0].get("connectivity", np.empty((0, 0), dtype=np.int32))
            )
        else:
            self.points = np.empty((0, 0))
            self.connectivity = np.empty((0, 0), dtype=np.int32)

        # Stacked time-varying fields (first-axis = time).
        self.u = self._stack(frames, "solution")
        self.vm = self._stack_scalar(frames, "scalar_value")
        self.vm_avg = self._stack_scalar(frames, "scalar_value_avg")
        self.stress = self._stack(frames, "tensor_value")
        self.pressure = self._stack(frames, "pressure")
        self.body_ids = self._static_scalar(frames, "body_ids")

        # Times: if not supplied, fall back to step indices.
        if times is not None:
            self.times = np.asarray(times, dtype=np.float64)
        else:
            self.times = np.arange(len(frames), dtype=np.float64)

    @staticmethod
    def _stack(frames, key):
        arrays = []
        for f in frames:
            arr = np.asarray(f.get(key, np.empty((0, 0))))
            arrays.append(arr)
        if not arrays:
            return np.empty((0, 0, 0))
        try:
            return np.stack(arrays, axis=0)
        except Exception:
            # Per-step shapes differ (very rare — e.g. remeshing); fall back
            # to a list to avoid silent truncation.
            return arrays

    @staticmethod
    def _stack_scalar(frames, key):
        """Like _stack, but also squeezes a trailing singleton axis so
        (n_steps, n_sampled, 1) becomes (n_steps, n_sampled) — matches the
        same flattening convention we use for ``body_ids``."""
        arr = HistoryView._stack(frames, key)
        if isinstance(arr, np.ndarray) and arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        return arr

    @staticmethod
    def _static_scalar(frames, key):
        if not frames:
            return np.empty((0,), dtype=np.int32)
        arr = np.asarray(frames[0].get(key, np.empty((0,), dtype=np.int32)))
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr[:, 0]
        return arr

    @property
    def available(self) -> bool:
        """True when at least one frame was populated."""
        return len(self._frames) > 0

    def __len__(self) -> int:
        return len(self._frames)

    def __bool__(self) -> bool:
        return self.available

    def __getitem__(self, idx: int):
        return self._frames[idx]

    def field_by_body(self, name: str, body_ids) -> Dict[int, np.ndarray]:
        """Split a history field (e.g. ``"vm"`` / ``"u"``) by per-sample body_id.

        Returns ``{body_id: array with leading time axis preserved}``. Requires
        ``body_ids`` to be aligned with the history's sampled mesh (same row
        count as ``history.points``). Pass ``result.history.body_ids`` or
        ``result.body_ids`` directly.
        """
        arr = getattr(self, name, None)
        if arr is None:
            raise KeyError(f"history has no field {name!r}")
        body_ids = np.asarray(body_ids)
        arr_np = np.asarray(arr)
        if arr_np.ndim < 2 or arr_np.shape[1] != body_ids.shape[0]:
            raise ValueError(
                f"history.{name} has sampled-axis length {arr_np.shape[1] if arr_np.ndim >= 2 else 'N/A'} "
                f"but body_ids has length {body_ids.shape[0]} — can't split."
            )
        return {int(bid): arr_np[:, body_ids == bid] for bid in np.unique(body_ids)}


class Result:
    """Mesh + solution fields. meshio/VTK compatible.

    Three field namespaces, in lookup priority order:

    - ``point_data`` (per-vertex, aligned with ``vertices``)
    - ``cell_data`` (per-element, aligned with the ``cells`` blocks)
    - ``sampled_data`` (populated by the sampled-VTU fallback; **not** aligned
      with ``vertices`` / ``cells``, it is on a different probe mesh written
      out by the solver). ``to_meshio()`` does **not** emit ``sampled_data``
      because it would attach the array to the wrong mesh.

    Time-history access (when the C++ backend populates per-step frames):

    - ``result.history.u``   — (n_steps, n_sampled, dim)
    - ``result.history.vm``  — (n_steps, n_sampled) — von Mises per point per step
    - ``result.history.times`` — (n_steps,)
    - ``result.body_ids``    — static per-sample body id (if available)

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
        history=None,
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
        self.history = history if isinstance(history, HistoryView) else HistoryView(history)
        self.point_data = _PointDataProxy(self)
        self.cell_data = _CellDataProxy(self)
        self.sampled_data = _SampledDataProxy(self)
        self.fields = _MergedFieldsView(self._point_data, self._cell_data, self._sampled_data)

    @property
    def body_ids(self):
        """Per-sample body ids, if available.

        Priority:
        1. sampled_data fallback path (older VTU-based route)
        2. history.body_ids (new in-memory SolutionFrame route)
        """
        arr = self.field("body_ids")
        if arr is not None:
            return arr
        if getattr(self, "history", None) is not None and self.history.available:
            if getattr(self.history, "body_ids", None) is not None and self.history.body_ids.size > 0:
                return self.history.body_ids
        return None

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

    def point_field(self, name, default=None):
        """Return a native point field without falling through to other namespaces."""
        return self._point_data.get(name, default)

    def cell_field(self, name, default=None):
        """Return a native cell field without falling through to other namespaces."""
        return self._cell_data.get(name, default)

    def sampled_field(self, name, default=None):
        """Return a sampled/probe-mesh field without falling through to native data."""
        return self._sampled_data.get(name, default)

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

    def field_by_body(self, name: str) -> Dict[int, np.ndarray]:
        """Split a per-point field into chunks keyed by ``body_id``.

        Useful when the sampled-VTU fallback has populated ``body_ids`` on a
        multi-body mesh (e.g. ``volume_selection=1`` for a lattice and
        ``volume_selection=2`` for an impactor block): this returns, for each
        distinct body id, the rows of ``name`` that belong to that body::

            stress_per_body = result.field_by_body("stress")
            for bid, arr in stress_per_body.items():
                print(bid, np.abs(arr).max())

        Requires ``result.field("body_ids")`` to exist — currently populated
        only on the sampled mesh by the fallback, so ``name`` must itself be
        a sampled-mesh field (``stress`` / ``von_mises`` / ``von_mises_avg``).
        Splitting a native-mesh field like ``u`` raises ``ValueError`` because
        the native mesh has no body_ids mapping today.

        Raises:
            RuntimeError: ``body_ids`` is not available. Enable the
                sampled-VTU fallback by setting
                ``output.fallback.sampled_vtu`` to ``"auto"`` or ``"always"``.
            KeyError: ``name`` is not a known field on this result.
            ValueError: ``name`` and ``body_ids`` have mismatched row counts
                (native vs sampled mesh).
        """
        body = self.field("body_ids")
        if body is None:
            raise RuntimeError(
                "Cannot split by body: body_ids is not available on this "
                "Result. Enable the sampled-VTU fallback "
                "(output.fallback.sampled_vtu='auto' or 'always') so body_ids "
                "is populated, or supply it via set_sampled_field('body_ids', "
                "arr) before calling field_by_body()."
            )

        arr = self.field(name)
        if arr is None:
            raise KeyError(f"Field {name!r} is not present on this result")

        body = np.asarray(body)
        arr = np.asarray(arr)
        if arr.shape[0] != body.shape[0]:
            raise ValueError(
                f"Cannot split {name!r} (rows={arr.shape[0]}) by body_ids "
                f"(rows={body.shape[0]}): lengths do not match. body_ids is "
                f"currently stored on the sampled mesh; splitting a native-"
                f"mesh field like {name!r} would require a native-mesh "
                f"body_id map, which is not populated."
            )

        return {int(bid): arr[body == bid] for bid in np.unique(body)}

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
        return sorted(
            set(self._point_data)
            | set(self._cell_data)
            | set(self._sampled_data)
        )

    def available_fields(self):
        """Return field names grouped by namespace for user-facing introspection."""
        return {
            "point_data": sorted(self._point_data),
            "cell_data": sorted(self._cell_data),
            "sampled_data": sorted(self._sampled_data),
        }

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

    def report(self, **kwargs):
        """Structured, human-oriented summary for CLI/reporting use."""
        from .report import summarize_result

        return summarize_result(self, **kwargs)

    def format_summary(self, **kwargs) -> str:
        """Return a compact multi-line summary string for this result."""
        from .report import format_result_summary

        return format_result_summary(self, **kwargs)

    def history_bundle(self, **kwargs):
        """Structured per-step history bundle for reporting/training use."""
        from .report import summarize_history_bundle

        return summarize_history_bundle(self, **kwargs)

    def format_history_bundle_txt(self, **kwargs) -> str:
        """Return a TSV-style text bundle for ``result.history``."""
        from .report import format_history_bundle_txt

        return format_history_bundle_txt(self, **kwargs)

    def write_history_bundle_txt(self, path, **kwargs):
        """Write a TSV-style history bundle to ``path``."""
        from .report import write_history_bundle_txt

        return write_history_bundle_txt(self, path, **kwargs)

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
