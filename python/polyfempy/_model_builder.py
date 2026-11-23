from __future__ import annotations

import json
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Literal, TypeVar, cast, overload

from ._selection_refs import SelectionPool, SelectionRef


_MISSING = object()
_Model = TypeVar("_Model", bound="ModelBuilder")


class ModelBuilder:
    def __init__(
        self,
        api_module,
        *,
        relationships=None,
        selection_pool=None,
        namespaces=None,
        selection_helper_rules=None,
    ):
        self._api = api_module
        manifest = _builder_manifest(api_module)
        groups = namespaces if namespaces is not None else manifest.get("namespaces", {})
        self._selection_pool = selection_pool or SelectionPool(groups)
        self._selection_refs = {}
        if relationships is None:
            relationships = manifest.get("relationships", [])
        self._relationships_by_api = {}
        for item in relationships:
            name = item["builder_api"]
            if name in self._relationships_by_api:
                raise ValueError(f"duplicate id relationship builder_api: {name!r}")
            self._relationships_by_api[name] = item
        self._config_consumer_sections = list(dict.fromkeys(
            item["consumer"].split(".", 1)[0].removesuffix("[*]") for item in self._relationships_by_api.values()
        ))
        if selection_helper_rules is None:
            selection_helper_rules = manifest.get("selection_rules", {})
        self._selection_helper_rules = dict(selection_helper_rules)
        self._mesh_by_uuid = {}
        self._automatic_volume_ids = set()
        self._rhs_only_volume_ids = {}
        self._consumer_lists = {}
        self._selection_backend_id_by_signature = {}
        self._implicit_surface_meshes = set()
        self._rhs = _MISSING
        self._config_sections = {}

    def body(self):
        return BodyHandle(self)

    def rhs(self, value):
        if self._consumer_lists.get(("boundary_conditions", "rhs")):
            raise TypeError("Use either model.rhs(...) or body.rhs(...), not both")
        if self._rhs is not _MISSING:
            raise TypeError("rhs already configured on this model")
        self._rhs = _copy_payload(value)
        return self

    def _register_body_mesh(self, payload):
        values = _payload_dict(payload)
        volume_selection = values.get("volume_selection")
        if values.get("is_obstacle") or (
            volume_selection is not None
            and (not isinstance(volume_selection, int) or volume_selection <= 0)
        ):
            raise ValueError("body.mesh requires a deformable mesh with a single volume group")
        body = self._item_geometry("mesh", **values)
        assert isinstance(body, BodyHandle)
        return body.mesh_uuid, body.volume_ref

    @overload
    def _item_geometry(self, factory_name: str, /, *, is_obstacle: Literal[False] = False,
                       volume_selection: None = None, **kwargs) -> BodyHandle: ...

    @overload
    def _item_geometry(self, factory_name: str, /, *, is_obstacle: Literal[False] = False,
                       volume_selection: int, **kwargs) -> BodyHandle | GeometryHandle: ...

    @overload
    def _item_geometry(self, factory_name: str, /, *, is_obstacle: Literal[True],
                       surface_selection: None = None, **kwargs) -> ObstacleHandle | None: ...

    @overload
    def _item_geometry(self, factory_name: str, /, *, is_obstacle: Literal[True],
                       surface_selection: int, **kwargs) -> SelectionHandle | None: ...

    @overload
    def _item_geometry(self, factory_name: str, /, **kwargs) -> BodyHandle | GeometryHandle | ObstacleHandle | SelectionHandle | None: ...

    def _item_geometry(self, factory_name, /, *, same_volume_as=None, **kwargs):
        values = dict(kwargs)
        if same_volume_as is not None:
            if not isinstance(same_volume_as, BodyHandle) or same_volume_as._model is not self:
                raise ValueError("same_volume_as requires a body from this model")
            if "volume_selection" in values or values.get("is_obstacle"):
                raise ValueError("Specify one volume group on a deformable geometry")
            values["volume_selection"] = same_volume_as._require_mesh().backend_id

        geometry_object = getattr(self._api, factory_name)(**values)
        self._reserve_geometry_ids(_payload_dict(geometry_object), shared_volume=same_volume_as is not None)
        if same_volume_as is not None:
            same_volume_as._bind_volume()
        mesh_uuid = str(uuid.uuid4())
        self._mesh_by_uuid[mesh_uuid] = geometry_object
        if values.get("is_obstacle"):
            surface_selection = values.get("surface_selection")
            if surface_selection is None and factory_name != "mesh_sequence":
                return ObstacleHandle(self, geometry_object)
            if not isinstance(surface_selection, int) or surface_selection <= 0:
                return None
            ref = self._allocate_selection("surface", backend_id=surface_selection)
            return SelectionHandle(self, ref)

        volume_selection = values.get("volume_selection")
        if volume_selection is not None and (
            not isinstance(volume_selection, int) or volume_selection <= 0
        ):
            return GeometryHandle(self, mesh_uuid)
        # Reserve an ID, but preserve mesh tags until a body-level setting needs it.
        volume_ref = self._allocate_selection(
            "volume", backend_id=volume_selection, mesh_uuid=mesh_uuid,
        )
        if volume_selection is None:
            self._automatic_volume_ids.add(volume_ref.backend_id)
        return BodyHandle(self, mesh_uuid, volume_ref)

    def _allocate_selection(self, namespace, **kwargs):
        ref = self._selection_pool.allocate(namespace, **kwargs)
        self._selection_refs[ref.uuid] = ref
        return ref

    def _reserve_geometry_ids(self, geometry, *, shared_volume):
        """Keep automatic IDs clear of IDs already present in geometry rules."""
        reservations = []
        for namespace in ("volume", "surface", "point", "curve"):
            value = geometry.get(namespace + "_selection")
            items = value if isinstance(value, list) else [value]
            for item in items:
                backend_id = item.get("id") if isinstance(item, dict) else item
                if not isinstance(backend_id, int) or isinstance(backend_id, bool) or backend_id < 1:
                    continue
                if namespace == "volume" and not shared_volume and backend_id in self._automatic_volume_ids:
                    raise ValueError(
                        f"volume ID {backend_id} is already assigned automatically to another body. "
                        "Choose a different ID or use same_volume_as to share its volume."
                    )
                reservations.append((namespace, backend_id))
        for namespace, backend_id in reservations:
            self._allocate_selection(namespace, backend_id=backend_id)

    def _section_geometry(self):
        geometry = []
        for mesh_uuid, mesh in self._mesh_by_uuid.items():
            item = _payload_dict(mesh)
            if mesh_uuid in self._rhs_only_volume_ids:
                item["volume_selection"] = self._rhs_only_volume_ids[mesh_uuid]
            geometry.append(item)
        return geometry

    @overload
    def _section_materials(self) -> list | dict: ...

    @overload
    def _section_materials(self: _Model, value: object) -> _Model: ...

    @overload
    def _section_materials(self: _Model, **kwargs) -> _Model | list | dict: ...

    def _section_materials(self: _Model, value=_MISSING, **kwargs) -> _Model | list | dict:
        """Set materials, or return the current materials when called without arguments."""
        if value is not _MISSING or kwargs:
            if value is not _MISSING and kwargs:
                raise TypeError("Use a material value or keyword fields, not both")
            if value is not _MISSING:
                items = value if isinstance(value, list) else [value]
                material = self._api.materials(items)
            else:
                material = self._api.materials(**kwargs)
            return self._set_consumer_section("materials", material)
        return cast("list | dict", self._model_section("materials"))

    @overload
    def _section_initial_conditions(self) -> dict: ...

    @overload
    def _section_initial_conditions(self: _Model, **kwargs) -> _Model | dict: ...

    def _section_initial_conditions(self: _Model, **kwargs) -> _Model | dict:
        if kwargs:
            value = self._api.initial_conditions(**kwargs)
            return self._set_consumer_section("initial_conditions", value)
        return cast(dict, self._model_section("initial_conditions"))

    def _section_space(self, **kwargs):
        if kwargs:
            return self._set_consumer_section("space", self._api.space(**kwargs))
        return self._model_section("space")

    def _section_boundary_conditions(self, **kwargs):
        """Save keyword settings and keep the legacy dictionary return value."""
        if kwargs:
            self._set_consumer_section("boundary_conditions", kwargs)
        return cast(dict, self._model_section("boundary_conditions"))

    def _set_consumer_section(self, name, value):
        payload = _as_dict_if_available(value)
        self._model_section(name, payload)
        return self._set_config_section(name, deepcopy(payload))

    def _model_section(self, name, explicit=_MISSING):
        """Combine explicit settings with body settings without discarding either."""
        bound = self._consumer_root_section(name)
        if explicit is _MISSING:
            explicit = self._config_sections.get(name, _MISSING)
        if explicit is _MISSING:
            return deepcopy(bound)
        if not bound:
            return deepcopy(explicit)
        if isinstance(explicit, list) and isinstance(bound, list):
            return deepcopy(explicit + bound)
        if not isinstance(explicit, dict) or not isinstance(bound, dict):
            raise ValueError("%s conflicts with body settings; use per-body lists" % name)
        section = deepcopy(explicit)
        for key, items in bound.items():
            if name == "boundary_conditions" and key == "rhs" and key in section:
                raise TypeError("Use either body.rhs(...) or explicit rhs, not both")
            if key in section and section[key] is not None:
                if not isinstance(section[key], list):
                    raise ValueError(
                        "%s.%s already has a non-list value" % (name, key)
                    )
                section[key] = section[key] + deepcopy(items)
            else:
                section[key] = deepcopy(items)
        return section

    def _set_config_section(self, name, value):
        self._config_sections[name] = value
        return self

    def config(self, **kwargs):
        # Older callers pass the dictionary returned by boundary_conditions().
        if (
            "boundary_conditions" in self._config_sections
            and kwargs.get("boundary_conditions") is not None
        ):
            if _payload_dict(kwargs["boundary_conditions"]) == self._section_boundary_conditions():
                kwargs.pop("boundary_conditions")
        duplicates = self._config_sections.keys() & kwargs.keys()
        if duplicates:
            raise TypeError("Configuration already set on model: %s" % ", ".join(sorted(duplicates)))
        values = dict(self._config_sections)
        for name in self._config_consumer_sections:
            if name in values and name != "boundary_conditions":
                values[name] = self._model_section(name)
        values.update(kwargs)
        rhs = values.pop("rhs", _MISSING)
        if self._rhs is not _MISSING:
            if rhs is not _MISSING:
                raise TypeError(
                    "Use either model.rhs(...) or config(rhs=...), not both"
                )
            rhs = self._rhs

        use_model_geometry = "geometry" not in values
        if use_model_geometry:
            values["geometry"] = self._section_geometry()

        explicit_boundary = values.pop("boundary_conditions", _MISSING)
        if explicit_boundary is _MISSING:
            boundary_kwargs = {}
        elif explicit_boundary is None:
            boundary_kwargs = {}
        else:
            boundary_kwargs = _payload_dict(explicit_boundary)

        if rhs is not _MISSING:
            if "rhs" in boundary_kwargs:
                raise TypeError(
                    "Use either rhs=... or boundary_conditions with rhs, not both"
            )
            boundary_kwargs["rhs"] = rhs

        for section in self._config_consumer_sections:
            if section == "boundary_conditions":
                continue
            if section in values:
                continue
            payload = self._consumer_root_section(section)
            if payload:
                values[section] = payload

        boundary_conditions = self._model_section("boundary_conditions", boundary_kwargs)
        if (
            boundary_conditions
            or explicit_boundary is not _MISSING
            or rhs is not _MISSING
        ):
            values["boundary_conditions"] = boundary_conditions

        if self._consumer_lists.get(("boundary_conditions", "rhs")):
            self._export_body_rhs(values, use_model_geometry)
        return self._api.config(**values)

    def _export_body_rhs(self, values, use_model_geometry):
        loads = values["boundary_conditions"]["rhs"]
        vector = None
        for entry in loads:
            value = entry.get("value")
            if entry.keys() != {"id", "value"} or not isinstance(value, (list, tuple)):
                return
            if not value or not all(type(component) in (int, float) for component in value):
                return
            if vector is None:
                vector = list(value)
            elif list(value) != vector:
                return

        geometry = values["geometry"]
        if not isinstance(geometry, (tuple, list)):
            return
        volume_ids = set()
        shared_ids = set()
        for mesh in geometry:
            item = _payload_dict(mesh)
            if item.get("is_obstacle"):
                continue
            volume_id = item.get("volume_selection")
            if type(volume_id) is not int or volume_id <= 0:
                return
            if volume_id in volume_ids:
                shared_ids.add(volume_id)
            volume_ids.add(volume_id)
        if volume_ids != {entry["id"] for entry in loads}:
            return

        values["boundary_conditions"]["rhs"] = vector
        if use_model_geometry and self._rhs_only_volume_ids:
            # Only omit tags introduced for the RHS itself. Raw config sections
            # can also refer to these IDs without going through a body handle.
            referenced_ids = shared_ids | self._other_volume_ids(values)
            for mesh_uuid, item in zip(self._mesh_by_uuid, geometry):
                if mesh_uuid in self._rhs_only_volume_ids and item["volume_selection"] not in referenced_ids:
                    del item["volume_selection"]

    def _other_volume_ids(self, values):
        ids = set()
        for relationship in self._relationships_by_api.values():
            if relationship["namespace"] != "volume" or relationship["builder_api"] == "body.rhs":
                continue
            path, id_field = _parse_list_item_consumer(relationship["consumer"])
            items = values
            for key in path:
                items = _as_dict_if_available(items)
                items = items.get(key) if isinstance(items, dict) else None
            items = _as_dict_if_available(items)
            for item in items if isinstance(items, list) else [items]:
                item = _as_dict_if_available(item)
                if isinstance(item, dict):
                    labels = item.get(id_field)
                    if not isinstance(labels, (list, tuple)):
                        labels = [labels]
                    ids.update(label for label in labels if type(label) is int)
        return ids

    def _shared_surface_id(self, id, same_as):
        if same_as is not None:
            if id is not None or not isinstance(same_as, SelectionHandle) or same_as._model is not self or same_as.selection_ref.namespace != "surface":
                raise ValueError("same_as requires a surface from this model")
            id = same_as.selection_ref.backend_id
        return {"id": id} if id is not None else {}

    def _whole_surface_method(self, mesh_uuid, name):
        builder_api = "surface." + name
        if not self.has_builder_api(builder_api):
            return None

        def bind_value(*args, **kwargs):
            if mesh_uuid is None:
                raise ValueError("body.mesh(...) must be called before configuring a boundary")
            value = _bind_payload_from_call(builder_api, args, kwargs)
            current = _payload_dict(self._mesh_by_uuid[mesh_uuid]).get("surface_selection")
            if current is not None and not isinstance(current, int):
                raise ValueError(
                    "This mesh already has a partial or file-based surface selection. "
                    "Apply the boundary condition to the selected surface."
                )
            ref = self._allocate_selection(
                "surface",
                backend_id=current if current is not None and current > 0 else None,
                mesh_uuid=mesh_uuid,
            )
            _set_payload_field(self._mesh_by_uuid[mesh_uuid], "surface_selection", ref.backend_id)
            surface = SelectionHandle(self, ref)
            self.bind(builder_api, ref, value)
            self._implicit_surface_meshes.add(mesh_uuid)
            return surface

        bind_value.__name__ = name
        return bind_value

    def _add_surface_selection(self, mesh_uuid, payload, *, append=True):
        return self._add_geometry_selection(
            mesh_uuid,
            namespace="surface",
            geometry_field="surface_selection",
            payload=payload,
            append=append,
        )

    def _add_geometry_selection(
        self,
        mesh_uuid,
        *,
        namespace,
        geometry_field,
        payload,
        append,
        geometry_class=None,
    ):
        values = dict(payload)
        backend_id = values.pop("id", None)
        if backend_id is None and not append and not values:
            current = _payload_dict(self._mesh_by_uuid[mesh_uuid]).get(geometry_field)
            if isinstance(current, int) and current > 0:
                backend_id = current
        signature = (mesh_uuid, _selection_signature(namespace, values))
        if backend_id is None and signature in self._selection_backend_id_by_signature:
            backend_id = self._selection_backend_id_by_signature[signature]
        selection_ref = self._allocate_selection(
            namespace,
            backend_id=backend_id,
            mesh_uuid=mesh_uuid,
        )
        if append or values:
            geometry_payload = {"id": selection_ref.backend_id, **values}
            if geometry_class:
                geometry_payload = _construct_api_class(self._api, geometry_class, geometry_payload) or geometry_payload
        else:
            geometry_payload = selection_ref.backend_id

        self._set_mesh_selection_field(
            mesh_uuid,
            geometry_field,
            geometry_payload,
            append=append,
        )
        self._selection_backend_id_by_signature.setdefault(
            signature,
            selection_ref.backend_id,
        )
        return SelectionHandle(self, selection_ref)

    def _set_mesh_selection_field(self, mesh_uuid, field, payload, *, append):
        if (
            field == "surface_selection"
            and mesh_uuid in self._implicit_surface_meshes
            and (append or not isinstance(payload, int))
        ):
            raise ValueError(
                "A whole-boundary condition is already bound to this mesh. "
                "Choose the partial surface before applying its boundary condition."
            )
        mesh_object = self._mesh_by_uuid[mesh_uuid]
        current = _payload_dict(mesh_object).get(field)

        if append:
            if current is None or current == []:
                next_value = [_copy_payload(payload)]
            elif isinstance(current, list):
                if payload in current:
                    return
                next_value = list(current) + [_copy_payload(payload)]
            else:
                raise ValueError("cannot append %s to existing non-list value" % field)
        else:
            if current is not None and current != []:
                if isinstance(current, int) and isinstance(payload, int) and current == payload:
                    return
                raise ValueError("%s already exists for this mesh" % field)
            next_value = payload

        _set_payload_field(mesh_object, field, next_value)

    def bind(self, builder_api, selection_ref, value):
        if self._selection_refs.get(selection_ref.uuid) != selection_ref:
            raise ValueError("selection must belong to this model")
        relationship = self._relationship_for(builder_api)
        if relationship["namespace"] != selection_ref.namespace:
            raise ValueError(
                "%s expects %s selection, got %s"
                % (builder_api, relationship["namespace"], selection_ref.namespace)
            )

        consumer_path, id_field = _parse_list_item_consumer(relationship["consumer"])
        payload = _payload_dict(value)
        payload[id_field] = selection_ref.backend_id
        if consumer_path == ("boundary_conditions", "rhs"):
            if self._rhs is not _MISSING:
                raise TypeError("Use either model.rhs(...) or body.rhs(...), not both")
            if any(item[id_field] == payload[id_field] for item in self._consumer_lists.get(consumer_path, ())):
                raise TypeError("rhs already configured for this body ID")
        items = self._consumer_lists.setdefault(consumer_path, [])
        if payload not in items:
            items.append(payload)
        return payload

    def _relationship_for(self, builder_api):
        relationship = self._relationships_by_api.get(builder_api)
        if relationship is None:
            raise ValueError("missing ID relationship for builder API %r" % builder_api)
        return relationship

    def has_builder_api(self, builder_api):
        return builder_api in self._relationships_by_api

    def has_selection_helper(self, method_name):
        return method_name in self._selection_helper_rules

    def add_selection_with_helper(self, mesh_uuid, method_name, payload, *, append=True):
        rule = self._selection_helper_rules[method_name]
        if "same_as" in payload:
            if rule["namespace"] != "surface":
                raise ValueError("same_as requires a surface")
            payload.update(self._shared_surface_id(payload.get("id"), payload.pop("same_as")))
        _validate_selection_helper_payload(method_name, payload, rule["allowed_fields"])
        return self._add_geometry_selection(
            mesh_uuid,
            namespace=rule["namespace"],
            geometry_field=rule["field"],
            payload=payload,
            append=append,
            geometry_class=None if append else rule.get("class_path"),
        )

    def _consumer_section(self, section_path):
        section = {}
        prefix_len = len(section_path)
        for consumer_path, items in self._consumer_lists.items():
            if (
                len(consumer_path) == prefix_len + 1
                and consumer_path[:prefix_len] == section_path
            ):
                section[consumer_path[-1]] = list(items)
        return section

    def _consumer_root_section(self, section):
        section_path = (section,)
        if section_path in self._consumer_lists:
            return list(self._consumer_lists[section_path])
        return self._consumer_section(section_path)


class GeometryHandle:
    def __init__(self, model, mesh_uuid):
        self._model = model
        self.mesh_uuid = mesh_uuid

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)

        boundary_method = self._model._whole_surface_method(self.mesh_uuid, name)
        if boundary_method is not None:
            return boundary_method
        selection_helper = self._dynamic_selection_helper(name)
        if selection_helper is not None:
            return selection_helper
        raise AttributeError(
            "%s has no configured selection helper %r"
            % (self.__class__.__name__, name)
        )

    def _dynamic_selection_helper(self, name):
        if not self._model.has_selection_helper(name):
            return None

        def add_selection(*args, **kwargs):
            if args:
                raise TypeError("%s accepts keyword fields only" % name)
            append = kwargs.pop("append", True)
            return self._model.add_selection_with_helper(
                self.mesh_uuid,
                name,
                kwargs,
                append=append,
            )

        add_selection.__name__ = name
        return add_selection

    def surface_all(self, *, id=None, same_as=None):
        payload = self._model._shared_surface_id(id, same_as)
        return self._model._add_surface_selection(
            self.mesh_uuid,
            payload,
            append=False,
        )


class BodyHandle:
    def __init__(self, model, mesh_uuid=None, volume_ref: SelectionRef | None = None):
        self._model = model
        self.mesh_uuid = mesh_uuid
        self.volume_ref = volume_ref

    def mesh(self, *args, **kwargs):
        if self.mesh_uuid is not None:
            raise ValueError("body.mesh(...) was already configured")
        payload = _bind_payload_from_call("body.mesh", args, kwargs)
        mesh_uuid, volume_ref = self._model._register_body_mesh(payload)
        self.mesh_uuid = mesh_uuid
        self.volume_ref = volume_ref
        return self

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)

        builder_api = "body.%s" % name
        if not self._model.has_builder_api(builder_api):
            boundary_method = self._model._whole_surface_method(self.mesh_uuid, name)
            if boundary_method is not None:
                return boundary_method
            selection_helper = self._dynamic_selection_helper(name)
            if selection_helper is not None:
                return selection_helper
            raise AttributeError(
                "%s has no configured builder API %r"
                % (self.__class__.__name__, builder_api)
            )

        def bind_value(*args, **kwargs):
            volume_ref = self._require_mesh()
            value = _bind_payload_from_call(builder_api, args, kwargs)
            self._model.bind(builder_api, volume_ref, value)
            if builder_api == "body.rhs":
                mesh = self._model._mesh_by_uuid[self.mesh_uuid]
                if "volume_selection" not in _payload_dict(mesh):
                    self._model._rhs_only_volume_ids[self.mesh_uuid] = volume_ref.backend_id
            else:
                self._bind_volume()
            return self

        bind_value.__name__ = name
        return bind_value

    def surface_from_file(self, file, *, tag=None):
        """Use a PolyFEM sideset. A tag is optional for a single group."""
        self._require_mesh()
        with Path(file).open(encoding="utf-8") as rows:
            tags = {int(row.split()[0]) for row in rows if row.strip()}
        if tag is None:
            if len(tags) != 1:
                raise ValueError("Specify a tag for a sideset with multiple boundary groups")
            tag = next(iter(tags))
        if tag not in tags or tag <= 0:
            raise ValueError("Expected a positive tag from the sideset")
        geometry = self._model._mesh_by_uuid[self.mesh_uuid]
        current = _payload_dict(geometry).get("surface_selection")
        payload = {"file": str(file)}
        if current != payload:
            selection = self._model._api.surface_selection_file(**payload)
            self._model._set_mesh_selection_field(self.mesh_uuid, "surface_selection", selection, append=False)
        for boundary_tag in tags:
            if boundary_tag > 0:
                self._model._allocate_selection("surface", backend_id=boundary_tag)
        ref = self._model._allocate_selection("surface", backend_id=tag, mesh_uuid=self.mesh_uuid)
        return SelectionHandle(self._model, ref)

    def _bind_volume(self):
        volume_ref = self._require_mesh()
        geometry = self._model._mesh_by_uuid[self.mesh_uuid]
        _set_payload_field(geometry, "volume_selection", volume_ref.backend_id)
        self._model._rhs_only_volume_ids.pop(self.mesh_uuid, None)

    def _require_mesh(self) -> SelectionRef:
        if self.mesh_uuid is None or self.volume_ref is None:
            raise ValueError("body.mesh(...) must be called before configuring body")
        return self.volume_ref

    def _dynamic_selection_helper(self, name):
        if not self._model.has_selection_helper(name):
            return None

        def add_selection(*args, **kwargs):
            self._require_mesh()
            if args:
                raise TypeError("%s accepts keyword fields only" % name)
            append = kwargs.pop("append", True)
            return self._model.add_selection_with_helper(
                self.mesh_uuid,
                name,
                kwargs,
                append=append,
            )

        add_selection.__name__ = name
        return add_selection

    def surface_all(self, *, id=None, same_as=None):
        self._require_mesh()
        payload = self._model._shared_surface_id(id, same_as)
        return self._model._add_surface_selection(
            self.mesh_uuid,
            payload,
            append=False,
        )


class ObstacleHandle:
    def __init__(self, model, geometry):
        self._model = model
        self._geometry = geometry
        self._surface = None

    def surface_all(self):
        if self._surface is None:
            ref = self._model._allocate_selection("surface")
            _set_payload_field(self._geometry, "surface_selection", ref.backend_id)
            self._surface = SelectionHandle(self._model, ref)
        return self._surface

    def __getattr__(self, name):
        if name.startswith("_") or not self._model.has_builder_api("surface." + name):
            raise AttributeError(name)
        return getattr(self.surface_all(), name)


class SelectionHandle:
    def __init__(self, model, selection_ref):
        self._model = model
        self.selection_ref = selection_ref

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)

        builder_api = "%s.%s" % (self.selection_ref.namespace, name)
        if not self._model.has_builder_api(builder_api):
            raise AttributeError(
                "%s has no configured builder API %r"
                % (self.__class__.__name__, builder_api)
            )

        def bind_value(*args, **kwargs):
            value = _bind_payload_from_call(builder_api, args, kwargs)
            self._model.bind(builder_api, self.selection_ref, value)
            return self

        bind_value.__name__ = name
        return bind_value


def _parse_list_item_consumer(consumer):
    parts = consumer.split(".")
    if len(parts) < 2 or parts[-1] != "id":
        raise ValueError("unsupported ID consumer path: %r" % consumer)

    list_path = []
    for part in parts[:-1]:
        if part.endswith("[*]"):
            list_path.append(part[:-3])
        else:
            list_path.append(part)

    return tuple(list_path), parts[-1]


def _bind_payload_from_call(builder_api, args, kwargs):
    if len(args) > 1:
        raise TypeError("%s accepts at most one payload object" % builder_api)
    if args and kwargs:
        raise TypeError(
            "%s accepts either a payload object or keyword fields, not both"
            % builder_api
        )
    if args:
        return args[0]
    if kwargs:
        return kwargs
    raise TypeError("%s requires a payload object or keyword fields" % builder_api)


def _as_dict_if_available(value: object):
    as_dict = getattr(value, "as_dict", None)
    return as_dict() if callable(as_dict) else value


def _payload_dict(value):
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "as_dict"):
        payload = value.as_dict()
        if not isinstance(payload, dict):
            raise TypeError("as_dict() must return a dict")
        return dict(payload)
    raise TypeError("expected dict or generated object with as_dict()")


def _copy_payload(value):
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, list):
        return [_copy_payload(item) for item in value]
    return value


def _selection_signature(namespace, values):
    return (
        namespace,
        tuple(
            (key, _freeze_value(value))
            for key, value in sorted(values.items())
        ),
    )


def _freeze_value(value):
    if isinstance(value, dict):
        return tuple(
            (key, _freeze_value(child))
            for key, child in sorted(value.items())
        )
    if isinstance(value, list):
        return tuple(_freeze_value(child) for child in value)
    return value


def _set_payload_field(value, field, field_value):
    if isinstance(value, dict):
        value[field] = field_value
        return
    if hasattr(value, field):
        setattr(value, field, field_value)
        return
    raise TypeError("expected dict or generated object with %s field" % field)


def _builder_manifest(api_module):
    module_file = getattr(api_module, "__file__", None)
    if module_file:
        path = Path(module_file).with_name("builder_api_manifest.json")
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _validate_selection_helper_payload(method_name, payload, allowed_fields):
    allowed = set(allowed_fields)
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise TypeError(
            "%s got unsupported field(s): %s"
            % (method_name, ", ".join(unknown))
        )


def _construct_api_class(api_module, class_path, payload):
    if not class_path:
        return None
    root = getattr(api_module, "Root", None)
    if root is None:
        return None

    cls = root
    parts = class_path.split(".")
    if not parts or parts[0] != "Root":
        return None
    for part in parts[1:]:
        cls = getattr(cls, part, None)
        if cls is None:
            return None
    return cls(**payload)
