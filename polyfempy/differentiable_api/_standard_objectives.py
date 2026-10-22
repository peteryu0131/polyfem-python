"""Schema-backed construction of standard objective payloads, without FEM solves.

Structural validation is not a statement of backend derivative support. Opaque
parametrizations and state-dependent dimensions remain backend responsibilities.
"""
import copy
import math
from collections.abc import Mapping

from ._objective_schema_data import FIELDS, OBJECTIVES


def _plain(value):
    if callable(getattr(value, "as_dict", None)):
        return _plain(value.as_dict())
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return copy.deepcopy(value)


def _validate(value, pointer):
    rules = FIELDS.get(pointer, [])
    if not rules:
        return
    kinds = {
        "int": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "float": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v),
        "string": lambda v: isinstance(v, str),
        "file": lambda v: isinstance(v, str),
        "bool": lambda v: isinstance(v, bool),
        "list": lambda v: isinstance(v, list),
        "object": lambda v: isinstance(v, dict),
    }
    matching = [r for r in rules if kinds.get(r["type"], lambda v: True)(value)]
    if not matching:
        raise TypeError(f"{pointer}: expected schema type(s) {[r['type'] for r in rules]}")
    def permitted(rule):
        return (("options" not in rule or value in rule["options"])
                and ("min" not in rule or value >= rule["min"])
                and ("max" not in rule or value <= rule["max"]))
    if not any(permitted(rule) for rule in matching):
        raise ValueError(f"{pointer}: value is outside schema options or bounds")
    if isinstance(value, list):
        for item in value:
            _validate(item, pointer + "/*")
    elif isinstance(value, dict):
        for key, item in value.items():
            _validate(item, pointer + "/" + key)


def payload(type_name, fields, *, defaults=True):
    from .objectives import _volume_selection

    if type_name not in OBJECTIVES:
        raise ValueError(f"Unknown standard objective {type_name!r}; local extensions use their explicit helpers")
    rule = OBJECTIVES[type_name]
    required = set(rule.get("required", []))
    allowed = required | set(rule.get("optional", []))
    values = dict(fields)
    # Convenience alias only when the schema identifies one unambiguous domain.
    if "selection" in values:
        choices = allowed & {"volume_selection", "surface_selection"}
        if len(choices) != 1:
            raise ValueError(f"{type_name}: selection has no unambiguous schema field")
        field = next(iter(choices))
        if field in values:
            raise ValueError(f"Pass selection or {field}, not both")
        selected = values.pop("selection")
        values[field] = [selected] if isinstance(selected, int) else selected
    unknown = set(values) - (allowed - {"type"})
    if unknown:
        raise ValueError(f"{type_name}: fields not in objective-spec.json: {sorted(unknown)}")
    for field in ("volume_selection", "surface_selection"):
        if field in values and values[field] is not None:
            value = values[field]
            # Preserve canonical numeric scalar/list payloads. Handles are conveniences.
            if not isinstance(value, (int, str, list, tuple)):
                if field == "surface_selection" and getattr(value, "volume_ref", None) is not None:
                    raise TypeError("surface_selection requires a boundary handle or boundary IDs, not a body")
                values[field] = list(_volume_selection(value))
            elif isinstance(value, (list, tuple)):
                if field == "surface_selection" and any(getattr(v, "volume_ref", None) is not None for v in value):
                    raise TypeError("surface_selection requires boundary handles, not bodies")
                values[field] = [v if isinstance(v, int) else _volume_selection(v)[0] for v in value]
    values = _plain(values)
    if defaults:
        # Single-state Python convenience, not a new backend default.
        if "state" in required:
            values.setdefault("state", 0)
        for field in allowed - {"type"}:
            for item in FIELDS.get("/" + field, []):
                if "default" in item and item["default"] != "skip":
                    values.setdefault(field, copy.deepcopy(item["default"]))
                    break
    missing = {name for name in required - {"type"} if name not in values or values[name] is None}
    if missing:
        raise ValueError(f"{type_name}: missing required fields {sorted(missing)}")
    for field, value in values.items():
        _validate(value, "/" + field)
    for field in ("objective", "static_objective"):
        if field not in values:
            continue
        def nested(item):
            if not isinstance(item, dict) or "type" not in item:
                raise ValueError(f"{field} requires an objective with a type")
            return payload(item["type"], {k: v for k, v in item.items() if k != "type"}, defaults=defaults)
        value = values[field]
        values[field] = [nested(v) for v in value] if isinstance(value, list) else nested(value)
    # The schema permits object/list unions, but the backend divide constructor
    # indexes exactly two operands. Reject an unsafe payload before native code.
    if type_name == "divide" and (not isinstance(values["objective"], list) or len(values["objective"]) != 2):
        raise ValueError("divide requires exactly two objectives")
    if type_name == "transient_integral" and values["static_objective"]["type"] == "transient_integral":
        raise ValueError("transient_integral requires a spatial objective")
    return {"type": type_name, **values}


def create(type_name, **fields):
    from .objectives import ObjectiveSpec

    value = payload(type_name, fields)
    value.pop("type")
    return ObjectiveSpec(
        type_name=type_name, state=value.pop("state", None),
        volume_selection=value.pop("volume_selection", None),
        power=value.pop("power", None), weight=value.pop("weight", None),
        print_energy=value.pop("print_energy", None), extra=tuple(value.items()),
    )
