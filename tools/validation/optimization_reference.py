"""Read original optimization configurations for reference comparisons."""

import json
from pathlib import Path
from typing import Any

from differentiable_example._scene import OPTIMIZATIONS


def load(case: str, output: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source = OPTIMIZATIONS / case
    spec = json.loads((source / "run.json").read_text())
    if case == "initial-opt":
        spec = {"states": [{"path": "run.json"}, {"path": "target.json"}]}
    states = []
    for i, entry in enumerate(spec["states"]):
        path = source / entry["path"]
        cfg = json.loads(path.read_text())
        for geometry in cfg["geometry"]:
            mesh = (path.parent / geometry["mesh"]).resolve()
            if (
                case == "shape-stress-bbw-opt"
                and geometry["mesh"] == "../bar-2523.msh"
                and not mesh.exists()
            ):
                mesh = OPTIMIZATIONS.parent / "bar-2523.msh"
            if not mesh.is_file():
                raise FileNotFoundError(mesh)
            geometry["mesh"] = str(mesh)
        cfg["solver"]["max_threads"] = 1
        cfg["solver"].setdefault("linear", {})["solver"] = "Eigen::SimplicialLDLT"
        cfg["output"] = {
            "directory": str(output / f"forward{i}"),
            "json": "",
            "log": {"level": "warning"},
            "paraview": {"file_name": ""},
            "advanced": {"save_time_sequence": False},
        }
        states.append(cfg)
    return spec, states


def target_objective(spec):
    import polyfem.diff as diff

    (outer,) = spec["functionals"]
    inner = outer["static_objective"].copy()
    assert inner.pop("matching", "exact") == "exact"
    target = diff.objectives.create(inner.pop("type"), **inner)
    return diff.objectives.create(
        outer["type"],
        static_objective=target,
        **{k: v for k, v in outer.items() if k not in ("type", "static_objective")},
    )


def material_loss(spec, states):
    from differentiable_example.optimizations.material_and_contact.material_opt import build_loss

    (mapping,) = spec["variable_to_simulation"]
    assert mapping["state"] == 0 and [m["type"] for m in mapping["composition"]] == [
        "exp",
        "per-body-to-per-elem",
    ]
    return build_loss(states[0], states[1], target_objective(spec))
