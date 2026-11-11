"""Rebuild the fixed BBW setup used by the four 3D PyTorch examples."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from differentiable_example.optimizations._bbw import geometry_file
from .optimization_reference import load
from polyfempy import polyfempy as backend

from .compare_original_json import run_native, write_json


def selections(spec, config):
    # The optional geometry tool uses PolyFEM's node numbering and original BBW weights.
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_json(root / "state.json", config)
        mappings = []
        for mapping in spec["variable_to_simulation"]:
            mapping = mapping.copy()
            if "active_geometry_nodes" in mapping:
                selection = mapping["active_geometry_nodes"]
                kind = selection["type"]
                mapping["composite_map_type"] = kind
                mapping[
                    "surface_selection" if kind == "boundary" else "volume_selection"
                ] = selection["selection"]
            mappings.append(mapping)
        write_json(
            root / "opt.json",
            {
                **spec,
                "variable_to_simulation": mappings,
                "states": [{"path": str(root / "state.json")}],
                "output": {"directory": str(root)},
            },
        )
        try:
            run_native(
                "polyfem_reference_probe.exe",
                [root / "opt.json", root / "geometry.json", "--shape-setup"],
                root / "setup.log",
            )
        except RuntimeError as error:
            raise RuntimeError(
                (root / "setup.log").read_text(errors="replace")[-1800:]
            ) from error
        return sorted(
            json.loads((root / "geometry.json").read_text()),
            key=lambda group: group["part"],
        )


if __name__ == "__main__":
    for case in (
        "3d-bspline-shape-matching",
        "3d-bspline-shape-mesh-matching",
        "3d-shape-layer-thickness",
        "shape-stress-bbw-opt",
    ):
        with TemporaryDirectory() as tmp:
            spec, (config,) = load(case, Path(tmp))
            path = geometry_file(spec, config)
            boundary, interior = selections(spec, config)
            session = backend.DifferentiableSession()
            session.set_settings(config)
            path.parent.mkdir(exist_ok=True)
            np.savez_compressed(
                path,
                skin=boundary["nodes"],
                inside=np.array(interior["nodes"], dtype=np.int64),
                handles=boundary["handles"],
                weights=boundary["weights"],
                vertices=session.mesh_vertices(),
            )
            print(case, path.name, flush=True)
