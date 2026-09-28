from __future__ import annotations

from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
DIFFDATA_ROOT = ROOT / "differentiability-data"
OPT_SPEC_PATH = DIFFDATA_ROOT / "input" / "neohookean-stress-3d-opt.json"
MESH_PATH = DIFFDATA_ROOT / "bunny.msh"


def gmsh_vertices(path: Path, *, dimension: int = 3) -> torch.Tensor:
    lines = path.read_text(encoding="utf-8").splitlines()
    node_marker = lines.index("$Nodes")
    header = lines[node_marker + 1].split()
    vertices_by_tag = {}

    if len(header) == 1:
        node_count = int(header[0])
        for line in lines[node_marker + 2 : node_marker + 2 + node_count]:
            node_id, x, y, z = line.split()
            vertices_by_tag[int(node_id)] = [float(x), float(y), float(z)]
    elif len(header) == 4:
        block_count = int(header[0])
        cursor = node_marker + 2
        for _block in range(block_count):
            _entity_dim, _entity_tag, _parametric, block_node_count = (
                int(value) for value in lines[cursor].split()
            )
            cursor += 1
            tags = [int(lines[cursor + i]) for i in range(block_node_count)]
            cursor += block_node_count
            for tag in tags:
                coords = [float(value) for value in lines[cursor].split()]
                cursor += 1
                vertices_by_tag[tag] = coords[:3]
    else:
        raise ValueError(f"Unsupported Gmsh $Nodes header: {lines[node_marker + 1]!r}")

    vertices = [vertices_by_tag[tag][:dimension] for tag in sorted(vertices_by_tag)]
    return torch.tensor(vertices, dtype=torch.float64, requires_grad=True)
