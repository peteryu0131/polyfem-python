"""Generate both meshes in a NEW directory; never replace the experiment meshes.

From the repository root:
python -m differentiable_example.paper_example.mesh_generator.mesh_generation --output-dir path/to/new_case

Lengths are cm. theta_deg is the opening angle, not the leg inclination.
Changing lattice parameters does not resize or move the block.
"""
import argparse
import json
import math
from pathlib import Path

# A file launched directly needs its package name for the relative imports below.
if not __package__:
    __package__ = "differentiable_example.paper_example.mesh_generator"

from .generate_falling_weight_square import generate_mesh as generate_block_mesh
from .generate_triangular_lattice_core import build_microstructure, generate_mesh as generate_lattice_mesh


def prepare_impact_meshes(*, output_dir, h=0.04, theta_deg=90.0, q=0.12,
                         w=0.50, j=1.0, nx=8, n_layers=4, plate_overhang=0.35,
                         mesh_size=0.05, block_width=14.195, block_height=2.988,
                         block_x_center=8.0, block_y_bottom=5.1294):
    """Return lattice path, block path and parameter metadata.

Defaults match the reference geometry dimensions, not necessarily its node IDs
or triangulation. New Gmsh meshes are for forward/data generation, not backward().
The lattice starts at x=0 (overhang extends left); the block stays fixed in space.
"""
    _, xmin, xmax, height = build_microstructure(
        h=h, q=q, w=w, j=j, theta_deg=theta_deg, nx=nx, ny=n_layers,
        plate_overhang=plate_overhang)
    if not math.isfinite(mesh_size) or mesh_size <= 0:
        raise ValueError("mesh_size must be positive and finite.")
    if not all(math.isfinite(v) for v in (block_width, block_height, block_x_center, block_y_bottom)):
        raise ValueError("Block parameters must be finite.")
    if min(block_width, block_height) <= 0 or block_y_bottom <= height:
        raise ValueError("Block dimensions must be positive and its bottom above the lattice.")
    if block_x_center + block_width / 2 <= xmin or block_x_center - block_width / 2 >= xmax:
        raise ValueError("The fixed block has no horizontal overlap with the lattice.")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    lattice_path = generate_lattice_mesh(
        h=h, q=q, w=w, j=j, theta_deg=theta_deg, nx=nx, ny=n_layers,
        plate_overhang=plate_overhang, mesh_size=mesh_size,
        output_path=root / "triangular_lattice.msh")
    block_path = generate_block_mesh(
        width=block_width, height=block_height, x_center=block_x_center,
        y_bottom=block_y_bottom, mesh_size=mesh_size,
        output_path=root / "falling_weight_block.msh")
    parameters = {
        "length_unit": "cm", "mesh_size": mesh_size,
        "lattice": {"h": h, "theta_deg": theta_deg, "q": q, "w": w, "j": j,
                    "nx": nx, "n_layers": n_layers, "plate_overhang": plate_overhang},
        "block": {"width": block_width, "height": block_height,
                  "x_center": block_x_center, "y_bottom": block_y_bottom},
        "lattice_bbox": [xmin, 0.0, xmax, height],
        "gap": block_y_bottom - height,
        "block_fixed_when_lattice_changes": True,
    }
    (root / "parameters.json").write_text(json.dumps(parameters, indent=2, allow_nan=False), encoding="utf-8")
    return lattice_path, block_path, parameters


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--thickness", type=float, default=0.04, help="Leg thickness in cm")
    parser.add_argument("--angle", type=float, default=90.0, help="Opening angle in degrees")
    parser.add_argument("--mesh-size", type=float, default=0.05)
    args = parser.parse_args()
    paths = prepare_impact_meshes(output_dir=args.output_dir, h=args.thickness,
                                 theta_deg=args.angle, mesh_size=args.mesh_size)
    print("Saved:", paths[0], paths[1])
