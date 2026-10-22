"""Generate the rectangular block, independently of the lattice design."""
import math
from pathlib import Path

import gmsh


def generate_mesh(*, width=14.195, height=2.988, x_center=8.0,
                  y_bottom=5.1294, mesh_size=0.05, output_path):
    """Lengths are in cm. Domain id 2, all boundary edges id 4 (never fixed id 3)."""
    if not all(math.isfinite(v) for v in (width, height, x_center, y_bottom, mesh_size)):
        raise ValueError("Block parameters must be finite.")
    if min(width, height, mesh_size) <= 0:
        raise ValueError("width, height and mesh_size must be positive.")
    path = Path(output_path)
    if path.exists():
        raise FileExistsError(path)
    if gmsh.isInitialized():
        raise RuntimeError("Use this generator outside an existing Gmsh session.")
    gmsh.initialize()
    try:
        gmsh.model.add("falling_weight_block")
        gmsh.option.setNumber("General.NumThreads", 1)
        surface = gmsh.model.occ.addRectangle(x_center - width / 2, y_bottom, 0, width, height)
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(2, [surface], tag=2)
        gmsh.model.setPhysicalName(2, 2, "block")
        edges = [tag for dim, tag in gmsh.model.getBoundary([(2, surface)], oriented=False)]
        gmsh.model.addPhysicalGroup(1, edges, tag=4)
        gmsh.model.setPhysicalName(1, 4, "block_boundary")
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.model.mesh.generate(2)
        path.parent.mkdir(parents=True, exist_ok=True)
        gmsh.write(str(path))
        return path
    finally:
        gmsh.finalize()
