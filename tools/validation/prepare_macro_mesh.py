"""Build the laptop-size CMM mesh from the original outer boundary and four holes."""

from pathlib import Path

import gmsh
import meshio
import numpy as np
from shapely.geometry import MultiLineString
from shapely.ops import polygonize


def prepare():
    root = Path(__file__).resolve().parents[2]
    source = root / "differentiability-data/optimizations/macro_strain-opt/CMM.msh"
    mesh = meshio.read(source)
    triangles = mesh.cells_dict["triangle"]
    edges = np.sort(triangles[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    segments = MultiLineString(mesh.points[unique[counts == 1], :2].tolist())
    domain = max(polygonize(segments), key=lambda p: p.area)
    # The exterior starts along a straight side. Use its exact rectangular envelope.
    assert domain.exterior.is_ring and len(domain.interiors) == 4
    assert (
        abs(
            domain.envelope.area
            - domain.area
            - sum(p.area for p in polygonize(segments) if not p.interiors)
        )
        < 1e-7
    )
    gmsh.initialize()
    try:
        gmsh.model.add("CMM-coarse")
        geo = gmsh.model.geo
        loops = []
        for ring in [domain.envelope.exterior, *domain.simplify(0.005).interiors]:
            vertices = [geo.addPoint(x, y, 0.0, 0.2) for x, y in list(ring.coords)[:-1]]
            edges = [
                geo.addLine(v, vertices[(i + 1) % len(vertices)])
                for i, v in enumerate(vertices)
            ]
            loops.append(geo.addCurveLoop(edges, reorient=True))
        surface = geo.addPlaneSurface(loops)
        geo.synchronize()
        gmsh.model.addPhysicalGroup(2, [surface], 1)
        gmsh.model.mesh.generate(2)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(
            str(root / "differentiable_example/optimizations/geometry/cmm_coarse.msh")
        )
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    prepare()
