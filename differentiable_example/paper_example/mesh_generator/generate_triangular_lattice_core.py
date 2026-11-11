from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import gmsh
from shapely import BufferCapStyle, BufferJoinStyle
from shapely.geometry import LineString, MultiPolygon, Polygon, box
from shapely.ops import unary_union
from shapely.geometry.polygon import orient


def _validate_params(
    h: float,
    q: float,
    w: float,
    j: float,
    theta_deg: float,
    nx: int,
    ny: int,
    mesh_size: float,
    plate_overhang: float = 0.0,
) -> None:
    if not all(math.isfinite(v) for v in (h, q, w, j, theta_deg, mesh_size, plate_overhang)):
        raise ValueError("Geometry parameters must be finite.")
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (nx, ny)):
        raise ValueError("nx and ny must be positive integers.")
    if h <= 0 or q <= 0 or w <= 0 or j <= 0 or mesh_size <= 0:
        raise ValueError("h, q, w, j, mesh_size must be positive.")
    if nx <= 0 or ny <= 0:
        raise ValueError("nx and ny must be positive integers.")
    if plate_overhang < 0:
        raise ValueError("plate_overhang must be non-negative.")
    if not (1.0 < theta_deg < 170.0):
        raise ValueError("theta_deg must be in (1, 170) degrees.")

    half_span = j * math.tan(math.radians(theta_deg) / 2.0)
    cell_width = 2.0 * half_span
    if h >= 0.90 * min(j, cell_width):
        raise ValueError(
            f"h={h} is too large for j={j}, theta_deg={theta_deg} (cell_width={cell_width:.6f}). "
            "Use smaller h or larger j/theta."
        )


def build_truss_layer(
    *,
    y0: float,
    y1: float,
    nx: int,
    half_span: float,
    beam_t: float,
) -> Polygon | MultiPolygon:
    # Continuous zigzag centerline: /\/\/\/... (2*nx+1 points)
    pts: list[tuple[float, float]] = []
    for i in range(2 * nx + 1):
        x = i * half_span
        y = y0 if i % 2 == 0 else y1
        pts.append((x, y))

    centerline = LineString(pts)
    return centerline.buffer(beam_t / 2.0, cap_style=BufferCapStyle.flat, join_style=BufferJoinStyle.mitre)


def core_horizontal_extent(*, j: float, theta_deg: float, nx: int) -> float:
    """Horizontal span of the triangular truss (same as former single ``width``)."""
    half_span = j * math.tan(math.radians(theta_deg) / 2.0)
    cell_width = 2.0 * half_span
    return nx * cell_width


def plate_horizontal_bounds(*, j: float, theta_deg: float, nx: int, plate_overhang: float) -> tuple[float, float]:
    """Returns ``(xmin, xmax)`` of the layered plates when truss stays on x in ``[0, core_width]``."""
    core = core_horizontal_extent(j=j, theta_deg=theta_deg, nx=nx)
    return (-plate_overhang, core + plate_overhang)


def build_microstructure(
    *,
    h: float,
    q: float,
    w: float,
    j: float,
    theta_deg: float,
    nx: int = 8,
    ny: int = 4,
    plate_overhang: float = 0.0,
) -> tuple[Polygon | MultiPolygon, float, float, float]:
    _validate_params(h, q, w, j, theta_deg, nx, ny, mesh_size=1e-3, plate_overhang=plate_overhang)

    half_span = j * math.tan(math.radians(theta_deg) / 2.0)
    cell_width = 2.0 * half_span
    core_width = nx * cell_width
    px0, px1 = plate_horizontal_bounds(
        j=j, theta_deg=theta_deg, nx=nx, plate_overhang=plate_overhang
    )
    # bottom plate: q, middle plates: q, top plate: w  (plates can extend past x=0..core_width)
    plates: list[Polygon] = []
    layers: list[Polygon | MultiPolygon] = []
    y_cursor = 0.0

    # bottom plate
    plates.append(box(px0, y_cursor, px1, y_cursor + q))
    y_cursor += q

    # ny truss layers with (ny-1) middle plates q
    for k in range(ny):
        y0 = y_cursor
        y1 = y0 + j
        layers.append(
            build_truss_layer(
                y0=y0,
                y1=y1,
                nx=nx,
                half_span=half_span,
                beam_t=h,
            )
        )
        y_cursor = y1
        if k < ny - 1:
            plates.append(box(px0, y_cursor, px1, y_cursor + q))
            y_cursor += q

    # top plate (thickness w)
    plates.append(box(px0, y_cursor, px1, y_cursor + w))
    y_cursor += w
    height = y_cursor

    geom = unary_union([*plates, *layers]).buffer(0)
    geom = geom.intersection(box(px0, 0.0, px1, height)).buffer(0)

    if geom.is_empty:
        raise RuntimeError("Generated geometry is empty.")
    if not geom.is_valid:
        raise RuntimeError("Generated geometry is invalid (self-intersection or collapse).")
    if isinstance(geom, MultiPolygon):
        # For PolyFEM this should be a connected solid.
        raise RuntimeError("Generated geometry is disconnected (MultiPolygon). Try milder h/q.")

    return geom, px0, px1, height


def _ring_to_loop(ring_coords: list[tuple[float, ...]], lc: float) -> int:
    # drop the repeated closing point and remove near-duplicates
    raw = ring_coords[:-1]
    pts: list[tuple[float, float]] = []
    eps = 1e-10
    for x, y in raw:
        xx = float(x)
        yy = float(y)
        if not pts:
            pts.append((xx, yy))
            continue
        px, py = pts[-1]
        if abs(xx - px) > eps or abs(yy - py) > eps:
            pts.append((xx, yy))
    if len(pts) >= 2:
        if abs(pts[0][0] - pts[-1][0]) <= eps and abs(pts[0][1] - pts[-1][1]) <= eps:
            pts.pop()
    if len(pts) < 3:
        raise RuntimeError("Degenerate ring for gmsh loop.")

    p_tags = [gmsh.model.occ.addPoint(x, y, 0.0, lc) for x, y in pts]
    l_tags = [gmsh.model.occ.addLine(p_tags[i], p_tags[(i + 1) % len(p_tags)]) for i in range(len(p_tags))]
    return gmsh.model.occ.addCurveLoop(l_tags)


def add_polygon_to_gmsh(geom: Polygon | MultiPolygon, mesh_size: float) -> list[int]:
    polygons: Iterable[Polygon]
    if isinstance(geom, Polygon):
        polygons = [geom]
    else:
        polygons = list(geom.geoms)

    surface_tags: list[int] = []
    for poly in polygons:
        # Consistent +z orientation: exterior CCW, holes CW.
        poly = orient(poly, sign=1.0)
        outer_loop = _ring_to_loop(list(poly.exterior.coords), mesh_size)
        # Never silently fill a hole if its conversion fails.
        hole_loops = [_ring_to_loop(list(ring.coords), mesh_size) for ring in poly.interiors]
        surface_tags.append(gmsh.model.occ.addPlaneSurface([outer_loop, *hole_loops]))

    gmsh.model.occ.synchronize()
    return surface_tags


def tag_boundaries(
    *,
    surface_tags: list[int],
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> None:
    # OCC curve bounding boxes usually include ~1e-7 padding, so use a robust tolerance.
    tol = max(1e-6, 1e-6 * max(xmax - xmin, ymax - ymin, 1.0))
    top_lines: list[int] = []
    bottom_lines: list[int] = []
    left_lines: list[int] = []
    right_lines: list[int] = []

    for dim, tag in gmsh.model.getEntities(1):
        x0, y0, _z0, x1, y1, _z1 = gmsh.model.getBoundingBox(dim, tag)
        if abs(y0 - ymax) < tol and abs(y1 - ymax) < tol:
            top_lines.append(tag)
        if abs(y0 - ymin) < tol and abs(y1 - ymin) < tol:
            bottom_lines.append(tag)
        if abs(x0 - xmin) < tol and abs(x1 - xmin) < tol:
            left_lines.append(tag)
        if abs(x0 - xmax) < tol and abs(x1 - xmax) < tol:
            right_lines.append(tag)

    pg_domain = gmsh.model.addPhysicalGroup(2, surface_tags, tag=1)
    gmsh.model.setPhysicalName(2, pg_domain, "domain")

    if top_lines:
        pg = gmsh.model.addPhysicalGroup(1, top_lines, tag=2)
        gmsh.model.setPhysicalName(1, pg, "top_boundary")
    if bottom_lines:
        pg = gmsh.model.addPhysicalGroup(1, bottom_lines, tag=3)
        gmsh.model.setPhysicalName(1, pg, "bottom_boundary")
    if left_lines:
        pg = gmsh.model.addPhysicalGroup(1, left_lines, tag=4)
        gmsh.model.setPhysicalName(1, pg, "left_boundary")
    if right_lines:
        pg = gmsh.model.addPhysicalGroup(1, right_lines, tag=5)
        gmsh.model.setPhysicalName(1, pg, "right_boundary")


def generate_mesh(
    *,
    h: float = 0.04,
    q: float = 0.12,
    w: float = 0.50,
    j: float = 1.0,
    theta_deg: float = 90.0,
    nx: int = 8,
    ny: int = 4,
    mesh_size: float = 0.05,
    plate_overhang: float = 0.35,
    output_path: Path | str,
) -> Path:
    """Generate a new P1 triangular mesh; lengths use the caller's unit (cm here).

    h is beam normal thickness; theta_deg is the triangle opening angle.
    This Gmsh operation is not differentiable and changes mesh connectivity.
    """
    _validate_params(h, q, w, j, theta_deg, nx, ny, mesh_size, plate_overhang)
    out_path = Path(output_path)
    if out_path.exists():
        raise FileExistsError(out_path)
    if gmsh.isInitialized():
        raise RuntimeError("Use this generator outside an existing Gmsh session.")
    geom, px0, px1, height = build_microstructure(
        h=h,
        q=q,
        w=w,
        j=j,
        theta_deg=theta_deg,
        nx=nx,
        ny=ny,
        plate_overhang=plate_overhang,
    )

    gmsh.initialize()
    try:
        gmsh.model.add("layered_microstructure_2d")
        gmsh.option.setNumber("General.NumThreads", 1)
        surface_tags = add_polygon_to_gmsh(geom, mesh_size)
        tag_boundaries(surface_tags=surface_tags, xmin=px0, xmax=px1, ymin=0.0, ymax=height)

        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.model.mesh.generate(2)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        gmsh.write(str(out_path))
        return out_path
    finally:
        gmsh.finalize()

