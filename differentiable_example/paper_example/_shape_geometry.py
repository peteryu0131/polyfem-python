"""Fixed-topology deformation of the layered zigzag mesh.

Boundary corners are intersections of moving plate/leg lines. Mesh boundary
nodes interpolate those corners; a fixed discrete harmonic extension moves
interior nodes. Shapely and SciPy are used only during reference setup.
"""

import meshio
import numpy as np
import torch
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import splu
from scipy.spatial import KDTree
from shapely.geometry import Polygon

if __package__:
    from .mesh_generator.generate_triangular_lattice_core import build_microstructure
else:
    from mesh_generator.generate_triangular_lattice_core import build_microstructure


class LatticeShape:
    def __init__(self, path, *, h=.04, theta_deg=90., j=1., q=.12, w=.5,
                 nx=8, ny=4, plate_overhang=.35):
        self.reference_h, self.reference_angle = h, theta_deg
        self.j, self.q, self.w = j, q, w
        self.nx, self.ny, self.overhang = nx, ny, plate_overhang
        mesh = meshio.read(path)
        self.base = torch.tensor(mesh.points[:, :2], dtype=torch.float64)
        self.faces = mesh.get_cells_type("triangle")
        if len(self.faces) == 0 or len(mesh.get_cells_type("triangle6")):
            raise ValueError("Expected a P1 triangle mesh")
        polygon, _, _, _ = build_microstructure(
            h=h, theta_deg=theta_deg, j=j, q=q, w=w, nx=nx, ny=ny,
            plate_overhang=plate_overhang)
        if not isinstance(polygon, Polygon):
            raise ValueError("Expected a single connected lattice polygon")
        rings = [np.asarray(r.coords)[:-1] for r in [polygon.exterior, *polygon.interiors]]
        corners = np.concatenate(rings)
        next_indices, start = [], 0
        for ring in rings:
            next_indices.extend(start + np.roll(np.arange(len(ring)), -1))
            start += len(ring)
        self.next_indices = np.asarray(next_indices)
        self.reference_corners = torch.tensor(corners, dtype=torch.float64)
        normals, offsets = self._lines(self.base.new_tensor(h), self.base.new_tensor(theta_deg))
        distances = np.abs(corners @ normals.numpy().T - offsets.numpy())
        errors = np.maximum(distances, distances[self.next_indices])
        self.edge_lines = errors.argmin(axis=1)
        if errors.min(axis=1).max() > 1e-8:
            raise ValueError("Could not identify the reference polygon edges")
        previous = np.argsort(self.next_indices)
        self.corner_lines = np.stack((self.edge_lines[previous], self.edge_lines), axis=1)
        self.weights = self._harmonic_weights(corners)
        self.reference_area = self._areas(self.base).detach()
        if bool((self.reference_area.abs() < 1e-14).any()):
            raise ValueError("Reference mesh contains degenerate elements")

    def _lines(self, h, angle):
        half = torch.deg2rad(angle) / 2
        a = self.j * torch.tan(half)
        one, zero = h.new_tensor(1.), h.new_tensor(0.)
        normals, offsets = [], []

        def line(nx, ny, offset):
            normals.append(torch.stack((nx, ny)))
            offsets.append(offset if isinstance(offset, torch.Tensor) else h.new_tensor(offset))

        line(one, zero, -self.overhang)
        line(one, zero, 2*self.nx*a + self.overhang)
        line(zero, one, 0.)
        for layer in range(self.ny):
            y0 = self.q + layer * (self.j + self.q)
            y1 = y0 + self.j
            line(zero, one, y0)
            line(zero, one, y1)
            for leg in range(2*self.nx):
                upward = leg % 2 == 0
                py = y0 if upward else y1
                dy = self.j if upward else -self.j
                length = torch.sqrt(a*a + self.j*self.j)
                tx, ty = a / length, h.new_tensor(dy) / length
                nx, ny = -ty, tx
                b = nx * leg*a + ny*py
                line(nx, ny, b + h/2)
                line(nx, ny, b - h/2)
                if leg == 0:
                    line(tx, ty, tx * leg*a + ty*py)
                if leg == 2*self.nx - 1:
                    line(tx, ty, tx * (leg+1)*a + ty*(py+dy))
        line(zero, one, self.ny*(self.j+self.q) + self.w)
        return torch.stack(normals), torch.stack(offsets)

    def corners(self, h, angle):
        normals, offsets = self._lines(h, angle)
        corners = torch.linalg.solve(normals[self.corner_lines], offsets[self.corner_lines])
        # Change span about the original center; the block stays in place.
        reference_half_width = self.nx*self.j*np.tan(np.deg2rad(self.reference_angle)/2)
        shift = reference_half_width - self.nx*self.j*torch.tan(torch.deg2rad(angle)/2)
        return corners + torch.stack((shift, shift*0))

    def _harmonic_weights(self, corners):
        edges = np.sort(np.concatenate([self.faces[:, [0, 1]], self.faces[:, [1, 2]],
                                        self.faces[:, [2, 0]]]), axis=1)
        edges, counts = np.unique(edges, axis=0, return_counts=True)
        boundary = np.unique(edges[counts == 1])
        interior = np.setdiff1d(np.arange(len(self.base)), boundary)
        start, end = corners, corners[self.next_indices]
        direction = end - start
        points = self.base.numpy()[boundary]
        t = np.clip(((points[:, None] - start) * direction).sum(axis=2)
                    / (direction*direction).sum(axis=1), 0., 1.)
        distances = np.linalg.norm(points[:, None] - (start + t[:, :, None]*direction), axis=2)
        selected = distances.argmin(axis=1)
        if distances.min(axis=1).max() > 1e-7:
            raise ValueError("Mesh does not match the supplied reference geometry parameters")
        weights = np.zeros((len(self.base), len(corners)))
        fraction = t[np.arange(len(boundary)), selected]
        weights[boundary, selected] = 1-fraction
        weights[boundary, self.next_indices[selected]] += fraction
        rows, cols = np.concatenate((edges[:, 0], edges[:, 1])), np.concatenate((edges[:, 1], edges[:, 0]))
        adjacency = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(self.base),)*2).tocsr()
        laplacian = (diags(np.asarray(adjacency.sum(axis=1)).ravel()) - adjacency).tocsr()
        if len(interior):
            rhs = -laplacian[interior][:, boundary] @ weights[boundary]
            weights[interior] = splu(laplacian[interior][:, interior].tocsc()).solve(rhs)
        return torch.tensor(weights, dtype=torch.float64)

    def _areas(self, vertices):
        a, b, c = (vertices[self.faces[:, i]] for i in range(3))
        return (b[:, 0]-a[:, 0])*(c[:, 1]-a[:, 1]) - (b[:, 1]-a[:, 1])*(c[:, 0]-a[:, 0])

    def __call__(self, h, angle):
        if not (.8*self.reference_h <= float(h.detach()) <= 1.2*self.reference_h
                and self.reference_angle-5 <= float(angle.detach()) <= self.reference_angle+5):
            raise ValueError("This fixed-topology example allows only local thickness/angle changes")
        vertices = self.base + self.weights @ (self.corners(h, angle) - self.reference_corners)
        with torch.no_grad():
            ratio = self._areas(vertices) / self.reference_area
            if not bool(torch.isfinite(vertices).all()) or bool((ratio <= .1).any()):
                raise ValueError("Shape update would invert or collapse a triangle; reduce the step")
        return vertices

    def write(self, path, vertices):
        points = vertices.detach().cpu().numpy()
        meshio.write_points_cells(path, np.column_stack((points, np.zeros(len(points)))),
                                  [("triangle", self.faces)])


def backend_vertex_order(reference_parts, backend_vertices):
    """Verify a one-to-one coordinate match instead of assuming mesh node order."""
    reference = torch.cat(reference_parts).detach().numpy()
    distance, indices = KDTree(reference).query(np.asarray(backend_vertices))
    indices = np.asarray(indices)
    if indices.shape != (len(reference),) or len(np.unique(indices)) != len(reference) or np.max(distance) > 1e-8:
        raise ValueError("Backend vertex order cannot be matched to the reference meshes")
    return torch.tensor(indices, dtype=torch.long)


def prepare_geometry(config, lattice_path, block_path):
    """Prepare a callable returning all vertices, with the block held fixed."""
    from polyfempy import polyfempy as backend

    lattice = LatticeShape(lattice_path)
    block = torch.tensor(meshio.read(block_path).points[:, :2], dtype=torch.float64)
    session = backend.DifferentiableSession()
    session.set_settings(config.as_dict())
    order = backend_vertex_order([lattice.base, block], session.mesh_vertices())

    def vertices(thickness_cm, angle_deg):
        return torch.cat((lattice(thickness_cm, angle_deg), block))[order]

    return vertices
