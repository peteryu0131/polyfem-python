"""Vertex-only design map for the supplied four-layer zigzag mesh.

No meshing or FEM solves. Reference dimensions are checked against the mesh;
unsupported geometry fails instead of silently applying an unrelated mapping.
"""
from pathlib import Path

import meshio
import numpy as np
import torch


class LatticeGeometry:
    def __init__(self, directory):
        directory = Path(directory)
        lattice = meshio.read(directory / "triangular_lattice.msh")
        block = meshio.read(directory / "falling_weight_block.msh")
        points = lattice.points[:, :2]
        self.count = len(points)
        self.cells = lattice.get_cells_type("triangle")
        self.base = torch.tensor(np.concatenate((points, block.points[:, :2])), dtype=torch.float64)
        self.center_x = float((points[:, 0].min() + points[:, 0].max()) / 2)
        # The historic mesh has unit-height layers, .12 plates and a .5 top plate.
        # Verify these against actual mesh bounds and boundary segments below.
        self.layer_bottoms = np.array([.12, 1.24, 2.36, 3.48])
        if not np.allclose(points.min(0), [-.35, 0], atol=1e-7) or not np.allclose(points.max(0), [16.35, 4.98], atol=1e-7):
            raise ValueError("This map requires the supplied 16.7 x 4.98 cm lattice")
        starts, ends = [], []
        for bottom in self.layer_bottoms:
            for segment in range(16):
                starts.append([segment, bottom + segment % 2])
                ends.append([segment + 1, bottom + (segment + 1) % 2])
        starts, ends = np.array(starts), np.array(ends)
        tangents = ends - starts
        lengths = np.linalg.norm(tangents, axis=1)
        normals = np.stack((-tangents[:, 1], tangents[:, 0]), axis=1) / lengths[:, None]

        def locate(p):
            diff = p[:, None, :] - starts
            s = np.clip(np.sum(diff * tangents, axis=2) / lengths**2, 0, 1)
            offsets = diff - s[:, :, None] * tangents
            nearest = np.argmin(np.sum(offsets**2, axis=2), axis=1)
            rows = np.arange(len(p))
            return nearest, s[rows, nearest], np.sum(offsets[rows, nearest] * normals[nearest], axis=1)

        # Use boundary edges well away from joints to measure actual beam thickness.
        edges = np.concatenate((self.cells[:, [0, 1]], self.cells[:, [1, 2]], self.cells[:, [2, 0]]))
        unique, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
        edges = unique[counts == 1]
        midpoints = points[edges].mean(1)
        nearest, s, offset = locate(midpoints)
        interior = (s > .25) & (s < .75) & (np.abs(offset) < .1)
        measured = 2 * np.abs(offset[interior])
        if len(measured) < 32 or not np.allclose(measured, .04, atol=1e-6):
            raise ValueError("Boundary edges do not match the reference .04 cm truss thickness")
        edge_vectors = points[edges[:, 1]] - points[edges[:, 0]]
        slopes = np.abs(edge_vectors[interior, 1] / edge_vectors[interior, 0])
        if not np.allclose(slopes, 1, atol=1e-5):
            raise ValueError("Reference truss edges are not at 45 degrees")
        self.initial_thickness = float(np.median(measured))
        self.initial_angle = 90.
        nearest, s, offset = locate(points)
        plate_distance = np.full(len(points), np.inf)
        for lo, hi in [(0, .12), (1.12, 1.24), (2.24, 2.36), (3.36, 3.48), (4.48, 4.98)]:
            plate_distance = np.minimum(plate_distance, np.maximum(np.maximum(lo-points[:, 1], points[:, 1]-hi), 0))
        def smoothstep(x):
            x = np.clip(x, 0, 1)
            return x*x*(3-2*x)
        weight = smoothstep(np.minimum(s, 1-s)/.18) * smoothstep(plate_distance/.10)
        self.offset = torch.tensor(offset, dtype=torch.float64)
        self.weight = torch.tensor(weight, dtype=torch.float64)
        self.tangent = torch.tensor(tangents[nearest], dtype=torch.float64)
        self.reference_areas = self.areas(points)
        self.report = {"lattice_vertices": self.count, "lattice_triangles": len(self.cells),
                       "block_vertices": len(block.points), "thickness_cm": self.initial_thickness,
                       "opening_angle_degrees": self.initial_angle, "leg_angle_degrees": 45.,
                       "thickness_boundary_samples": len(measured), "fixed_connectivity": True,
                       "thickness_semantics": "interior normal thickness; tapered to zero change near joints"}

    def __call__(self, thickness, angle):
        reference_angle = angle.new_tensor(self.initial_angle)
        reference_tangent = torch.tan(torch.deg2rad(reference_angle) / 2)
        scale = 1 + (torch.tan(torch.deg2rad(angle) / 2)-reference_tangent)/reference_tangent
        base = self.base.to(thickness)
        lattice = base[:self.count]
        tangent = self.tangent.to(thickness)
        tx, ty = tangent[:, 0]*scale, tangent[:, 1]
        length = torch.sqrt(tx.square()+ty.square())
        normal = torch.stack((-ty/length, tx/length), dim=1)
        # Compensate the angle map's thickness change in beam interiors.
        affine_thickness = scale * torch.sqrt(tangent.square().sum(1)) / length
        delta = self.offset.to(thickness) * (thickness/self.initial_thickness-affine_thickness) * self.weight.to(thickness)
        x = lattice[:, 0] + (lattice[:, 0]-self.center_x)*(scale-1)
        mapped = torch.stack((x, lattice[:, 1]), dim=1) + normal * delta[:, None]
        return torch.cat((mapped, base[self.count:]), dim=0)

    def areas(self, points):
        triangle = points[self.cells]
        a, b = triangle[:, 1]-triangle[:, 0], triangle[:, 2]-triangle[:, 0]
        return a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0]

    def check(self, thickness, angle):
        """Cheap geometry-only guard before an expensive solve."""
        with torch.no_grad():
            vertices = self(torch.tensor(thickness, dtype=torch.float64), torch.tensor(angle, dtype=torch.float64)).numpy()
        ratio = self.areas(vertices[:self.count])/self.reference_areas
        if not np.isfinite(vertices).all() or ratio.min() <= .1:
            raise ValueError("Design flips or excessively collapses mesh elements")
        if not np.array_equal(vertices[self.count:], self.base.numpy()[self.count:]):
            raise ValueError("Block geometry changed")
        return {"minimum_signed_area_ratio": float(ratio.min()), "block_unchanged": True}
