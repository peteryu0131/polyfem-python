"""Map boundary controls and interior coordinates to the full 3D mesh."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from polyfempy import polyfempy as backend


def geometry_file(spec, config):
    (geometry,) = config["geometry"]
    mesh = Path(geometry["mesh"])
    description = {
        "geometry": {**geometry, "mesh": mesh.name},
        "maps": spec["variable_to_simulation"],
    }
    digest = hashlib.sha256(
        json.dumps(description, sort_keys=True).encode() + mesh.read_bytes()
    )
    return Path(__file__).with_name("geometry") / (digest.hexdigest() + ".npz")


# Precomputed weights for the three meshes, produced by prepare_optimization_geometry.
BOX_WEIGHTS = "895dcb3f9360394e502fbdc931287dc633135fa017f9f8d75b2425c2b7a9e3bf.npz"
BAR_WEIGHTS = "3b92807a21f6fea0e26a50a8d42d25032d3691f286264fb06d4da8a23101d0bb.npz"
LAYER_WEIGHTS = "0536957434beaded192e680901401f552773dadcc979a4b895ae15904c5394ad.npz"


def rotate(points, angles):
    # PolyFEM applies rotations about x, then y, then z, around each handle.
    a, b, c = angles.unbind(1)
    x, y, z = points.unbind(2)
    y, z = y * a.cos() - z * a.sin(), y * a.sin() + z * a.cos()
    x, z = x * b.cos() + z * b.sin(), -x * b.sin() + z * b.cos()
    x, y = x * c.cos() - y * c.sin(), x * c.sin() + y * c.cos()
    return torch.stack((x, y, z), dim=2)


def load_handle_data(config, weights_file):
    """Read reference vertices, boundary and interior indices, weights, and pivots."""
    session = backend.DifferentiableSession()
    session.set_settings(config)
    vertices = torch.tensor(session.mesh_vertices(), dtype=torch.float64)
    cache = Path(__file__).with_name("geometry") / weights_file
    with np.load(cache, allow_pickle=False) as data:
        skin = torch.from_numpy(data["skin"])
        inside = torch.from_numpy(data["inside"])
        weights = torch.from_numpy(data["weights"])
        pivots = vertices[torch.from_numpy(data["handles"])]
        if not torch.equal(torch.from_numpy(data["vertices"]), vertices):
            raise ValueError("Mesh vertex order differs from the stored BBW setup")
    return vertices, skin, inside, weights, pivots
