"""Geometry-only checks on supplied meshes, no simulation or remeshing."""
import importlib.util
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("lattice_geometry", HERE / "_lattice_geometry.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture(scope="module")
def geometry():
    return module.LatticeGeometry(HERE)


def test_reference_and_design_bounds(geometry):
    assert geometry.initial_thickness == pytest.approx(.04, abs=1e-10)
    reference = geometry(torch.tensor(geometry.initial_thickness, dtype=torch.float64),
                         torch.tensor(90., dtype=torch.float64))
    torch.testing.assert_close(reference, geometry.base, rtol=0, atol=0)
    for h in (.0396, .0404):
        for angle in (89.9, 90.1):
            assert geometry.check(h, angle)["block_unchanged"]


@pytest.mark.parametrize("parameter", ["thickness", "angle"])
def test_vertex_mapping_derivative(geometry, parameter):
    h = torch.tensor(.04, dtype=torch.float64, requires_grad=True)
    angle = torch.tensor(90., dtype=torch.float64, requires_grad=True)
    generator = torch.Generator().manual_seed(32)
    weights = torch.randn(geometry.base.shape, generator=generator, dtype=torch.float64)
    loss = (geometry(h, angle) * weights).sum()
    loss.backward()
    target = h if parameter == "thickness" else angle
    derivative = float(target.grad)
    eps = 1e-6 if parameter == "thickness" else 1e-4
    with torch.no_grad():
        target.add_(eps)
        plus = float((geometry(h, angle)*weights).sum())
        target.sub_(2*eps)
        minus = float((geometry(h, angle)*weights).sum())
    assert derivative == pytest.approx((plus-minus)/(2*eps), rel=1e-5, abs=1e-6)
