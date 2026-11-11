"""Small mesh-generation checks only; no FEM simulation."""
import sys
from pathlib import Path

import meshio
import numpy as np
import pytest

PAPER_EXAMPLE = Path(__file__).resolve().parents[2] / "differentiable_example" / "paper_example"

sys.path.insert(0, str(PAPER_EXAMPLE))
from mesh_generator.generate_triangular_lattice_core import build_microstructure, build_truss_layer, generate_mesh
from mesh_generator.mesh_generation import prepare_impact_meshes


def triangle_area(mesh):
    points = mesh.points[mesh.cells_dict["triangle"], :2]
    u, v = points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]
    areas = (u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]) / 2
    assert np.all(areas > 0), "Flipped or collapsed triangle"
    return areas.sum()


def test_reference_dimensions_and_thickness():
    geom, xmin, xmax, height = build_microstructure(
        h=.04, q=.12, w=.5, j=1., theta_deg=90., nx=8, ny=4, plate_overhang=.35)
    assert (xmin, xmax, height) == pytest.approx((-.35, 16.35, 4.98))
    assert geom.is_valid and geom.geom_type == "Polygon" and len(geom.interiors) > 0
    # Away from joints, intersect a leg with its normal to measure physical thickness.
    from shapely.geometry import LineString
    layer = build_truss_layer(y0=0, y1=1, nx=1, half_span=1, beam_t=.04)
    cut = LineString([(.4, .6), (.6, .4)])
    assert layer.intersection(cut).length == pytest.approx(.04)


@pytest.mark.parametrize("angle", [70., 90., 110.])
def test_generated_mesh_area_tags_and_fixed_block(tmp_path, angle):
    lattice, block, params = prepare_impact_meshes(
        output_dir=tmp_path / "case", nx=2, n_layers=1, theta_deg=angle,
        block_width=1., block_height=.3, block_x_center=2., block_y_bottom=2., mesh_size=.15)
    geom, *_ = build_microstructure(h=.04, q=.12, w=.5, j=1, theta_deg=angle,
                                  nx=2, ny=1, plate_overhang=.35)
    lm, bm = meshio.read(lattice), meshio.read(block)
    assert triangle_area(lm) == pytest.approx(geom.area, rel=1e-8)
    assert triangle_area(bm) == pytest.approx(.3)
    np.testing.assert_allclose(bm.points[:, :2].min(0), [1.5, 2.])
    np.testing.assert_allclose(bm.points[:, :2].max(0), [2.5, 2.3])
    tags = lm.cell_data_dict["gmsh:physical"]["line"]
    bottom_edges = lm.cells_dict["line"][tags == 3]
    assert len(bottom_edges) > 0
    np.testing.assert_allclose(lm.points[bottom_edges, 1], 0., atol=1e-10)
    assert set(bm.cell_data_dict["gmsh:physical"]["line"]) == {4}
    assert set(bm.cell_data_dict["gmsh:physical"]["triangle"]) == {2}
    assert params["block_fixed_when_lattice_changes"]
    with pytest.raises(FileExistsError):
        generate_mesh(output_path=lattice)


@pytest.mark.parametrize("kwargs", [{"mesh_size": 0}, {"h": float("nan")}, {"nx": 1.5}, {"ny": True}])
def test_invalid_inputs_before_meshing(tmp_path, kwargs):
    with pytest.raises(ValueError):
        generate_mesh(output_path=tmp_path / "invalid.msh", **kwargs)
    assert not (tmp_path / "invalid.msh").exists()
