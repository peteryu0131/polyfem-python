"""Geometry checks and a small static FEM thickness/angle gradient check."""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from shapely.affinity import translate
from shapely.geometry import Point, Polygon

PAPER_EXAMPLE = Path(__file__).resolve().parents[2] / "differentiable_example" / "paper_example"

sys.path.insert(0, str(PAPER_EXAMPLE))
from _shape_geometry import LatticeShape, backend_vertex_order
from mesh_generator.generate_triangular_lattice_core import build_microstructure, generate_mesh


@pytest.fixture(scope="module")
def full_geometry():
    return LatticeShape(PAPER_EXAMPLE / "triangular_lattice.msh")


def test_reference_identity_and_fixed_backend_order(full_geometry):
    g = full_geometry
    v = g(torch.tensor(.04, dtype=torch.float64), torch.tensor(90., dtype=torch.float64))
    torch.testing.assert_close(v, g.base, atol=1e-12, rtol=0)
    block = torch.tensor([[30., 0.], [30., 1.]], dtype=torch.float64)
    reference = torch.cat((v, block))
    permutation = torch.arange(len(reference)-1, -1, -1)
    order = backend_vertex_order([v, block], reference[permutation].numpy())
    torch.testing.assert_close(order, permutation)
    with pytest.raises(ValueError, match="matched"):
        backend_vertex_order([v, block], reference.numpy()+.01)


@pytest.mark.parametrize("h,angle", [(.039, 89.), (.041, 91.)])
def test_boundary_matches_generator_and_triangles_stay_valid(full_geometry, h, angle):
    g = full_geometry
    polygon, _, _, _ = build_microstructure(h=h, theta_deg=angle, j=1., q=.12, w=.5,
                                           nx=8, ny=4, plate_overhang=.35)
    shift = 8 - 8*np.tan(np.deg2rad(angle)/2)
    polygon = translate(polygon, xoff=shift)
    h, angle = torch.tensor(h, dtype=torch.float64), torch.tensor(angle, dtype=torch.float64)
    corners = g.corners(h, angle).detach().numpy()
    assert max(polygon.boundary.distance(Point(p)) for p in corners) < 1e-9
    # Check full rings, not just their vertices.
    lengths = [len(r.coords)-1 for r in [polygon.exterior, *polygon.interiors]]
    start, rings = 0, []
    for length in lengths:
        rings.append(corners[start:start+length])
        start += length
    reconstructed = Polygon(rings[0], rings[1:])
    assert reconstructed.symmetric_difference(polygon).area < 1e-9
    vertices = g(h, angle)
    assert bool((g._areas(vertices)/g.reference_area > .1).all())


def test_geometry_chain_rule_and_two_updates(full_geometry):
    g = full_geometry
    scale = torch.tensor([.04, 90.], dtype=torch.float64)
    x = torch.ones(2, dtype=torch.float64, requires_grad=True)

    def loss(p):
        return g(*(p*scale)).square().mean()

    loss(x).backward()
    analytical = x.grad.clone()
    with torch.no_grad():
        for i in range(2):
            d = torch.zeros_like(x)
            d[i] = 1e-5
            fd = (loss(x+d)-loss(x-d))/(2e-5)
            torch.testing.assert_close(analytical[i], fd, rtol=1e-5, atol=1e-7)
    optimizer = torch.optim.Adam([x], lr=.0001)
    for _ in range(2):
        optimizer.zero_grad()
        loss(x).backward()
        optimizer.step()
    assert bool((x.detach() != 1).all())


def test_example_one_update_and_fixed_block(tmp_path, monkeypatch, full_geometry):
    import ast
    import meshio
    import polyfem.diff as diff
    from polyfempy import polyfempy as native
    directory = PAPER_EXAMPLE
    block = torch.tensor(meshio.read(directory / "falling_weight_block.msh").points[:, :2],
                         dtype=torch.float64)
    base = torch.cat((full_geometry.base, block))
    permutation = torch.arange(len(base)-1, -1, -1)
    seen = []

    class Session:
        def set_settings(self, config):
            assert config["solver"]["contact"]["barrier_stiffness"] == 1e8

        def mesh_vertices(self):
            return base[permutation].numpy()

    def factory(config, *, selection, objective):
        assert objective == diff.Objective.VON_MISES_PEAK

        def evaluate(*, V):
            unpacked = V[permutation]
            torch.testing.assert_close(unpacked[len(full_geometry.base):], block, atol=0, rtol=0)
            seen.append(V.detach())
            return unpacked[:len(full_geometry.base)].square().mean()*1e7

        return evaluate

    monkeypatch.setattr(native, "DifferentiableSession", Session)
    monkeypatch.setattr(diff, "shape", factory)
    path = directory / "shape_opt.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    output_assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                             and any(isinstance(t, ast.Name) and t.id == "OUTPUT_DIR" for t in node.targets))
    output_assignment.value = ast.Call(func=ast.Name(id="Path", ctx=ast.Load()),
                                       args=[ast.Constant(str(tmp_path / "run"))], keywords=[])
    result = {"__file__": str(path)}
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), result)
    assert len(seen) == 1
    assert result["thickness_cm"].item() != .04
    assert result["angle_deg"].item() != 90.
    assert result["value"].item() == pytest.approx(full_geometry.base.square().mean().item())
    assert torch.isfinite(result["x"].grad).all()
    assert not (tmp_path / "run" / "result.json").exists()


def test_small_fem_thickness_and_angle(tmp_path):
    import polyfem.diff as diff
    from polyfempy import polyfempy as native
    import polyfem
    path = generate_mesh(output_path=tmp_path / "small.msh", h=.12, nx=1, ny=1,
                         q=.2, w=.2, mesh_size=.08, plate_overhang=.2)
    g = LatticeShape(path, h=.12, nx=1, ny=1, q=.2, w=.2, plate_overhang=.2)
    model = polyfem.model()
    body = model.body().mesh(mesh=str(path))
    body.material(type="NeoHookean", E=100., nu=.25)
    body.surface_axis(axis=-2, position=.0001, relative=True, id=1).dirichlet(value=[0., 0.])
    body.surface_axis(axis=2, position=.9999, relative=True, id=2).dirichlet(value=[.001, -.002])
    body.rhs(value=[0., 0.])
    config = model.config(
        space=polyfem.space(discr_order=1),
        solver=polyfem.solver(max_threads=1, linear=polyfem.linear(solver="Eigen::SimplicialLDLT"),
                         nonlinear=polyfem.nonlinear(grad_norm_tol=1e-10)),
        output=polyfem.output(directory=str(tmp_path), json="", paraview=polyfem.output_paraview(file_name=""),
                         advanced=polyfem.output_advanced(save_time_sequence=False)),
    )
    session = native.DifferentiableSession()
    session.set_settings(config.as_dict())
    order = backend_vertex_order([g.base], session.mesh_vertices())
    del session
    obj = diff.shape(config, selection=body, objective=diff.Objective.VON_MISES_PEAK)
    scale = torch.tensor([.12, 90.], dtype=torch.float64)
    x = torch.ones(2, dtype=torch.float64, requires_grad=True)

    def loss(p):
        return obj(V=g(*(p*scale))[order])

    value = loss(x)
    value.backward()
    analytical = x.grad.clone()
    with torch.no_grad():
        for i in range(2):
            d = torch.zeros_like(x)
            d[i] = 1e-5
            fd = (loss(x+d)-loss(x-d))/(2e-5)
            assert abs(float(fd)) > 1e-8
            torch.testing.assert_close(analytical[i], fd, rtol=.02, atol=1e-5)
    optimizer = torch.optim.Adam([x], lr=.0001)
    optimizer.step()
    with torch.no_grad():
        assert loss(x) < value.detach()
