"""Small two-body probe for the paper experiment's final-state output path."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest


def test_final_stress_export_without_visualization():
    pytest.importorskip("polyfempy.polyfempy")
    meshio = pytest.importorskip("meshio")
    from polyfempy import solve
    import polyfem

    # Eight triangles, two time steps: validate IO without running the full lattice.
    with TemporaryDirectory(prefix="polyfem-output-probe-") as temporary:
        root = Path(temporary)
        model = polyfem.model()
        points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 0]])
        triangles = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
        for index, young in enumerate((20, 40)):
            path = root / f"body_{index}.msh"
            meshio.write(path, meshio.Mesh(
                points + [2 * index, 0, 0], [("triangle", triangles)],
                cell_data={"gmsh:physical": [np.ones(4, dtype=int)],
                           "gmsh:geometrical": [np.ones(4, dtype=int)]},
            ), file_format="gmsh22", binary=False)
            body = model.body().mesh(mesh=str(path))
            body.material(type="NeoHookean", E={"value": young, "unit": "MPa"},
                          nu=0.3, rho={"value": 1100, "unit": "kg/m^3"})
            body.surface_all(id=index + 3).dirichlet(value=["0.1*x*t", 0])
        arrays = []
        for visualize in (False, True):
            output_dir = root / ("visualization" if visualize else "data_only")
            output_dir.mkdir()
            config = model.config(
                units=polyfem.units(length="cm", mass="g", time="s"),
                space=polyfem.space(discr_order=1),
                time=polyfem.time(polyfem.object1(dt=0.01, tend=0.02, integrator="BDF")),
                contact=polyfem.contact(enabled=False),
                solver=polyfem.solver(max_threads=1,
                    linear=polyfem.linear(solver="Eigen::PardisoLDLT")),
                output=polyfem.output(directory=str(output_dir), json="stats.json",
                    # Native global loggers keep files open on Windows; this tiny
                    # temporary probe uses captured terminal diagnostics only.
                    log=polyfem.log(level="off", file_level="off", path=""),
                    paraview=polyfem.output_paraview(file_name="impact.pvd" if visualize else "",
                        surface=False, wireframe=False, points=False,
                        options=polyfem.options(body_ids=True), vismesh_rel_area=10000000),
                    advanced=polyfem.output_advanced(save_time_sequence=visualize,
                        save_solve_sequence_debug=False),
                    data=polyfem.output_data(solution="solution.txt", stress_mat="stress.txt",
                        mises="von_mises.txt")),
            )
            result = solve(cfg=config)
            solution = np.loadtxt(output_dir / "solution.txt")
            stress = np.loadtxt(output_dir / "stress.txt")
            mises = np.loadtxt(output_dir / "von_mises.txt")
            assert np.isfinite(solution).all() and np.isfinite(stress).all()
            assert np.isfinite(mises).all() and mises.min() > 0
            np.testing.assert_allclose(solution, result.sol.ravel())
            assert stress.shape == (8, 3)  # P1 triangles: one export quadrature point each.
            assert mises.shape == (8,)
            # Native 2D Cauchy components are [xx, yy, xy], in g/(cm*s^2).
            xx, yy, xy = stress.T
            np.testing.assert_allclose(mises, np.sqrt(xx**2 - xx*yy + yy**2 + 3*xy**2))
            # The same strain in the second, twice-as-stiff body gives twice the stress.
            np.testing.assert_allclose(mises[4:].mean() / mises[:4].mean(), 2, rtol=1e-5)
            assert 0.01 < mises[0] / 1e7 < 1  # Convert this small-strain probe to MPa.
            arrays.append((solution, stress, mises))
            if not visualize:
                assert not any(p.suffix in {".vtu", ".vtm", ".pvd"}
                               for p in output_dir.rglob("*"))
            else:
                assert len(list(output_dir.glob("step_*.vtu"))) == 3
                mesh = meshio.read(output_dir / "step_2.vtu")
                ids = mesh.point_data["body_ids"].ravel()
                # Confirm body-group order against independently labeled VTU data.
                for body_id in (1, 2):
                    sampled = mesh.point_data["von_mises"][ids == body_id].ravel()
                    exported = mises[(body_id - 1) * 4:body_id * 4]
                    np.testing.assert_allclose(np.unique(sampled), np.unique(exported), rtol=1e-6)
        for without_vtu, with_vtu in zip(*arrays):
            np.testing.assert_allclose(without_vtu, with_vtu, rtol=1e-10, atol=1e-10)
