#!/usr/bin/env python3
"""Forward-only lattice impact. Run from the repository root in the polyfem env."""

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from polyfempy import solve
from polyfempy.generated_api import generated_api as pf


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "runs" / "impact_forward"
DT = 0.005
END_TIME = 0.03


def build_config(output_dir=OUTPUT_DIR):
    model = pf.model()
    lattice = pf.body(model=model)
    lattice.mesh(pf.mesh(mesh=str(HERE / "triangular_lattice.msh")))
    lattice.material(pf.neo_hookean(
        E=pf.neo_hookean_e_object3(value=20, unit="MPa"), nu=0.45,
        rho=pf.neo_hookean_rho_object3(value=1100, unit="kg/m^3"),
    ))
    lattice.surface_axis(axis=-2, position=0.0001, relative=True, id=3).dirichlet(value=[0, 0])
    lattice.rhs(value=[0, 980])

    block = pf.body(model=model)
    block.mesh(pf.mesh(mesh=str(HERE / "falling_weight_block.msh")))
    block.material(pf.neo_hookean(
        E=pf.neo_hookean_e_object3(value=200, unit="GPa"), nu=0.45,
        rho=pf.neo_hookean_rho_object3(value=7850, unit="kg/m^3"),
    ))
    # Both original meshes contain physical boundary id 3. Override the block's
    # imported tags so fixing lattice bottom id 3 does not also pin the block.
    block.surface_all(id=4)
    block.rhs(value=[0, 980])
    block.velocity(pf.velocity_item(value=[0, 0]))

    return pf.config(
        model=model, units=pf.units(length="cm", mass="g", time="s"),
        space=pf.space(discr_order=1),
        time=pf.time(pf.object1(t0=0, dt=DT, tend=END_TIME, integrator="BDF")),
        contact=pf.contact(enabled=True, dhat=0.012),
        solver=pf.solver(
            max_threads=4,
            linear=pf.linear(solver="Eigen::PardisoLDLT"),
            nonlinear=pf.nonlinear(
                solver="Newton", iterations_per_strategy=1,
                grad_norm_tol=1e-6,
                line_search=pf.line_search(method="RobustArmijo"),
            ),
            contact=pf.solver_contact(barrier_stiffness="adaptive"),
        ),
        output=pf.output(
            directory=str(output_dir), json="impact_stats.json",
            log=pf.log(level="info", file_level="debug", path="polyfem.log"),
            paraview=pf.output_paraview(
                file_name="impact.pvd", surface=False, wireframe=False, points=False,
                options=pf.options(material=True, body_ids=True, velocity=True, tensor_values=True,
                                   contact_forces=True),
                vismesh_rel_area=10000000,
            ),
            advanced=pf.output_advanced(save_time_sequence=True, save_solve_sequence_debug=False),
        ),
    )


def verify_outputs(output_dir):
    """Check this experiment's exported samples, not a differentiable objective."""
    import meshio

    entries = ET.parse(output_dir / "impact.pvd").findall("./Collection/DataSet")
    expected_times = np.arange(round(END_TIME / DT) + 1) * DT
    times = np.array([float(entry.attrib["timestep"]) for entry in entries])
    if len(times) != len(expected_times) or not np.allclose(times, expected_times):
        raise RuntimeError("Incomplete exported time sequence")
    history = []
    for entry, time in zip(entries, times):
        blocks = ET.parse(output_dir / entry.attrib["file"])
        volume_file = blocks.find(".//Block[@name='Volume']/DataSet").attrib["file"]
        contact_file = blocks.find(".//Block[@name='Contact']/DataSet").attrib["file"]
        mesh = meshio.read(output_dir / volume_file)
        contact = meshio.read(output_dir / contact_file)
        fields = mesh.point_data
        for key in ("body_ids", "displacement", "von_mises", "cauchy_stess_1", "cauchy_stess_2"):
            if key not in fields or not np.isfinite(fields[key]).all():
                raise RuntimeError(f"Missing or non-finite output field: {key}")
        body_ids = fields["body_ids"].ravel()
        lattice, block = body_ids == 1, body_ids == 2
        if not lattice.any() or not block.any():
            raise RuntimeError("Both bodies must appear in the volume output")
        displacement = fields["displacement"]
        bottom = lattice & (np.abs(mesh.points[:, 1]) < 1e-8)
        if not bottom.any() or np.max(np.abs(displacement[bottom])) > 1e-8:
            raise RuntimeError("Lattice bottom is not fixed")
        force = contact.point_data["contact_forces"]
        if not np.isfinite(force).all():
            raise RuntimeError("Non-finite contact forces")
        # VTU points are reference coordinates. The original block is wholly
        # above the lattice, so this separates its collision vertices exactly.
        block_surface = contact.points[:, 1] >= mesh.points[block, 1].min() - 1e-8
        triangles = mesh.cells_dict["triangle"]
        rest = mesh.points[triangles, :2]
        moved = (mesh.points + displacement)[triangles, :2]
        def signed_area2(points):
            a, b = points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]
            return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
        ratio = signed_area2(moved) / signed_area2(rest)
        if not np.isfinite(ratio).all() or ratio.min() <= 0:
            raise RuntimeError("Inverted or degenerate deformed triangle")
        history.append({
            "time_s": float(time),
            "block_mean_displacement_y_cm": float(displacement[block, 1].mean()),
            "block_contact_force_y_g_cm_s2": float(force[block_surface, 1].sum()),
            "fixed_bottom_max_displacement_cm": float(np.abs(displacement[bottom]).max()),
            # g/(cm*s^2) = 0.1 Pa; report visualization-sampled maxima, not
            # a continuous/global peak or a differentiable stress objective.
            "lattice_sampled_max_von_mises_MPa": float(fields["von_mises"][lattice].max() / 1e7),
            "min_area_ratio": float(ratio.min()),
        })
    if history[-1]["block_mean_displacement_y_cm"] >= -0.01:
        raise RuntimeError("Block did not fall; check imported boundary labels")
    contact_steps = [row for row in history if row["block_contact_force_y_g_cm_s2"] > 1e-6]
    if not contact_steps or history[-1]["lattice_sampled_max_von_mises_MPa"] <= 0:
        raise RuntimeError("Impact contact and nonzero lattice stress were not observed")
    summary = {"status": "verified", "dt_s": DT, "end_time_s": END_TIME,
               "first_sample_with_block_contact_s": contact_steps[0]["time_s"], "history": history}
    (output_dir / "verification.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(output_dir=OUTPUT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Do not leave a previous success report after an interrupted/failed rerun.
    (output_dir / "verification.json").unlink(missing_ok=True)
    config = build_config(output_dir)
    (output_dir / "config.json").write_text(json.dumps(config.as_dict(), indent=2), encoding="utf-8")
    result = solve(cfg=config)
    if not np.isfinite(result.sol).all():
        raise RuntimeError("Forward solution contains non-finite values")
    summary = verify_outputs(output_dir)
    print(json.dumps(summary, indent=2))
    print(f"Verified forward impact. Open in ParaView: {output_dir / 'impact.pvd'}")
    return summary


if __name__ == "__main__":
    main()
