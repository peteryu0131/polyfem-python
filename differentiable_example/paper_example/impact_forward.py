#!/usr/bin/env python3
"""Forward-only lattice impact. Run from the repository root in the polyfem env."""

import json
from pathlib import Path

from polyfempy import solve
from polyfempy.generated_api import generated_api as pf


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "runs" / "impact_forward"
DT = 0.01
END_TIME = 0.5


def build_config(output_dir=OUTPUT_DIR):
    model = pf.model()
    lattice = model.body().mesh(mesh=str(HERE / "triangular_lattice.msh"))
    lattice.material(
        type="NeoHookean",
        E={"value": 20, "unit": "MPa"}, nu=0.45,
        rho={"value": 1100, "unit": "kg/m^3"},
    )
    lattice.surface_axis(axis=-2, position=0.0001, relative=True, id=3).dirichlet(value=[0, 0])
    lattice.rhs(value=[0, 980])

    block = model.body().mesh(mesh=str(HERE / "falling_weight_block.msh"))
    block.material(
        type="NeoHookean",
        E={"value": 200, "unit": "GPa"}, nu=0.45,
        rho={"value": 7850, "unit": "kg/m^3"},
    )
    # Both original meshes contain physical boundary id 3. Override the block's
    # imported tags so fixing lattice bottom id 3 does not also pin the block.
    block.surface_all(id=4)
    block.rhs(value=[0, 980])
    block.velocity(value=[0, 0])

    return model.config(
        units=pf.units(length="cm", mass="g", time="s"),
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


def main(output_dir=OUTPUT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = build_config(output_dir)
    (output_dir / "config.json").write_text(json.dumps(config.as_dict(), indent=2), encoding="utf-8")
    result = solve(cfg=config)
    print(f"Forward solve completed. Open in ParaView: {output_dir / 'impact.pvd'}")
    return result


if __name__ == "__main__":
    main()
