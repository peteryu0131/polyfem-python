"""Optimize lattice E with one Adam update."""

from pathlib import Path

import torch

from polyfempy import differentiable_api as diff
from polyfempy.generated_api import generated_api as pf


INITIAL_E_MPA = 20.


HERE = Path(__file__).resolve().parent
DT = 0.01
END_TIME = 0.5
RUN_DIR = HERE / "runs" / "material_opt_01"  # Choose a new directory for each run.

# Forward model: fixed block material, trainable lattice E.
model = pf.model()
lattice = model.body().mesh(mesh=str(HERE / "triangular_lattice.msh"))
lattice.material(
    type="NeoHookean",
    E={"value": INITIAL_E_MPA, "unit": "MPa"}, nu=0.45,
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

solver = pf.solver(
    max_threads=1,
    linear=pf.linear(solver="Eigen::PardisoLDLT"),
    nonlinear=pf.nonlinear(
        solver="Newton", iterations_per_strategy=1,
        grad_norm_tol=1e-8,
        line_search=pf.line_search(method="RobustArmijo"),
    ),
    contact=pf.solver_contact(barrier_stiffness="adaptive"),
)

output = pf.output(
    directory=str(RUN_DIR / "gradient"), json="impact_stats.json",
    log=pf.log(level="info", file_level="debug", path="polyfem.log"),
    paraview=pf.output_paraview(file_name=""),
    advanced=pf.output_advanced(save_time_sequence=False, save_solve_sequence_debug=False),
)

config = model.config(
    units=pf.units(length="cm", mass="g", time="s"),
    space=pf.space(discr_order=1),
    time=pf.time(pf.object1(t0=0, dt=DT, tend=END_TIME, integrator="BDF")),
    contact=pf.contact(enabled=True, dhat=0.012),
    solver=solver,
    output=output,
)

# Maximum block stress across all time steps; 1e-7 converts internal stress to MPa.
objective = diff.objectives.von_mises_peak(selection=block, weight=1e-7)


if __name__ == "__main__":
    # Choose a new directory; never overwrite an earlier experiment.
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    lattice_E_MPa = torch.tensor(INITIAL_E_MPA, dtype=torch.float64, requires_grad=True)

    # Only lattice E is trainable. nu and all block properties inherit the config.
    problem = diff.material_opt(
        diff.model(config),
        materials={lattice: {"E": lattice_E_MPa, "unit": "MPa"}},
        objective=objective,
    )

    problem.optimizer(torch.optim.Adam, lr=0.02).steps(1).optimize()

    print(f"Updated lattice E: {lattice_E_MPa.item():.6f} MPa")
