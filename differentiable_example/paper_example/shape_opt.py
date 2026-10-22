"""Optimize lattice thickness to reduce the block peak stress over the impact."""

import json
from pathlib import Path

from _lattice_geometry import LatticeGeometry

import torch

from polyfempy import differentiable_api as diff
from polyfempy.generated_api import generated_api as pf


INITIAL_E_MPA = 20.
OBJECTIVE_WEIGHT = 1e-7  # Internal stress -> MPa.
VALIDATION_THREADS = 1  # Reproducible baseline comparison; not a library-wide limit.
OPTIMIZATION_TOLERANCE = 1e-8
CONTACT_STIFFNESS = 4197560.464919463  # Initial value recorded in the original adaptive run; held fixed here.


HERE = Path(__file__).resolve().parent
DT = 0.01
END_TIME = 0.5
RUN_DIR = HERE / "runs" / "shape_thickness_01"  # Choose a new directory for each run.

# Forward model: both materials fixed; only lattice vertices will change.
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
    max_threads=VALIDATION_THREADS,
    linear=pf.linear(solver="Eigen::PardisoLDLT"),
    nonlinear=pf.nonlinear(
        solver="Newton", iterations_per_strategy=1,
        grad_norm_tol=OPTIMIZATION_TOLERANCE,
        line_search=pf.line_search(method="RobustArmijo"),
    ),
    contact=pf.solver_contact(barrier_stiffness=CONTACT_STIFFNESS),
)

output = pf.output(
    directory=str(RUN_DIR / "baseline"), json="impact_stats.json",
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

# Maximum block stress across every simulated step, expressed in MPa.
objective = diff.objectives.von_mises_peak(selection=block, weight=OBJECTIVE_WEIGHT)


if __name__ == "__main__":
    # Refuse to overwrite an earlier experiment. Choose a fresh RUN_DIR above.
    RUN_DIR.mkdir(parents=True, exist_ok=False)

    # The supplied mesh has 0.4 mm legs at 45 degrees (90-degree opening).
    # Change only interior leg thickness; joints taper smoothly, block stays fixed.
    geometry = LatticeGeometry(HERE)
    initial_thickness = geometry.initial_thickness  # cm, measured from the mesh
    thickness = torch.tensor(initial_thickness, dtype=torch.float64, requires_grad=True)
    angle = torch.tensor(geometry.initial_angle, dtype=torch.float64)  # fixed
    geometry.check(initial_thickness, float(angle))

    problem = diff.shape_opt(
        diff.model(config),
        parameters={"thickness": thickness},
        geometry=lambda thickness: geometry(thickness=thickness, angle=angle),
        objective=objective,
    )

    # One design update; each loss evaluation simulates all 50 physical time steps.
    learning_rate = initial_thickness * 0.001
    result = problem.optimizer(torch.optim.Adam, lr=learning_rate).steps(1).optimize()
    before = float(result.loss.detach())  # Loss BEFORE the update.

    # Equivalent explicit loop: use this INSTEAD of the three lines above.
    # optimizer = torch.optim.Adam(problem.parameters, lr=learning_rate)
    # optimizer.zero_grad()
    # loss = problem()
    # loss.backward()
    # before = float(loss.detach())
    # optimizer.step()

    # A proposed update is not a verified improvement until evaluated again.
    updated_thickness = float(thickness.detach())
    checks = geometry.check(updated_thickness, float(angle))
    config.output.directory = str(RUN_DIR / "candidate")
    candidate = diff.shape_opt(
        diff.model(config), parameters={"thickness": thickness},
        geometry=lambda thickness: geometry(thickness=thickness, angle=angle),
        objective=objective,
    )
    with torch.no_grad():
        after = float(candidate())

    summary = {
        "thickness_before_mm": initial_thickness * 10,
        "thickness_after_mm": updated_thickness * 10,
        "opening_angle_degrees": float(angle),
        "fixed_contact_stiffness": CONTACT_STIFFNESS,
        "block_peak_before_MPa": before,
        "block_peak_after_MPa": after,
        "accepted": after < before,
        **checks,
    }
    text = json.dumps(summary, indent=2, allow_nan=False)
    (RUN_DIR / "result.json").write_text(text, encoding="utf-8")
    print(text)
    # This checks one update, not convergence to an optimal thickness.
