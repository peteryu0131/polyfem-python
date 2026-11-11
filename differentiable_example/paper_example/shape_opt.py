from pathlib import Path

import torch
import polyfem.diff as diff
import polyfem

from _shape_geometry import prepare_geometry


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "runs" / "shape_opt_01"  # Change this folder name for a new run.

model = polyfem.model()
lattice = model.body().mesh(mesh=str(SCRIPT_DIR / "triangular_lattice.msh"))
lattice.material(type="NeoHookean", E={"value": 20., "unit": "MPa"}, nu=.45,
                 rho={"value": 1100., "unit": "kg/m^3"})
lattice.surface_axis(axis=-2, position=.0001, relative=True, id=3).dirichlet(value=[0., 0.])
lattice.rhs(value=[0., 980.])

block = model.body().mesh(mesh=str(SCRIPT_DIR / "falling_weight_block.msh"))
block.material(type="NeoHookean", E={"value": 200., "unit": "GPa"}, nu=.45,
               rho={"value": 7850., "unit": "kg/m^3"})
block.surface_all(id=4)
block.rhs(value=[0., 980.])
block.velocity(value=[0., 0.])

# Trial fixed stiffness: shape peak gradients do not support adaptive contact.
# This value is not calibrated for the full impact experiment.
config = model.config(
    units=polyfem.units(length="cm", mass="g", time="s"),
    space=polyfem.space(discr_order=1),
    time=polyfem.time(t0=0., dt=.01, tend=.5, integrator="BDF"),
    contact=polyfem.contact(enabled=True, dhat=.012),
    solver=polyfem.solver(
        max_threads=1,
        linear=polyfem.linear(solver="Eigen::PardisoLDLT"),
        nonlinear=polyfem.nonlinear(solver="Newton", iterations_per_strategy=1,
                               grad_norm_tol=1e-8,
                               line_search=polyfem.line_search(method="RobustArmijo")),
        contact=polyfem.solver_contact(barrier_stiffness=1e8),
    ),
    output=polyfem.output(
        directory=str(OUTPUT_DIR), json="impact_stats.json",
        log=polyfem.log(level="info", file_level="debug", path="polyfem.log"),
        paraview=polyfem.output_paraview(file_name=""),
        advanced=polyfem.output_advanced(save_time_sequence=False, save_solve_sequence_debug=False),
    ),
)

# Keep earlier results. Prepare the fixed-topology geometry once.
OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
geometry = prepare_geometry(config, SCRIPT_DIR / "triangular_lattice.msh",
                            SCRIPT_DIR / "falling_weight_block.msh")
obj = diff.shape(config, selection=block, objective=diff.Objective.VON_MISES_PEAK)

# Normal leg thickness (cm) and full apex angle (degrees).
reference_design = torch.tensor([.04, 90.], dtype=torch.float64)
x = torch.ones(2, dtype=torch.float64, requires_grad=True)


def loss(x):
    thickness_cm, angle_deg = x * reference_design
    V = geometry(thickness_cm, angle_deg)
    return obj(V=V) * 1e-7


# Update both parameters once, using relative changes.
optimizer = torch.optim.Adam([x], lr=.001)
optimizer.zero_grad()
value = loss(x)
value.backward()
optimizer.step()

thickness_cm, angle_deg = (x * reference_design).detach()
print(f"Updated leg thickness: {thickness_cm.item():.6f} cm")
print(f"Updated apex angle: {angle_deg.item():.6f} degrees")
