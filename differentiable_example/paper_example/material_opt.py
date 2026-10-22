from pathlib import Path
import torch
from polyfempy import differentiable_api as diff
from polyfempy.generated_api import generated_api as pf

INITIAL_LATTICE_E_MPA = 20.
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "runs" / "material_opt_01"  # Change this folder name for a new run.

model = pf.model()
lattice = model.body().mesh(mesh=str(SCRIPT_DIR / "triangular_lattice.msh"))
lattice.material(
    type="NeoHookean",
    E={"value": INITIAL_LATTICE_E_MPA, "unit": "MPa"},
    nu=0.45,
    rho={"value": 1100, "unit": "kg/m^3"},
)
# Fix the bottom of the lattice.
lattice.surface_axis(axis=-2, position=0.0001, relative=True, id=3).dirichlet(value=[0, 0])
lattice.rhs(value=[0, 980])

block = model.body().mesh(mesh=str(SCRIPT_DIR / "falling_weight_block.msh"))
block.material(
    type="NeoHookean",
    E={"value": 200, "unit": "GPa"},
    nu=0.45,
    rho={"value": 7850, "unit": "kg/m^3"},
)
# Both meshes use boundary ID 3. Give the block ID 4 so it stays free to move.
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

# Save logs and solver statistics, but skip ParaView files.
output = pf.output(
    directory=str(OUTPUT_DIR), json="impact_stats.json",
    log=pf.log(level="info", file_level="debug", path="polyfem.log"),
    paraview=pf.output_paraview(file_name=""),
    advanced=pf.output_advanced(save_time_sequence=False, save_solve_sequence_debug=False),
)

config = model.config(
    units=pf.units(length="cm", mass="g", time="s"),
    space=pf.space(discr_order=1),
    time=pf.time(t0=0, dt=0.01, tend=0.5, integrator="BDF"),
    contact=pf.contact(enabled=True, dhat=0.012),
    solver=solver,
    output=output,
)

# Minimize the highest von Mises stress in the block over the simulation.
# The factor 1e-7 converts stress from the cm-g-s units to MPa.
objective = diff.objectives.von_mises_peak(selection=block, weight=1e-7)

# Stop if this output folder already exists, to keep earlier results.
OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
lattice_e_mpa = torch.tensor(INITIAL_LATTICE_E_MPA, dtype=torch.float64, requires_grad=True)

# Change only the lattice E. Keep all other material properties fixed.
problem = diff.material_opt(
    diff.model(config),
    materials={lattice: {"E": lattice_e_mpa, "unit": "MPa"}},
    objective=objective,
)

# Update E once. This is not a converged optimum.
problem.optimizer(torch.optim.Adam, lr=0.02).steps(1).optimize()
print(f"Updated lattice E: {lattice_e_mpa.item():.6f} MPa")
