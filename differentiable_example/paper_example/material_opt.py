from pathlib import Path
import torch
import polyfem.diff as diff
import polyfem

INITIAL_LATTICE_E_MPA = 20.
INITIAL_LATTICE_NU = 0.45
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "runs" / "material_opt_01"  # Change this folder name for a new run.

model = polyfem.model()
lattice = model.body().mesh(mesh=str(SCRIPT_DIR / "triangular_lattice.msh"))
lattice.material(
    type="NeoHookean",
    E={"value": INITIAL_LATTICE_E_MPA, "unit": "MPa"},
    nu=INITIAL_LATTICE_NU,
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

solver = polyfem.solver(
    max_threads=1,
    linear=polyfem.linear(solver="Eigen::PardisoLDLT"),
    nonlinear=polyfem.nonlinear(
        solver="Newton", iterations_per_strategy=1,
        grad_norm_tol=1e-8,
        line_search=polyfem.line_search(method="RobustArmijo"),
    ),
    contact=polyfem.solver_contact(barrier_stiffness="adaptive"),
)

# Save logs and solver statistics, but skip ParaView files.
output = polyfem.output(
    directory=str(OUTPUT_DIR), json="impact_stats.json",
    log=polyfem.log(level="info", file_level="debug", path="polyfem.log"),
    paraview=polyfem.output_paraview(file_name=""),
    advanced=polyfem.output_advanced(save_time_sequence=False, save_solve_sequence_debug=False),
)

config = model.config(
    units=polyfem.units(length="cm", mass="g", time="s"),
    space=polyfem.space(discr_order=1),
    time=polyfem.time(t0=0, dt=0.01, tend=0.5, integrator="BDF"),
    contact=polyfem.contact(enabled=True, dhat=0.012),
    solver=solver,
    output=output,
)

# Stop if this output folder already exists, to keep earlier results.
OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
x = torch.tensor([INITIAL_LATTICE_E_MPA, INITIAL_LATTICE_NU],
                 dtype=torch.float64, requires_grad=True)

obj = diff.material(
    config,
    body=lattice,
    selection=block,
    objective=diff.Objective.VON_MISES_PEAK,
)
o = torch.ones(obj.n_elements, dtype=torch.float64)


def loss(x):
    # Convert E from MPa to cm-g-s stress units, and the result back to MPa.
    E = x[0] * 1e7 * o
    nu = x[1] * o
    return obj(E=E, nu=nu) * 1e-7


# Update E and nu once. This is not a converged optimum.
optimizer = torch.optim.Adam([x], lr=0.001)
optimizer.zero_grad()
value = loss(x)
value.backward()
optimizer.step()
print(f"Updated lattice E: {x[0].item():.6f} MPa")
print(f"Updated lattice nu: {x[1].item():.6f}")
