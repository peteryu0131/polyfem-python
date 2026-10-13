#!/usr/bin/env python3
"""Optimize a launch angle: start at 20 degrees and approach 45 degrees.

Run from the repository root in the polyfem environment:
    python differentiable_example/initial_condition/projectile_angle.py

Edit the settings below, then read loss(theta) and the ordinary PyTorch loop.
We reconstruct equal-height landing from a simulated position under constant
gravity. This benchmark has no drag or ground contact; it is not an event solver.
Details: docs/projectile-angle-example.md
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from polyfempy import differentiable_api as diff
from polyfempy.generated_api import generated_api as polyfem


EXAMPLE_DIR = Path(__file__).resolve().parent
MESH_PATH = EXAMPLE_DIR.parents[1] / "polyfem-data/contact/meshes/2D/simple/square.obj"
OUTPUT_DIR = EXAMPLE_DIR / "runs" / "projectile_angle"

# User settings: normalized units; angles are displayed in degrees.
START_ANGLE_DEGREES = 20.0
SPEED = 1.0
GRAVITY = 1.0
STEPS = 8
LEARNING_RATE = 0.2  # Step size relative to the characteristic range SPEED**2/GRAVITY.
DT = 0.01
PROBE_TIME = 0.2  # Simulate to this time, then reconstruct the ballistic landing.

# Check settings before preparing the backend or starting a long solve.
for name, value in {
    "SPEED": SPEED, "GRAVITY": GRAVITY, "DT": DT,
    "PROBE_TIME": PROBE_TIME, "LEARNING_RATE": LEARNING_RATE,
}.items():
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive; got {value!r}.")
if not 5.0 <= START_ANGLE_DEGREES <= 85.0:
    raise ValueError("START_ANGLE_DEGREES must be between 5 and 85 degrees.")
if isinstance(STEPS, bool) or not isinstance(STEPS, int) or STEPS < 1:
    raise ValueError("STEPS must be a positive integer.")
if PROBE_TIME < DT or not math.isclose(PROBE_TIME / DT, round(PROBE_TIME / DT), rel_tol=0.0, abs_tol=1e-9):
    raise ValueError("PROBE_TIME must be a positive integer multiple of DT.")
# BDF1's reconstructed flight time is 2*SPEED*sin(theta)/GRAVITY - DT.
# Require a positive return time throughout the allowed 5..85 degree search.
if GRAVITY * DT >= 2 * SPEED * math.sin(math.radians(5.0)):
    raise ValueError("DT is too large for this SPEED and GRAVITY at 5 degrees; reduce DT.")


# Forward model: a free, unit-area square, with no fixed boundaries or contact.
if not MESH_PATH.is_file():
    raise FileNotFoundError(
        f"Missing example mesh: {MESH_PATH}. "
        "Run 'git submodule update --init polyfem-data' from the repository root."
    )

model = polyfem.model()
body = polyfem.body(model=model)
body.mesh(polyfem.mesh(mesh=str(MESH_PATH)))
body.material(polyfem.neo_hookean(E=1.0e4, nu=0.3, rho=1.0))
# The backend residual convention makes positive RHS-y accelerate down.
body.rhs(value=[0.0, GRAVITY])
polyfem_config = polyfem.config(
    model=model,
    time=polyfem.time(polyfem.object1(dt=DT, tend=PROBE_TIME, integrator="BDF")),
    space=polyfem.space(discr_order=1),
    contact=polyfem.contact(enabled=False),
    solver=polyfem.solver(
        max_threads=1,
        linear=polyfem.linear(solver="Eigen::SimplicialLDLT"),
        nonlinear=polyfem.nonlinear(
            solver="Newton", iterations_per_strategy=1,
            line_search=polyfem.line_search(method="RobustArmijo"),
        ),
        advanced=polyfem.solver_advanced(lump_mass_matrix=True),
    ),
    output=polyfem.output(
        directory=str(OUTPUT_DIR), json="", log=polyfem.log(level="warning"),
        paraview=polyfem.output_paraview(file_name=""),
        advanced=polyfem.output_advanced(
            save_solve_sequence_debug=False, save_time_sequence=False,
        ),
    ),
)

# Differentiable model: these API operations also work in other examples.
diff_model = diff.model([polyfem_config])
metadata = diff_model.metadata()
dof_count = metadata.n_dofs
position_x = diff.objectives.position(dim=0, selection=body).final()
position_y = diff.objectives.position(dim=1, selection=body).final()


def loss(theta: torch.Tensor) -> torch.Tensor:
    """One angle in radians -> negative range. Minimizing it maximizes range."""
    # g(theta): give every node the same velocity in this free 2D P1 mesh.
    velocity = SPEED * torch.stack((torch.cos(theta), torch.sin(theta)))
    initial = metadata.uniform_initial_condition(velocity)

    # P(g(theta)): two scalar solves. For this unit-area square, integral = mean.
    x = diff.initial_condition_opt(diff_model, initial, objective=position_x)()
    y = diff.initial_condition_opt(diff_model, initial, objective=position_y)()

    # f(P): reconstruct x=a*t, y=b*t-g*t^2/2; return to initial height at 2*b/g.
    # Keep flight_time in the PyTorch graph so its angle derivative is included.
    horizontal_speed = x / PROBE_TIME
    vertical_coefficient = y / PROBE_TIME + 0.5 * GRAVITY * PROBE_TIME
    flight_time = 2 * vertical_coefficient / GRAVITY
    if float(flight_time.detach()) <= 0:
        raise ValueError("Launch must have a positive return time; increase the angle.")
    return -horizontal_speed * flight_time


def main():
    # Ordinary PyTorch: the optimizer owns only theta.
    theta = torch.tensor(math.radians(START_ANGLE_DEGREES), dtype=torch.float64, requires_grad=True)
    # Range and its angle gradient scale with SPEED**2/GRAVITY. Normalize the
    # optimizer step so changing physical units does not amplify or shrink it.
    range_scale = SPEED ** 2 / GRAVITY
    optimizer = torch.optim.SGD([theta], lr=LEARNING_RATE / range_scale)
    history = []
    print("Maximize equal-height landing range (continuous reference: 45 degrees).", flush=True)
    for step in range(STEPS):
        optimizer.zero_grad()
        value = loss(theta)
        value.backward()
        history.append({
            "step": step, "angle_degrees": math.degrees(float(theta.detach())),
            "range": -float(value.detach()), "angle_gradient": float(theta.grad),
        })
        print(f"Step {step:2d}: angle={history[-1]['angle_degrees']:7.3f} deg, "
              f"range={history[-1]['range']:.6f}", flush=True)
        optimizer.step()
        # Restrict the search to upward launches; the optimum is interior.
        with torch.no_grad():
            theta.clamp_(math.radians(5), math.radians(85))
    with torch.no_grad():
        final_range = -float(loss(theta))
    summary = {
        "objective": "negative_ballistic_reconstructed_range",
        "angle_degrees": math.degrees(float(theta.detach())),
        "range": final_range,
        "reference_angle_degrees": 45.0,
        "reference_max_range": SPEED ** 2 / GRAVITY,
        "start_angle_degrees": START_ANGLE_DEGREES,
        "speed": SPEED, "gravity": GRAVITY, "steps": STEPS,
        "learning_rate": LEARNING_RATE,
        "effective_learning_rate": LEARNING_RATE / range_scale,
        "dt": DT, "probe_time": PROBE_TIME, "dof_count": dof_count,
        "history": history,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Final: {summary['angle_degrees']:.3f} deg, range={final_range:.6f}")
    print(f"Results saved to {summary_path}")


if __name__ == "__main__":
    main()
