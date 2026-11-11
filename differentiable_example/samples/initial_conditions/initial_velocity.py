"""Fit the position target by changing the initial horizontal speed.

Run: python -m differentiable_example.samples.initial_conditions.initial_velocity
"""

import torch
import polyfem.diff as diff
import polyfem

from differentiable_example._scene import DATA_DIR

model = polyfem.model()
body = model.mesh(mesh=str(DATA_DIR / "square.obj"))
body.material(type="NeoHookean", E=100., nu=.3, rho=1.)
body.rhs(value=[0., 0.])

# A free unit-area square, with no contact or gravity.
config = model.config(
    space=polyfem.space(discr_order=1),
    time=polyfem.time(t0=0., dt=.05, tend=.1, integrator="BDF"),
    solver=polyfem.solver(
        max_threads=1,
        linear=polyfem.linear(solver="Eigen::SimplicialLDLT"),
        nonlinear=polyfem.nonlinear(grad_norm_tol=1e-10),
    ),
    output=polyfem.output(
        json="",
        log=polyfem.log(level="warning"),
        paraview=polyfem.output_paraview(file_name=""),
        advanced=polyfem.output_advanced(save_time_sequence=False),
    ),
)

# The position integral equals the mean x-position for this unit-area square.
obj = diff.initial(
    config, objective=diff.objectives.position(dim=0, selection=body).final(),
)
x = torch.tensor([1.], dtype=torch.float64, requires_grad=True)


def loss(x):
    velocity = torch.stack((x[0], x[0] * 0.))
    position = obj(velocity=velocity)
    return (position - .2).square()


# Adjust horizontal speed once to approach x = 0.2 at t = 0.1.
optimizer = torch.optim.Adam([x], lr=.02)
optimizer.zero_grad()
value = loss(x)
value.backward()
optimizer.step()
print(f"Updated horizontal speed: {x[0].item():.6f}")
