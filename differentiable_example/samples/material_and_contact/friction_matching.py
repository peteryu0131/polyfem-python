"""Fit the reference state by changing the friction coefficient.

Run: python -m differentiable_example.samples.material_and_contact.friction_matching
"""

import torch
import polyfem
import polyfem.diff as diff
from differentiable_example._scene import DATA_DIR, control_solver, example_output


def build_scene(mesh, *, coefficient):
    model = polyfem.model()
    body = model.mesh(mesh=str(mesh), surface_selection=1)
    body.material(type="NeoHookean", E=100., nu=.3, rho=1.)
    body.velocity(value=[.5, 0.])
    model.obstacle_mesh(
        mesh=str(mesh), surface_selection=3,
        transformation=polyfem.transformation(translation=[0., -1.01]),
    )
    body.rhs(value=[0., -1.])
    time = polyfem.time(dt=.05, tend=.15, integrator={"type": "BDF", "steps": 1})
    return model.config(
        space=polyfem.space(discr_order=1),
        time=time,
        contact=polyfem.contact(enabled=True, dhat=.05, friction_coefficient=coefficient),
        solver=control_solver(contact=polyfem.solver_contact(barrier_stiffness=100.)),
        output=example_output(),
    )


mesh = DATA_DIR / "square.obj"
config = build_scene(mesh, coefficient=.3)
target_config = build_scene(mesh, coefficient=.5)

obj = diff.friction(
    config, target=target_config,
    objective=diff.objectives.target(target_state=1, selection=1).final(),
)
x = torch.tensor(.3, dtype=torch.float64, requires_grad=True)


def loss(x):
    return obj(coefficient=x)


optimizer = torch.optim.Adam([x], lr=.001)
optimizer.zero_grad()
value = loss(x)
value.backward()
optimizer.step()
print(f'Updated friction: {x.item():.6f}')
