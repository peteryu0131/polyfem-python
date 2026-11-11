"""Fit the reference state by changing the damping coefficients psi and phi.

Run: python -m differentiable_example.samples.material_and_contact.damping_matching
"""

import torch
import polyfem
import polyfem.diff as diff
from differentiable_example._scene import DATA_DIR, control_solver, example_output


def build_scene(mesh, *, psi, phi):
    model = polyfem.model()
    body = model.mesh(mesh=str(mesh))
    body.material(type="NeoHookean", E=100., nu=.3, rho=1., psi=psi, phi=phi)
    left = body.surface_box(box=[[-.501, -.501], [-.499, .501]], relative=False, id=1)
    left.dirichlet(value=[0., 0.])
    body.rhs(value=[0., -1.])
    time = polyfem.time(dt=.05, tend=.15, integrator={"type": "BDF", "steps": 1})
    return model.config(
        space=polyfem.space(discr_order=1),
        time=time,
        solver=control_solver(),
        output=example_output(),
    )


mesh = DATA_DIR / "square.obj"
config = build_scene(mesh, psi=0., phi=0.)
target_config = build_scene(mesh, psi=.4, phi=.6)

obj = diff.damping(
    config, target=target_config,
    objective=diff.objectives.target(target_state=1).final(),
)
x = torch.tensor([.1, .2], dtype=torch.float64, requires_grad=True)


def loss(x):
    return obj(psi=x[0], phi=x[1])


optimizer = torch.optim.Adam([x], lr=.01)
optimizer.zero_grad()
value = loss(x)
value.backward()
optimizer.step()
print(f'Updated damping [psi, phi]: {x.detach().tolist()}')
