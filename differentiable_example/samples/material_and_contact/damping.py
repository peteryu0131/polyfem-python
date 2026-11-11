"""Bring the final vertical position objective toward zero by changing psi and phi.

Run: python -m differentiable_example.samples.material_and_contact.damping
"""

import torch

import polyfem
import polyfem.diff as diff
from differentiable_example._scene import DATA_DIR, control_solver, example_output


def build_scene(mesh, *, psi=0., phi=0.):
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


if __name__ == "__main__":
    config = build_scene(DATA_DIR / "square.obj")
    obj = diff.damping(
        config, objective=diff.objectives.position(dim=1, selection=1).final(),
    )
    x = torch.tensor([.1, .1], dtype=torch.float64, requires_grad=True)

    def loss(x):
        position = obj(psi=x[0], phi=x[1])
        return position.square()

    optimizer = torch.optim.Adam([x], lr=.001)
    optimizer.zero_grad()
    value = loss(x)
    value.backward()
    optimizer.step()
    print('Updated psi, phi:', x.detach().tolist())
