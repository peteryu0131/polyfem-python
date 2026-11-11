"""Fit the position target by changing the pressure on boundary 2.

Run: python -m differentiable_example.samples.boundary_controls.pressure
"""

import torch

import polyfem
import polyfem.diff as diff
from differentiable_example._scene import DATA_DIR, control_solver, example_output


def build_scene(mesh):
    model = polyfem.model()
    body = model.mesh(mesh=str(mesh))
    body.material(type="NeoHookean", E=100., nu=.3, rho=1.)
    left = body.surface_box(box=[[-.501, -.501], [-.499, .501]], relative=False, id=1)
    left.dirichlet(value=[0., 0.])
    body.surface_box(box=[[-1., -1.], [1., 1.]], relative=False, id=2)
    body.rhs(value=[0., 0.])
    return model.config(
        space=polyfem.space(discr_order=1),
        solver=control_solver(),
        boundary_conditions=polyfem.boundary_conditions(pressure_boundary=[{"id": 2, "value": 0.}]),
        output=example_output(),
    )


if __name__ == "__main__":
    config = build_scene(DATA_DIR / "square.obj")
    obj = diff.pressure(
        config, boundary_ids=[2], objective=diff.objectives.position(dim=0, selection=1),
    )
    x = torch.tensor([1.], dtype=torch.float64, requires_grad=True)

    def loss(x):
        return (obj(values=x) - .01).square()

    optimizer = torch.optim.Adam([x], lr=.001)
    optimizer.zero_grad()
    value = loss(x)
    value.backward()
    optimizer.step()
    print('Updated pressure:', x.detach().tolist())
