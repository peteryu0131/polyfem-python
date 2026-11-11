"""Minimize compliance with one stiffness variable per element.

Run: python -m differentiable_example.samples.material_and_contact.material_field
"""

import torch
import polyfem.diff as diff
import polyfem

from differentiable_example._scene import DATA_DIR

model = polyfem.model()
body = model.mesh(mesh=str(DATA_DIR / "square.obj"))
body.material(type="LinearElasticity", E=100., nu=.3)
body.surface_box(box=[[-.501, -.501], [-.499, .501]], relative=False, id=1).dirichlet(value=[0., 0.])
body.rhs(value=[0., -.1])
config = model.config(
    space=polyfem.space(discr_order=1),
    solver=polyfem.solver(
        max_threads=1,
        linear=polyfem.linear(solver="Eigen::SimplicialLDLT"),
    ),
    output=polyfem.output(
        json="", log=polyfem.log(level="warning"),
        paraview=polyfem.output_paraview(file_name=""),
    ),
)

obj = diff.material(config, body=body, objective=diff.Objective.COMPLIANCE)
x = torch.ones(obj.n_elements, dtype=torch.float64, requires_grad=True)


def loss(x):
    E = 100. * x.pow(5)
    return obj(E=E, nu=.3)


# One independent stiffness update per element; no material budget is imposed.
optimizer = torch.optim.Adam([x], lr=.001)
optimizer.zero_grad()
value = loss(x)
value.backward()
optimizer.step()
print("Updated element E:", (100. * x.pow(5)).detach().tolist())
