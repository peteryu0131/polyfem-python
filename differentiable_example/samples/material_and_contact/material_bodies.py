"""Minimize compliance with one stiffness multiplier per body.

Run: python -m differentiable_example.samples.material_and_contact.material_bodies
"""

import torch
import polyfem.diff as diff
import polyfem

from differentiable_example._scene import DATA_DIR

model = polyfem.model()
bodies = []
for index in range(2):
    body = model.mesh(mesh=str(DATA_DIR / "square.obj"))
    body.material(type="LinearElasticity", E=100., nu=.3)
    body.surface_box(box=[[-.501, -.501], [-.499, .501]], relative=False,
                     id=index + 1).dirichlet(value=[0., 0.])
    body.rhs(value=[0., -.1])
    bodies.append(body)

# Two independent, overlapping meshes; contact is disabled.
config = model.config(
    contact=polyfem.contact(enabled=False),
    space=polyfem.space(discr_order=1),
    solver=polyfem.solver(max_threads=1, linear=polyfem.linear(solver="Eigen::SimplicialLDLT")),
    output=polyfem.output(json="", log=polyfem.log(level="warning"),
                     paraview=polyfem.output_paraview(file_name="")),
)
obj = diff.material(config, objective=diff.Objective.COMPLIANCE)
x = torch.ones(2, dtype=torch.float64, requires_grad=True)


def loss(x):
    return obj(materials={
        bodies[0]: {"E": 100. * x[0]},
        bodies[1]: {"E": 100. * x[1]},
    })


optimizer = torch.optim.Adam([x], lr=.001)
optimizer.zero_grad()
value = loss(x)
value.backward()
optimizer.step()
print("Updated body E:", (100. * x).detach().tolist())
