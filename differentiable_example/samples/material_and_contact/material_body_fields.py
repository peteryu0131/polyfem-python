"""Minimize compliance with a separate stiffness variable for each element of each body.

Run: python -m differentiable_example.samples.material_and_contact.material_body_fields
"""

import torch
import polyfem.diff as diff
import polyfem

from differentiable_example._scene import DATA_DIR

model = polyfem.model()
for index in range(2):
    body = model.mesh(mesh=str(DATA_DIR / "square.obj"))
    body.material(type="LinearElasticity", E=100., nu=.3)
    body.surface_box(box=[[-.501, -.501], [-.499, .501]], relative=False,
                     id=index + 1).dirichlet(value=[0., 0.])
    body.rhs(value=[0., -.1])

# Two independent, overlapping meshes; contact is disabled.
config = model.config(
    contact=polyfem.contact(enabled=False),
    space=polyfem.space(discr_order=1),
    solver=polyfem.solver(max_threads=1, linear=polyfem.linear(solver="Eigen::SimplicialLDLT")),
    output=polyfem.output(json="", log=polyfem.log(level="warning"),
                     paraview=polyfem.output_paraview(file_name="")),
)
obj = diff.material(config, objective=diff.Objective.COMPLIANCE)
x = torch.ones(obj.n_elements, dtype=torch.float64, requires_grad=True)


def loss(x):
    return obj(materials={
        body_id: {"E": 100. * x[list(indices)].pow(5)}
        for body_id, indices in obj.element_indices_by_body.items()
    })


optimizer = torch.optim.Adam([x], lr=.001)
optimizer.zero_grad()
value = loss(x)
value.backward()
optimizer.step()
print("Updated element E:", (100. * x.pow(5)).detach().tolist())
