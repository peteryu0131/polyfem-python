"""Minimize the sum of two load-case compliances using one shared stiffness field.

Run: python -m differentiable_example.samples.material_and_contact.material_load_cases
"""

import torch
import polyfem.diff as diff
import polyfem

from differentiable_example._scene import DATA_DIR


def load_case(force):
    model = polyfem.model()
    body = model.mesh(mesh=str(DATA_DIR / "square.obj"))
    body.material(type="LinearElasticity", E=100., nu=.3)
    body.surface_box(box=[[-.501, -.501], [-.499, .501]], relative=False,
                     id=1).dirichlet(value=[0., 0.])
    body.rhs(value=force)
    config = model.config(
        space=polyfem.space(discr_order=1),
        solver=polyfem.solver(max_threads=1, linear=polyfem.linear(solver="Eigen::SimplicialLDLT")),
        output=polyfem.output(json="", log=polyfem.log(level="warning"),
                         paraview=polyfem.output_paraview(file_name="")),
    )
    return diff.material(config, body=body, objective=diff.Objective.COMPLIANCE)


vertical = load_case([0., -.1])
horizontal = load_case([.1, 0.])
x = torch.ones(vertical.n_elements, dtype=torch.float64, requires_grad=True)


def loss(x):
    E = 100. * x.pow(5)
    return vertical(E=E) + horizontal(E=E)


# Two independent solves share the same element stiffness parameters.
optimizer = torch.optim.Adam([x], lr=.001)
optimizer.zero_grad()
value = loss(x)
value.backward()
optimizer.step()
print("Updated element E:", (100. * x.pow(5)).detach().tolist())
