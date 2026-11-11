"""Match the reference boundary positions by changing Young's modulus.

Run: python -m differentiable_example.samples.material_and_contact.material_matching
"""

import torch
import polyfem.diff as diff
import polyfem

from differentiable_example._scene import DATA_DIR


def configuration(E):
    model = polyfem.model()
    body = model.mesh(mesh=str(DATA_DIR / "square.obj"))
    body.material(type="LinearElasticity", E=E, nu=.3)
    body.surface_box(box=[[-.501, -.501], [-.499, .501]], relative=False, id=1).dirichlet(value=[0., 0.])
    body.rhs(value=[0., -1.])
    config = model.config(
        space=polyfem.space(discr_order=1),
        solver=polyfem.solver(max_threads=1, linear=polyfem.linear(solver="Eigen::SimplicialLDLT")),
        output=polyfem.output(json="", log=polyfem.log(level="warning"),
                         paraview=polyfem.output_paraview(file_name="")),
    )
    return config, body


config, body = configuration(100.)
target_config, _ = configuration(200.)
obj = diff.material(
    config, body=body, target=target_config,
    objective=diff.objectives.target(target_state=1),
)
x = torch.tensor([1.], dtype=torch.float64, requires_grad=True)


def loss(x):
    # Match boundary positions; the factor only scales the optimization loss.
    return obj(E=100. * x[0]) * 1e5


optimizer = torch.optim.Adam([x], lr=.02)
optimizer.zero_grad()
value = loss(x)
value.backward()
optimizer.step()
print(f"Updated E: {(100. * x[0]).item():.6f}")
