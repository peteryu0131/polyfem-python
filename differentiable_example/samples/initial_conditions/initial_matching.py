"""Fit the reference motion by changing the initial velocity.

Run: python -m differentiable_example.samples.initial_conditions.initial_matching
"""

import torch
import polyfem
import polyfem.diff as diff

from differentiable_example._scene import DATA_DIR, example_output


def build_scene(mesh, velocity):
    model = polyfem.model()
    body = model.mesh(mesh=str(mesh), volume_selection=1)
    body.material(type="NeoHookean", E=100., nu=.3, rho=1.)
    body.rhs(value=[0., 0.])
    body.velocity(value=velocity)
    return model.config(
        space=polyfem.space(discr_order=1),
        time=polyfem.time(t0=0., dt=.05, tend=.1, integrator={"type": "BDF", "steps": 1}),
        solver=polyfem.solver(
            max_threads=1,
            linear=polyfem.linear(solver="Eigen::SimplicialLDLT"),
            nonlinear=polyfem.nonlinear(grad_norm_tol=1e-10),
        ),
        output=example_output(),
    )


mesh = DATA_DIR / "square.obj"
config = build_scene(mesh, velocity=[0., 0.])
target_config = build_scene(mesh, velocity=[2., 0.])

obj = diff.initial(
    config, target=target_config,
    objective=diff.objectives.target(target_state=1).final(),
)
x = torch.tensor([1., .2], dtype=torch.float64, requires_grad=True)


def loss(x):
    return obj(velocity=x)


optimizer = torch.optim.Adam([x], lr=.02)
optimizer.zero_grad()
value = loss(x)
value.backward()
optimizer.step()
print(f"Updated velocity: {x.detach().tolist()}")
