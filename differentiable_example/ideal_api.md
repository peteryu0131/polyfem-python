# Differentiable API Ideal Examples

These examples show the target public API style. They are proposed examples,
not the current runnable backend smoke tests.

The goal is to keep the differentiable examples visually close to the existing
forward generated examples:

```python
from polyfempy.generated_api import generated_api as polyfem
from polyfempy import differentiable_api as diff
```

Future packaging can shorten this to:

```python
import polyfem
import polyfem.diff as diff
```

## Example 1 - Direct Vertex Shape Optimization

This is the differentiable version of the generated forward-example style:
build a normal PolyFEM model first, then wrap it with `diff.model(...)`.

```python
#!/usr/bin/env python3
"""2D shape optimization example using generated_api and differentiable_api.

This is the target public API shape.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

EXAMPLE_DIR = Path(__file__).resolve().parent
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from _diff_common import (  # noqa: E402
    DIFF_EXAMPLES_DIR,
    MESHES_DIR,
    polyfem,
    diff,
)


SOURCE_JSON = DIFF_EXAMPLES_DIR / "2D" / "shape-beam.json"
TARGET_U_PATH = DIFF_EXAMPLES_DIR / "2D" / "shape-beam-target.pt"

LEARNING_RATE = 1e-2
OPTIMIZATION_STEPS = 50

model = polyfem.model()

rubber = polyfem.neo_hookean(E=1e5, nu=0.3, rho=1150)

body = model.mesh(
    mesh=str(MESHES_DIR / "2D" / "beam.msh"),
    transformation=polyfem.transformation(scale=1.0),
)
body.material(rubber)
body.surface_box(
    id=1,
    min=[-0.01, -0.01],
    max=[0.01, 1.01],
).dirichlet(value=[0, 0])

solver = polyfem.solver(
    linear=polyfem.linear(solver="Eigen::PardisoLDLT"),
    nonlinear=polyfem.nonlinear(
        solver="Newton",
        iterations_per_strategy=1,
        line_search=polyfem.line_search(method="RobustArmijo"),
    ),
)

output = polyfem.output(
    json="sim.json",
    paraview=polyfem.output_paraview(
        file_name="sim.pvd",
        options=polyfem.options(
            material=True,
            body_ids=True,
            tensor_values=False,
            discretization_order=False,
            nodes=False,
        ),
    ),
    advanced=polyfem.output_advanced(
        save_solve_sequence_debug=False,
        save_time_sequence=True,
    ),
)

polyfem_config = model.config(
    rhs=[0, -9.8],
    time_tend=0.02,
    time_dt=0.01,
    space=polyfem.space(discr_order=2),
    solver=solver,
    output=output,
)

# Wrap the generated forward config for differentiable use.
# This does not run PolyFEM yet.
# The list form keeps the API ready for future multi-state optimization.
diff_model = diff.model([polyfem_config])

# target_u is the desired solution/displacement tensor.
# It can come from data, measurement, or another simulation.
target_u = torch.load(TARGET_U_PATH)


def shape_loss(vertices: torch.Tensor) -> torch.Tensor:
    # ShapeOpt is the torch.autograd.Function boundary:
    # forward: vertices -> PolyFEM solve -> u
    # backward: dloss/du -> PolyFEM adjoint -> dloss/dvertices
    u = diff.ShapeOpt.apply(
        diff_model,
        body,
        vertices,
    )
    return torch.nn.functional.mse_loss(u, target_u)


def main() -> int:
    # Create the differentiable shape tensor from the selected body.
    # requires_grad=True means PyTorch will store gradients in vertices.grad.
    vertices = diff.shape_vertices(
        diff_model,
        selection=body,
        requires_grad=True,
    )

    optimizer = torch.optim.Adam([vertices], lr=LEARNING_RATE)

    for step in range(OPTIMIZATION_STEPS):
        optimizer.zero_grad()
        loss = shape_loss(vertices)

        # loss.backward() triggers ShapeOpt.backward, which calls the
        # PolyFEM adjoint path and returns the shape gradient to PyTorch.
        loss.backward()
        optimizer.step()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Important points:

```text
polyfem.model() builds the same kind of forward config as the existing examples.
diff.model([polyfem_config]) wraps one or more forward configs for differentiation.
diff.ShapeOpt.apply(...) is the torch autograd boundary.
```

## Example 2 - User Parameters Mapped To Shape

The optimized variable does not have to be the full vertex array. The user can
optimize a smaller parameter vector and build vertices from it with torch.

```python
INITIAL_ANGLE = 0.0
INITIAL_THICKNESS = 0.1
INITIAL_HEIGHT = 1.0

params = torch.tensor(
    [INITIAL_ANGLE, INITIAL_THICKNESS, INITIAL_HEIGHT],
    dtype=torch.float64,
    requires_grad=True,
)

reference_vertices = diff.shape_vertices(
    diff_model,
    selection=body,
    requires_grad=False,
)


def build_vertices_from_params(params: torch.Tensor) -> torch.Tensor:
    angle, thickness, height = params

    # Keep this mapping fully differentiable in torch.
    # Do not call detach(), numpy(), item(), or use torch.no_grad().
    x = reference_vertices.clone()
    x[:, 0] = x[:, 0] * thickness
    x[:, 1] = x[:, 1] * height

    rotation = torch.stack(
        [
            torch.stack([torch.cos(angle), -torch.sin(angle)]),
            torch.stack([torch.sin(angle), torch.cos(angle)]),
        ]
    )
    return x @ rotation.T


def shape_loss_from_params(params: torch.Tensor) -> torch.Tensor:
    vertices = build_vertices_from_params(params)
    u = diff.ShapeOpt.apply(diff_model, body, vertices)
    return torch.nn.functional.mse_loss(u, target_u)


optimizer = torch.optim.Adam([params], lr=LEARNING_RATE)

for step in range(OPTIMIZATION_STEPS):
    optimizer.zero_grad()
    loss = shape_loss_from_params(params)
    loss.backward()
    optimizer.step()
```

The gradient chain is:

```text
loss -> u -> vertices -> params
```

PolyFEM handles `u -> vertices`. PyTorch handles `vertices -> params`.

## Example 3 - Future PolyFEM Objective Helper

For physical objectives such as max stress, stress norm, compliance, volume, or
contact objectives, the loss may need backend objective code.

```python
objective = diff.MaxStress(selection=body)


def shape_objective_loss(vertices: torch.Tensor) -> torch.Tensor:
    return diff.shape_objective(
        objective=objective,
        model=diff_model,
        selection=body,
        tensor=vertices,
    )
```

This should be a second route, separate from `ShapeOpt`:

```text
Route A:
    vertices -> ShapeOpt -> u -> PyTorch loss

Route B:
    vertices -> PolyFEM objective helper -> scalar loss
```

Current implementation should validate Route A first. Route B should come after
the backend objective value/derivative surface is confirmed.
