# Differentiable Simulations

This module provides differentiable simulation support for PolyFEM, enabling automatic gradient computation through the adjoint method.

For a file-by-file explanation of how the differentiable solve, objectives, and
optimization helpers connect, see [`CODE_MAP.md`](CODE_MAP.md).

## Quick Start

```python
import torch
import numpy as np
from polyfempy.differentiable import prepare_differentiable_simulation

# Setup mesh
V = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]], dtype=np.float64)
C = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

# Configuration
cfg = {
    "materials": [{"type": "LinearElasticity", "E": 1e6, "nu": 0.3}],
    "boundary_conditions": {
        "dirichlet_boundary": [{"selection": 1, "value": [0.0, 0.0]}],
        "neumann_boundary": [{"selection": 2, "value": [0.0, -1000.0]}],
    },
    "output": {"directory": "out", "paraview": {"file_name": "solution"}}
}

# Make vertices differentiable
vertices = torch.tensor(V, requires_grad=True)

# Prepare a differentiable simulation result
result = prepare_differentiable_simulation(vertices, C, cfg)

# Compute loss and gradient
loss = torch.norm(result.u)
loss.backward()
grad = vertices.grad  # Automatic gradient computation!
```

## Features

- **Automatic gradient computation**: No need to manually write `torch.autograd.Function`
- **Unified API**: Uses the same configuration format as `solve()`
- **Multiple derivative types**: Support for shape, material, initial condition derivatives
- **PyTorch integration**: Seamless integration with PyTorch's autograd

## Parameterized Shape API

The main optimization cleanup is that experiment-specific parameters such as
`h` and `theta_deg` are no longer a special optimization problem type.

Before this refactor, the h/theta experiment owned most of the chain:

```text
h, theta -> hard-coded HThetaOptimizationProblem -> vertices -> PolyFEM
```

After this refactor, the library owns the generic part:

```text
user parameters
  -> user-provided PyTorch vertex_map(...)
  -> vertices X
  -> PolyFEM shape autograd
  -> dL/dX
  -> PyTorch chain rule back to user parameters
```

PolyFEM still differentiates with respect to vertices. The new API makes the
`params -> vertices` map explicit, so PyTorch can compute gradients such as
`dL/dh` and `dL/dtheta_deg` without hand-writing those derivatives.

### Minimal Example

```python
import torch
from polyfempy.differentiable import (
    make_parameter,
    make_optimizer,
    make_von_mises_loss,
    prepare_parameterized_shape_problem,
    run_optimization,
)

torch.set_default_dtype(torch.float64)

# User design variables. These are normal torch Parameters with name/bounds
# metadata used by the wrapper.
h = make_parameter("h", 0.04, bounds=(0.03, 0.07))
theta_deg = make_parameter("theta_deg", 90.0, bounds=(60.0, 110.0))


def vertex_map(params, base_vertices, context):
    """Map user parameters to a fixed-topology vertex tensor.

    Replace this with the real geometry rule for a specific experiment. The
    important rule is that this function uses torch operations and returns the
    same vertex shape/topology as base_vertices.
    """
    h_value = params["h"]
    theta_value = params["theta_deg"]
    width_scale = theta_value / 90.0

    vertices = base_vertices.clone()
    vertices[:, 0] = base_vertices[:, 0] * width_scale
    vertices[:, 1] = base_vertices[:, 1] + h_value
    return vertices


problem = prepare_parameterized_shape_problem(
    cfg=cfg,
    parameters=[h, theta_deg],
    vertex_map=vertex_map,
)

optimizer = make_optimizer(problem, name="adam", lr=1e-2)
loss_fn = make_von_mises_loss(
    volume_selection=1,
    time_aggregation="smooth_max",
)

run_optimization(
    problem,
    steps=5,
    optimizer=optimizer,
    loss_fn=loss_fn,
    workspace="runs/parameterized_shape_example",
)
```

The user-provided `vertex_map` may also use a `context` dictionary for cached
indices, masks, or precomputed non-differentiable metadata. Do not detach or
convert differentiable values to NumPy inside the gradient path.

`prepare_parameterized_shape_problem(...)` is a user-friendly wrapper. It
builds the lower-level `ParameterizedVertexDesign`, passes a name dictionary to
`vertex_map`, uses parameter names for reports, and clamps parameters to their
optional bounds after each optimizer step.

### h/theta Example

The paper h/theta script can use the same API. Its experiment-specific part is
only the vertex map:

```python
h = make_parameter("h", 0.04, bounds=(0.03, 0.07))
theta_deg = make_parameter("theta_deg", 90.0, bounds=(60.0, 110.0))

problem = prepare_parameterized_shape_problem(
    cfg=cfg,
    parameters=[h, theta_deg],
    vertex_map=_h_theta_vertex_map,
    context={},
)
```

In that script, `HThetaVertexMap` is not a library concept anymore. It is just
one experiment's implementation of `vertex_map(params, base_vertices, context)`.
PolyFEM still computes the underlying shape derivative with respect to vertices
`X`; PyTorch chains that gradient back to `h` and `theta_deg`.

## API Reference

### `prepare_differentiable_simulation()`

Main user-facing function for one-off differentiable loss/gradient runs.

```python
result = prepare_differentiable_simulation(
    V,                    # Vertices (numpy or torch.Tensor), or None for config+mesh mode
    C,                    # Connectivity, or None for config+mesh mode
    cfg,                  # Configuration (dict or SimulationConfig)
    differentiable_params=["geometry"],  # Which parameters are differentiable
    derivative_type="shape",            # Type of derivative
)
```

`solve_differentiable()` is kept as the lower-level backward-compatible alias.

### `PolyFEMFunction`

Low-level PyTorch Function wrapper for advanced use cases.

### `DifferentiableResult`

Result container with PyTorch tensors that support `.backward()`.

## Requirements

- PyTorch (for gradient computation)
- PolyFEM C++ module (built with differentiable support)

## Notes

- The C++ module must be built with differentiable support
- Currently uses the old `pf.Solver()` API internally for direct access to differentiable features
