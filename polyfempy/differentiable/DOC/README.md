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

## API Boundary

Most user scripts should stay on this small surface:

```python
from polyfempy.differentiable import (
    make_optimizer,
    make_von_mises_loss,
    prepare_differentiable_simulation,
    prepare_optimization_problem,
    prepare_parameterized_shape_problem,
    run_optimization,
)
```

Lower-level diagnostics, finite-difference checks, and the older torch-bridge
probe helpers live under:

```python
from polyfempy.differentiable import advanced
```

Some advanced helpers are still importable from `polyfempy.differentiable` for
backward compatibility, but new examples should not rely on them as the normal
API path.

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
    make_optimizer,
    make_von_mises_loss,
    prepare_parameterized_shape_problem,
    run_optimization,
)

torch.set_default_dtype(torch.float64)

# User design variables are plain PyTorch Parameters.
h = torch.nn.Parameter(torch.tensor(0.04))
theta_deg = torch.nn.Parameter(torch.tensor(90.0))


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
    parameter_names=["h", "theta_deg"],
    bounds={"h": (0.03, 0.07), "theta_deg": (60.0, 110.0)},
    vertex_map=vertex_map,
)

optimizer = make_optimizer(problem, name="adam", lr=1e-2)
loss_fn = make_von_mises_loss(
    body=1,
    time="smooth_max",
)

run = run_optimization(
    problem,
    steps=5,
    optimizer=optimizer,
    loss_fn=loss_fn,
    workspace="runs/parameterized_shape_example",
    return_result=True,
)
print(run.final_loss)
```

### Vertex Map Contract

The user-provided `vertex_map` is the only experiment-specific shape rule:

```python
def vertex_map(params, base_vertices, context):
    ...
    return vertices
```

Inputs:

```text
params:
  dict[str, torch.Tensor] by default, keyed by parameter_names

base_vertices:
  fixed-topology mesh vertices loaded from cfg

context:
  mutable cache for masks, index arrays, or other non-differentiable metadata
```

Output:

```text
vertices:
  torch.Tensor with the same shape as base_vertices
```

Rules:

```text
use torch operations for differentiable math
keep vertex count and cell topology fixed
do not remesh inside vertex_map
do not detach or convert differentiable values to NumPy
put cached masks or precomputed index sets in context
```

The API validates this contract before calling PolyFEM. Common mistakes fail
early with user-facing errors:

```text
wrong return type:
  vertex_map must return a torch.Tensor, got ndarray

changed topology:
  vertex_map must return vertices with shape (...), got (...)

broken autograd chain:
  vertex_map returned vertices that are not connected to differentiable parameters

invalid geometry:
  vertex_map returned vertices containing NaN or Inf values
```

The parameter names are arbitrary. `h` and `theta_deg` are examples, not API
requirements.

`prepare_parameterized_shape_problem(...)` is a user-friendly wrapper. It
builds the lower-level `ParameterizedVertexDesign`, passes a name dictionary to
`vertex_map`, uses parameter names for reports, and clamps parameters to their
optional bounds after each optimizer step. Plain `torch.nn.Parameter` objects
are the recommended inputs. The older `make_parameter(...)` helper remains
available for compatibility when a script wants to attach name/bounds metadata
directly to the parameter object.

The lower-level `prepare_parameterized_shape_optimization_problem(...)` remains
available for advanced callers that already own a `ParameterizedVertexDesign`
or need custom `parameter_map` / `project` plumbing. New examples should prefer
`prepare_parameterized_shape_problem(...)`.

### h/theta Example

The paper h/theta script can use the same API. Its experiment-specific part is
only the vertex map:

```python
h = torch.nn.Parameter(torch.tensor(0.04))
theta_deg = torch.nn.Parameter(torch.tensor(90.0))

problem = prepare_parameterized_shape_problem(
    cfg=cfg,
    parameters=[h, theta_deg],
    parameter_names=["h", "theta_deg"],
    bounds={"h": (0.03, 0.07), "theta_deg": (60.0, 110.0)},
    vertex_map=_h_theta_vertex_map,
    context={},
)
```

In that script, `HThetaVertexMap` is not a library concept anymore. It is just
one experiment's implementation of `vertex_map(params, base_vertices, context)`.
PolyFEM still computes the underlying shape derivative with respect to vertices
`X`; PyTorch chains that gradient back to `h` and `theta_deg`.

### Non h/theta Example

This uses two unrelated design parameters: a vertical scale on one selected
body and a horizontal shift. The names and formulas are entirely user-defined:

```python
lattice_y_scale = torch.nn.Parameter(torch.tensor(1.0))
block_shift_x = torch.nn.Parameter(torch.tensor(0.0))


def vertex_map(params, base_vertices, context):
    lattice_mask = context.get("lattice_mask")
    block_mask = context.get("block_mask")
    if lattice_mask is None:
        y = base_vertices[:, 1]
        lattice_mask = y <= 4.98
        block_mask = ~lattice_mask
        context["lattice_mask"] = lattice_mask
        context["block_mask"] = block_mask

    vertices = base_vertices.clone()
    vertices[lattice_mask, 1] = base_vertices[lattice_mask, 1] * params["lattice_y_scale"]
    vertices[block_mask, 0] = base_vertices[block_mask, 0] + params["block_shift_x"]
    return vertices


problem = prepare_parameterized_shape_problem(
    cfg=cfg,
    parameters=[lattice_y_scale, block_shift_x],
    parameter_names=["lattice_y_scale", "block_shift_x"],
    bounds={"lattice_y_scale": (0.98, 1.02), "block_shift_x": (-0.10, 0.10)},
    vertex_map=vertex_map,
    context={},
)
```

After `loss.backward()`, PyTorch gives:

```text
lattice_y_scale.grad = dL/d lattice_y_scale
block_shift_x.grad   = dL/d block_shift_x
```

## Scalar Material API

The material side has the same user-facing parameter style for the common
scalar Young's modulus case:

```python
import torch
from polyfempy.differentiable import (
    make_optimizer,
    make_von_mises_loss,
    prepare_optimization_problem,
    run_optimization,
)

E_lattice = torch.nn.Parameter(torch.tensor(20.0))

problem = prepare_optimization_problem(
    cfg=cfg,
    kind="material",
    body_id=1,
    E_parameter=E_lattice,
    parameter_name="E_lattice_MPa",
    bounds=(1.0, None),
    E_unit="MPa",
)

optimizer = make_optimizer(problem, name="adam", lr=1e-2)
loss_fn = make_von_mises_loss(
    body=1,
    time="smooth_max",
)

run = run_optimization(
    problem,
    steps=5,
    optimizer=optimizer,
    loss_fn=loss_fn,
    workspace="runs/scalar_material_example",
    return_result=True,
)
print(run.final_loss)
```

This path optimizes the physical scalar `E_lattice_MPa` directly. Internally,
the solver still receives Lamé `lambda/mu` tensors because that is the current
PolyFEM material backward interface.

`make_von_mises_loss(...)` works for both shape and material differentiable
results. It detects material results and routes them to the elastic objective
bridge internally. The older `make_material_von_mises_loss(...)` helper remains
available for compatibility and explicit advanced use.

`body=...` and `time=...` are user-facing aliases. The older
`volume_selection=...` and `time_aggregation=...` names still work for
compatibility and for callers that want to use PolyFEM's lower-level wording.

The older `prepare_material_optimization_problem(...)` remains available and
optimizes an unconstrained internal `log_E` through a softplus transform. It is
kept for compatibility and for experiments that prefer that parameterization.

`run_optimization(...)` returns the legacy list of step records by default.
New code can request a stable result object:

```python
run = run_optimization(..., return_result=True)
run.steps
run.final_step
run.final_loss
run.summary()
```

The unified dispatcher can also call this physical-E path when an explicit
`E_parameter` is supplied:

```python
from polyfempy.differentiable import prepare_optimization_problem

problem = prepare_optimization_problem(
    cfg=cfg,
    kind="material",
    body_id=1,
    E_parameter=E_lattice,
    parameter_name="E_lattice_MPa",
    bounds=(1.0, None),
    E_unit="MPa",
)
```

Calling `prepare_optimization_problem(cfg=cfg, kind="material", body_id=1)`
without `E_parameter` keeps the old `log_E` behavior for compatibility.

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
