# Differentiable / Optimization Code Map

This document describes the main responsibility boundaries inside
`polyfempy.differentiable`. The central design is:

```text
user design variable
  -> PyTorch map to a PolyFEM-supported tensor
  -> PolyFEM forward solve
  -> scalar objective
  -> PolyFEM adjoint derivative
  -> PyTorch chain rule back to the user design variable
```

PolyFEM currently differentiates most directly with respect to:

- mesh vertices `X`
- material tensors derived from Lame parameters
- selected initial-condition quantities

User-facing variables such as `E_lattice`, `h`, and `theta_deg` are not native
C++ differentiable variables by themselves. They must first be mapped to one of
the supported tensors.

## Public Entry Points

Recommended imports for user scripts:

```python
from polyfempy.differentiable import (
    make_optimizer,
    make_von_mises_loss,
    prepare_differentiable_simulation,
    prepare_optimization_problem,
    prepare_parameterized_shape_problem,
    run_optimization,
    save_training_sample,
    shape_gradient_for_body,
)
```

One-off differentiable shape solve:

```text
prepare_differentiable_simulation(..., derivative_type="shape")
  -> DifferentiableResult
  -> make_von_mises_loss(result=result, ...)
  -> loss.backward()
  -> shape_gradient_for_body(result, body_id=...)
```

Generic optimization flow:

```text
prepare_optimization_problem(...)
  -> make_optimizer(problem)
  -> make_von_mises_loss(...)
  -> run_optimization(problem, ...)
```

`report_optimization_baseline(...)` is available when an experiment needs a
baseline report, but it is not part of the minimal optimization path.

Advanced diagnostics and legacy bridge probes should be imported from:

```python
from polyfempy.differentiable import advanced
```

## File Responsibilities

### `__init__.py`

Defines the public export surface. `PUBLIC_API` is the recommended user-facing
set. `ADVANCED_COMPAT_API` keeps older diagnostics available without making
them the preferred entry point.

### `solve_diff.py`

Converts API configs and mesh inputs into a C++ `Solver`, then calls the torch
autograd bridge. It owns:

- config normalization
- disk-mesh and array-mesh setup
- solver construction
- body-slot helpers for material masks
- high-level differentiable solve entry points

Key exported functions:

```text
prepare_differentiable_simulation
solve_differentiable_material_from_youngs
youngs_value_to_internal
youngs_to_lame
```

`solve_differentiable(...)` is still exported as the lower-level compatibility
entry point, but new user scripts should prefer
`prepare_differentiable_simulation(...)`.

### `torch_integration.py`

Contains `PolyFEMFunction`, the low-level `torch.autograd.Function` wrapper.
Its forward pass calls the C++ solve, stores the solver/cache context, and
returns solution tensors. Its backward pass calls the C++ adjoint path and
returns gradients in the layout expected by PyTorch.

This file should stay close to the C++ binding contract. User-level objective
logic belongs in `objective_bridge.py` or higher layers.

### `result_diff.py`

Defines differentiable result containers. These are the differentiable analogs
of API `Result` objects and carry tensors, solver ownership, body ids, metadata,
and release hooks.

### `objective_bridge.py`

Builds scalar objectives from differentiable results. It owns common stress
objectives such as smooth-max von Mises and stress-norm losses, plus the
aggregation semantics over time.

Use this layer instead of duplicating objective extraction in experiment scripts.

### `shape_mask.py`

Provides helpers for extracting body-specific masks and gradients, such as
`body_vertex_mask(...)` and `shape_gradient_for_body(...)`.

### `design.py`

Defines the user-design adapter layer for named parameters and vertex maps.
The most important class is:

```text
ParameterizedVertexDesign
```

It maps user parameters to vertices:

```text
parameters
  -> optional parameter_map(parameters)
  -> vertex_map(params, base_vertices)
  -> mapped vertices
```

It validates that the returned vertices are tensors, preserve shape/device, are
finite, and remain connected to the differentiable parameters.

### `geometry_maps.py`

Small reusable building blocks for vertex maps:

```text
relative_scale
tan_half_angle_scale
vertices_axis_le
scale_selected_vertices_about_axis_center
```

These are intentionally simple. Complex geometry semantics should stay in the
calling experiment or application layer.

### `shape_problem.py` and `shape_optimization.py`

Reusable problem objects and loops for shape optimization. They own the repeated
pattern:

```text
optimizer.zero_grad()
result = problem.solve()
loss = loss_fn(result)
loss.backward()
optimizer.step()
optional projection
record step summary
```

Direct shape optimization differentiates vertices directly. Parameterized shape
optimization differentiates named user parameters through a vertex map.

### `material_config.py` and `material_optimization.py`

Material helpers for scalar Young's modulus optimization. A scalar `E` is mapped
through the material chain:

```text
log_E or E parameter
  -> positive Young's modulus
  -> lambda/mu material tensors
  -> PolyFEM material derivative
  -> chain rule back to scalar E
```

Use `prepare_optimization_problem(..., kind="material", E_parameter=...)` for
the public one-scalar material workflow. The lower-level
`prepare_scalar_youngs_material_problem(...)` helper remains available for
compatibility.

### `optimization_problem.py` and `optimization_runner.py`

Higher-level dispatch and shared optimization result objects. The unified entry
point is:

```python
prepare_optimization_problem(cfg=cfg, kind="shape" | "material", ...)
```

The dispatcher should keep public scripts thin while preserving different
implementation paths for shape and material variables.

### `material_diagnostics.py`

Finite-difference and diagnostic helpers for validating material-gradient
chains. These are useful for debugging but are not the normal user path.

### `training_data.py`

Exports small supervised samples from differentiable runs. The public helper is:

```python
save_training_sample(...)
```

It should only package successful runs and should keep metadata explicit enough
for downstream ML code to interpret vertices, cells, body ids, and gradients.

## Design-Variable Semantics

Do not conflate objective, design variable, and gradient target:

| Workflow | Objective | Design variable | Gradient |
| --- | --- | --- | --- |
| Shape gradient | smooth-max von Mises | vertices `X` | `dL/dX` |
| Scalar material | smooth-max von Mises | lattice `E` | `dL/dE` |
| Parameterized shape | smooth-max von Mises | `h`, `theta_deg` | `dL/dh`, `dL/dtheta_deg` |

`derivative_type` selects the backward path used by the low-level solve. It is
not, by itself, a full optimization-mode switch. Public helpers prepare the
correct tensors and chain-rule bridge for each workflow.

## Current Guidance

- Keep public examples short and use the helpers above.
- Put one-off probes and finite-difference checks under `advanced` or
  experiment-specific folders.
- Keep material-specific scalar `E` logic separate from generic shape logic;
  the user-facing scripts can look similar, but the differentiable tensor chain
  is different.
- Prefer explicit body ids and body masks when reporting gradients.
