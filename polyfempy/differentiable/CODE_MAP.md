# Differentiable / Optimization Code Map

这份文档解释 `polyfempy.differentiable` 这一层每个主要文件在做什么，以及
shape、material、h/theta 三条 backward/optimization 链路是怎么接起来的。

核心观点：

```text
user design variable
  -> PyTorch map to a PolyFEM-supported tensor
  -> PolyFEM forward solve
  -> objective scalar loss
  -> PolyFEM adjoint derivative
  -> PyTorch chain rule back to the user design variable
```

PolyFEM 当前真正支持的可微输入主要是：

```text
vertices X
lambda/mu material tensors
initial velocity
```

`h/theta`、`E_lattice` 这类名字是用户实验语义。它们必须先通过 Python /
PyTorch map 转成 PolyFEM 支持的 tensor，才能进入 C++ adjoint 链路。

## Public Entry Points

用户脚本通常应该只看这一层：

```python
from polyfempy.differentiable import (
    prepare_differentiable_simulation,
    make_parameter,
    prepare_parameterized_shape_problem,
    prepare_optimization_problem,
    report_optimization_baseline,
    make_optimizer,
    make_von_mises_loss,
    make_material_von_mises_loss,
    run_optimization,
)
```

常见 one-off differentiable solve：

```text
prepare_differentiable_simulation(...)
  -> result
  -> make_von_mises_loss(result=result, ...)
  -> loss.backward()
  -> result.vertices.grad
```

常见 optimization flow：

```text
prepare_optimization_problem(...)
  -> report_optimization_baseline(...)
  -> make_optimizer(problem)
  -> make_*_loss(...)
  -> run_optimization(problem, ...)
```

## File Map

### `__init__.py`

Top-level export surface for `polyfempy.differentiable`.

It currently exports both user-facing helpers and many debugging helpers. This is convenient during
development, but it makes the public API look larger than it really should be.

Important user-facing exports:

```text
prepare_differentiable_simulation
make_parameter
prepare_parameterized_shape_problem
prepare_optimization_problem
report_optimization_baseline
make_optimizer
make_von_mises_loss
make_material_von_mises_loss
run_optimization
```

Likely advanced/debug exports:

```text
objective_solution_rhs_diagnostics
collect_material_chain_diagnostics
finite_difference_material_E_gradient
apply_finite_difference_gradient_fallback
```

### `design.py`

This file contains the new user-design adapter layer.

Main class:

```text
ParameterizedVertexDesign
```

Main user-level helpers:

```text
make_parameter(...)
make_named_parameter_map(...)
make_bounds_projector(...)
```

It maps user-owned PyTorch parameters to the fixed-topology vertex tensor that
PolyFEM can differentiate:

```text
parameters
  -> optional parameter_map(parameters)
  -> vertex_map(design_value, base_vertices, context)
  -> vertices
```

It also owns optional post-step projection:

```text
project(parameters)
```

The intended layering is:

```text
make_parameter / make_bounds_projector
  -> ParameterizedVertexDesign
  -> ParameterizedShapeOptimizationProblem
  -> prepare_parameterized_shape_problem(...)
```

This is the first generic version of the old h/theta-specific pattern.
`HThetaVertexMap` is now one experiment-specific `vertex_map`, not a special
library concept.

### `shape_problem.py`

Reusable PyTorch problem objects for direct and parameterized shape
optimization.

Main classes:

```text
ShapeOptimizationProblem
ParameterizedShapeOptimizationProblem
ShapeOptimizationStep
```

This file owns the per-iteration solve/step mechanics:

```text
optimizer.zero_grad()
result = problem.solve()
loss = loss_fn(result)
loss.backward()
optimizer.step()
optional design projection
yield ShapeOptimizationStep(...)
```

Direct shape solve path:

```text
vertices Parameter
  -> PolyFEMFunction.apply(...)
  -> DifferentiableResult(vertices=vertices)
```

Parameterized shape solve path:

```text
user parameters
  -> ParameterizedVertexDesign.vertices()
  -> PolyFEMFunction.apply(...)
  -> DifferentiableResult(vertices=mapped_vertices)
```

This module should stay about reusable problem state and optimization steps.
Config loading, baseline reporting, loss wrappers, and console formatting live
one layer above.

### `solve_diff.py`

This is the main differentiable solve adapter. It converts API configs and mesh inputs into a C++
`pf.Solver`, then wraps the solve in a PyTorch autograd function.

Main functions:

```text
prepare_differentiable_simulation(...)
solve_differentiable(...)
solve_differentiable_material(...)
solve_differentiable_material_from_youngs(...)
youngs_to_lame(...)
build_lame_from_youngs(...)
solver_body_slot_mask(...)
```

Shape solve path:

```text
cfg / V,C
  -> build/load solver mesh
  -> V_torch
  -> PolyFEMFunction.apply(solver, V_torch, derivative_type, solve_log_level)
  -> DifferentiableResult(u, vertices=V_torch)
```

Material solve path:

```text
E, nu
  -> lambda/mu tensors
  -> PolyFEMPerElementMaterialFunction.apply(solver, lam, mu, ...)
  -> DifferentiableMaterialResult(u, lam, mu)
```

Important boundary:

```text
derivative_type is a low-level backward selector, not a full optimization mode.
```

### `torch_integration.py`

This file contains the low-level PyTorch `Function` classes that directly call C++ PolyFEM.

`PolyFEMFunction`:

```text
input:  vertices
forward:
  solver.mesh().set_vertices(vertices)
  solver.solve()
backward:
  solver.solve_adjoint(dL/du)
  shape_derivative / elastic_material_derivative / initial_velocity_derivative
output grad:
  gradient for vertices input slot
```

This is used for shape-style differentiable solves.

`PolyFEMPerElementMaterialFunction`:

```text
input:  lambda, mu
forward:
  solver.set_per_element_material(lambda, mu)
  solver.solve()
backward:
  solver.solve_adjoint(dL/du)
  elastic_material_derivative(solver)
output grad:
  dL/dlambda, dL/dmu
```

This is used for material-style differentiable solves.

Important boundary:

```text
Do not put user concepts like h/theta or E_lattice into these low-level Function classes.
They should only know PolyFEM-supported tensor inputs.
```

### `result_diff.py`

Containers returned by differentiable solves.

`DifferentiableResult`:

```text
u
solver
vertices
shape_gradient -> vertices.grad
history
body_ids
```

Used by shape and parameterized-shape paths.

`DifferentiableMaterialResult`:

```text
u
solver
lam
mu
history
body_ids
```

Used by material paths.

The result object keeps the C++ solver alive until backward/reporting is done. Call
`release_solver()` after the graph is no longer needed.

### `objective_bridge.py`

This file turns PolyFEM C++ objectives into PyTorch scalar losses.

Main user-facing loss builders:

```text
make_von_mises_loss(...)
make_stress_norm_loss(...)
make_material_von_mises_loss(...)
```

Shape objective bridge:

```text
PolyFEMAutogradObjective
  forward:  obj.value(vertices)
  backward: obj.derivative(..., wrt="solution")
            obj.derivative(..., wrt="shape")
```

Material objective bridge:

```text
PolyFEMElasticAutogradObjective
  forward:  obj.value([lambda, mu])
  backward: obj.derivative(..., wrt="solution")
            obj.derivative(..., wrt="elastic")
```

Time aggregation is also handled here:

```text
last / first / max / mean / sum / smooth_max
```

Current API issue:

```text
Users need to choose make_von_mises_loss vs make_material_von_mises_loss.
Long term, make_von_mises_loss could dispatch from result type.
```

### `shape_optimization.py`

Preparation and reporting helpers for shape optimization.

It keeps the old import path for:

```text
ShapeOptimizationProblem
ParameterizedShapeOptimizationProblem
ShapeOptimizationStep
```

but the classes themselves now live in `shape_problem.py`.

This file owns:

```text
prepare_shape_optimization_problem(...)
prepare_parameterized_shape_problem(...)
prepare_parameterized_shape_optimization_problem(...)
make_von_mises_shape_loss(...)
format_shape_optimization_step(...)
report_optimization_baseline(...)
run_shape_optimization(...)
```

Boundary:

```text
shape_problem.py:
  reusable problem state and optimizer loop

shape_optimization.py:
  build/load shape problems and present them to user scripts
```

User-facing wrapper:

```text
prepare_parameterized_shape_problem(...)
  -> named parameters from make_parameter(...)
  -> default dict-style parameter_map
  -> default bounds projection
  -> prepare_parameterized_shape_optimization_problem(...)
```

Direct shape still means:

```text
design variable == vertices
PolyFEM-supported tensor == vertices
```

Parameterized shape still means:

```text
design variable == user parameters
PolyFEM-supported tensor == mapped vertices
```

### `material_optimization.py`

Reusable optimization helpers for scalar Young's modulus optimization.

Main class:

```text
ScalarMaterialOptimizationProblem
```

It owns:

```text
log_E: torch.nn.Parameter
slot_mask for selected body
nu
other material constants
solver
```

Its solve path:

```text
log_E
  -> softplus(log_E) + e_floor
  -> E
  -> unit conversion
  -> lambda/mu through youngs_to_lame
  -> solve_differentiable_material_from_youngs(...)
  -> DifferentiableMaterialResult
```

Its optimization step:

```text
optimizer.zero_grad()
result = problem.solve()
loss = loss_fn(result)
loss.backward()
gradient = current_E.grad
optimizer.step()
```

This is scalar material optimization:

```text
design variable == log_E
PolyFEM-supported tensor == lambda/mu
```

The first part of this file also contains diagnostics and finite-difference fallback utilities.
Those are useful for debugging but should be considered advanced/debug API.

### `optimization_problem.py`

Thin public dispatcher that makes shape and material optimization look similar from the user
script.

Main functions:

```text
prepare_optimization_problem(cfg=..., kind="shape"|"parameterized_shape"|"material", ...)
report_optimization_baseline(...)
make_optimizer(problem, ...)
run_optimization(problem, ...)
```

Dispatch rules:

```text
kind="shape" or "geometry"
  -> prepare_shape_optimization_problem(...)

kind="parameterized_shape" or "parametric_shape"
  -> prepare_parameterized_shape_optimization_problem(...)

kind="material", "e", or "youngs"
  -> prepare_material_optimization_problem(...)
```

Current API issue:

```text
run_optimization still contains type-specific defaults:
  shape: gradient_dir, max_vertex_step
  material: no gradient_dir, no max_vertex_step

Long term, these constraints should live on the design/problem object.
```

### `shape_mask.py`

Utility functions for selecting vertices by body id.

Main functions:

```text
body_vertex_mask(result, body_id=...)
shape_gradient_for_body(result, body_id=...)
```

Useful for:

```text
masking shape gradients to lattice-only vertices
checking fixed/outside vertices
building diagnostics/training samples
```

### `training_data.py`

Helper for saving differentiable run outputs into a training-sample directory.

It saves arrays such as:

```text
vertices.npy
cells.npy
body_ids.npy
shape_gradient.npy
metadata.json
```

This is data plumbing, not a core autograd primitive.

### `summary.py`

Small formatting helpers:

```text
gradient_norm(...)
print_loss_summary(...)
```

### `torch_bridge.py`

Older / experiment-facing bridge utilities for running differentiable PolyFEM steps and optimizer
probes.

It overlaps conceptually with the newer `solve_diff.py`, `shape_optimization.py`, and
`optimization_problem.py` stack. Treat it as compatibility / experiment support, not the cleanest
new API surface.

## Chain 1: Direct Shape Optimization

User script:

```python
problem = prepare_optimization_problem(cfg=cfg, kind="shape")
optimizer = make_optimizer(problem)
loss_fn = make_von_mises_loss(volume_selection=1, time_aggregation="smooth_max")
run_optimization(problem, steps=1, optimizer=optimizer, loss_fn=loss_fn)
```

Code flow:

```text
new_api_experiment02_von_mises_shape_optimization.py
  -> prepare_optimization_problem(kind="shape")
  -> prepare_shape_optimization_problem(...)
  -> prepare_differentiable_simulation(...)
  -> ShapeOptimizationProblem(vertices=Parameter(initial_vertices))
  -> run_optimization(...)
  -> run_shape_optimization(...)
  -> ShapeOptimizationProblem.optimize(...)
  -> ShapeOptimizationProblem.solve()
  -> PolyFEMFunction.apply(solver, vertices, "shape", ...)
  -> make_von_mises_loss(result)
  -> PolyFEMAutogradObjective.apply(...)
  -> loss.backward()
```

Backward details:

```text
PolyFEMAutogradObjective.backward:
  emits dJ/du and direct dJ/dX from the objective

PolyFEMFunction.backward:
  receives dJ/du
  calls solve_adjoint(dJ/du)
  calls shape_derivative(solver)
  returns dL/dX to PyTorch

PyTorch:
  accumulates gradient into problem.vertices.grad
```

Final optimized variable:

```text
vertices
```

## Chain 2: Scalar E Material Optimization

User script:

```python
problem = prepare_optimization_problem(cfg=cfg, kind="material", body_id=1)
optimizer = make_optimizer(problem)
loss_fn = make_material_von_mises_loss(volume_selection=1, time_aggregation="smooth_max")
run_optimization(problem, steps=1, optimizer=optimizer, loss_fn=loss_fn)
```

Code flow:

```text
new_api_experiment02_von_mises_E_optimization.py
  -> prepare_optimization_problem(kind="material", body_id=1)
  -> prepare_material_optimization_problem(...)
  -> ScalarMaterialOptimizationProblem(log_E=Parameter(...))
  -> run_optimization(...)
  -> run_scalar_material_optimization(...)
  -> ScalarMaterialOptimizationProblem.optimize(...)
  -> ScalarMaterialOptimizationProblem.solve()
  -> current_E = softplus(log_E) + e_floor
  -> youngs_value_to_internal(...)
  -> solve_differentiable_material_from_youngs(...)
  -> build_lame_from_youngs(...)
  -> PolyFEMPerElementMaterialFunction.apply(solver, lam, mu, ...)
  -> make_material_von_mises_loss(result)
  -> PolyFEMElasticAutogradObjective.apply(...)
  -> loss.backward()
```

Backward details:

```text
PolyFEMElasticAutogradObjective.backward:
  emits dJ/du and direct dJ/dlambda,dJ/dmu from objective wrt elastic params

PolyFEMPerElementMaterialFunction.backward:
  receives dJ/du
  calls solve_adjoint(dJ/du)
  calls elastic_material_derivative(solver)
  returns dL/dlambda,dL/dmu

PyTorch:
  chains dL/dlambda,dL/dmu through lambda/mu(E)
  chains dL/dE through E = softplus(log_E) + e_floor
  accumulates gradient into log_E.grad
```

Final optimized variable:

```text
log_E
```

Reported physical variable:

```text
current_E = softplus(log_E) + e_floor
```

## Chain 3: h/theta Parameterized Shape

Current location:

```text
experiment/experiment_api_solve/paper_experiments/paper_h_theta_autograd_opt.py
experiment/experiment_api_solve/new_api_experiment02_von_mises_theta_h_optimization.py
experiment/compute_canada_training_cases/h_theta_diff/compute_canada_von_mises_h_theta_diff.py
```

Current status:

```text
The paper h/theta script now uses the generic parameterized_shape API.
The h/theta geometry rule is still experiment-specific, but the optimization
problem is a generic ParameterizedShapeOptimizationProblem.
```

Conceptual flow:

```text
h_theta = [h, theta_deg]
  -> vertex_map(h_theta)
  -> vertices = X(h, theta_deg)
  -> PolyFEM shape solve
  -> objective loss
  -> loss.backward()
  -> h_theta.grad = [dL/dh, dL/dtheta_deg]
```

Code flow in the paper script:

```text
prepare_h_theta_optimization_problem(...)
  -> design = torch.nn.Parameter([h/H_SCALE, theta/THETA_SCALE])
  -> prepare_optimization_problem(kind="parameterized_shape", ...)
  -> ParameterizedVertexDesign(
         parameters=[design],
         parameter_map=_current_h_theta,
         vertex_map=_h_theta_vertex_map,
         project=_project_h_theta_design,
     )

run_h_theta_optimization(...)
  -> h_theta = problem.design.design_value()
  -> vertices = problem.design.vertex_map(h_theta, base_vertices, context)
  -> PolyFEMFunction.apply(solver, vertices, "shape", ...)
  -> make_von_mises_loss(result)
  -> loss.backward()
  -> h_theta.grad
  -> optimizer.step() updates design
```

Important point:

```text
PolyFEM only gives dL/dX.
PyTorch gives dX/dh and dX/dtheta through vertex_map.
The final h_theta gradient is not hand-written.
```

What became generic:

```text
HThetaVertexMap is just one user-defined geometry parameterization.
The library now lets users provide their own vertex_map or torch.nn.Module
through kind="parameterized_shape".
```

Generic API shape:

```python
h = make_parameter("h", 0.04, bounds=(0.03, 0.07))
theta_deg = make_parameter("theta_deg", 90.0, bounds=(60.0, 110.0))

def vertex_map(params, base_vertices, context):
    h_value = params["h"]
    theta_value = params["theta_deg"]
    # user-defined torch operations
    return vertices

problem = prepare_parameterized_shape_problem(
    cfg=cfg,
    parameters=[h, theta_deg],
    vertex_map=vertex_map,
)
```

Lower-level API shape:

```python
params = torch.nn.Parameter(torch.tensor([0.04, 90.0], dtype=torch.float64))

problem = prepare_optimization_problem(
    cfg=cfg,
    kind="parameterized_shape",
    parameters=[params],
    vertex_map=vertex_map,
)
```

Module/callable geometry path:

```python
class MyGeometry(torch.nn.Module):
    def forward(self):
        return vertices

problem = prepare_optimization_problem(
    cfg=cfg,
    kind="parameterized_shape",
    geometry=MyGeometry(...),
)
```

Rules for user-defined `vertex_map`:

```text
return shape must match the mesh vertices shape
mesh connectivity/topology must stay fixed
use torch operations for differentiable math
do not detach or convert differentiable values to numpy in the gradient path
do not remesh inside the differentiable forward
discrete changes like nx jumps are not represented by this gradient
```

## Current Pain Points

1. The public wrapper is cleaner than the internals.

```text
The user-facing API hides complexity, but the internal abstractions are not yet clean.
```

2. `derivative_type` leaks low-level implementation detail.

```text
It selects C++ backward routines.
It should not be treated as the user-facing optimization mode.
```

3. Shape and material have parallel but duplicated optimization loops.

```text
ShapeOptimizationProblem.optimize
ScalarMaterialOptimizationProblem.optimize
run_shape_optimization
run_scalar_material_optimization
```

4. Objective selection is still type-specific.

```text
make_von_mises_loss
make_material_von_mises_loss
```

A cleaner API could let `make_von_mises_loss` inspect the result type and route to shape or elastic
objective internally.

5. Diagnostics are exported alongside production API.

```text
finite-difference fallback and RHS probes should likely move to an advanced/debug namespace.
```

## Recommended Refactor Direction

Continue turning the optimization stack into a real design layer:

```text
Design
  -> parameters()
  -> forward PolyFEM-supported tensor
  -> post-step projection / constraints
  -> gradient/report extraction
```

Concrete designs:

```text
DirectVertexDesign:
  design variable = vertices
  map = identity

ParameterizedVertexDesign:
  design variable = user params
  map = user vertex_map(params) -> vertices
  status = implemented

ScalarYoungsModulusDesign:
  design variable = log_E
  map = log_E -> E -> lambda/mu
  status = still embedded in ScalarMaterialOptimizationProblem

MappedMaterialDesign:
  design variable = user params
  map = material_map(params) -> lambda/mu
  status = not implemented yet
```

Then the optimization driver can become generic:

```text
for each iteration:
  optimizer.zero_grad()
  result = problem.solve()
  loss = loss_fn(result)
  loss.backward()
  record diagnostics
  optimizer.step()
  design.project_if_needed()
```

The low-level PyTorch `Function` classes should stay simple:

```text
vertices -> solve -> dL/dvertices
lambda/mu -> solve -> dL/dlambda,dL/dmu
```

The generic part should be above them:

```text
user params -> PolyFEM-supported tensors
```

## Teacher-Facing Summary

The code currently works by stacking three layers:

```text
1. PyTorch user-design layer:
   h/theta, vertices, or log_E are ordinary torch Parameters.

2. PolyFEM-supported tensor layer:
   user parameters are mapped to vertices or lambda/mu.

3. C++ adjoint layer:
   PolyFEM solves the PDE/contact problem and returns derivatives with respect
   to vertices or lambda/mu.
```

Two cleanup steps are now in place:

```text
1. h/theta uses a generic ParameterizedVertexDesign.
   The experiment-specific part is only the vertex_map.

2. Shape problem objects moved into shape_problem.py.
   shape_optimization.py is now mostly preparation/reporting glue.
```

The next cleanup step is to make direct vertex shape and scalar E look like the
same kind of design object, so the optimization loop can stop knowing about
every special case.
