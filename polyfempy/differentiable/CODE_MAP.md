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
    prepare_scalar_youngs_material_problem,
    run_optimization,
    OptimizationRunResult,
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
  -> make_von_mises_loss(...)
  -> run_optimization(problem, ...)
```

诊断、finite-difference 检查和旧 torch bridge probe 统一放在 advanced 命名空间：

```python
from polyfempy.differentiable import advanced
```

## File Map

### `__init__.py`

Top-level export surface for `polyfempy.differentiable`.

The recommended API is tracked by `PUBLIC_API`. Older debugging and compatibility exports are
tracked separately by `ADVANCED_COMPAT_API` and are also available from
`polyfempy.differentiable.advanced`.

Important user-facing exports:

```text
prepare_differentiable_simulation
make_parameter
prepare_parameterized_shape_problem
prepare_optimization_problem
report_optimization_baseline
make_optimizer
make_von_mises_loss
prepare_scalar_youngs_material_problem
run_optimization
```

Advanced/debug helpers are still re-exported for compatibility, but new code should import them
through `polyfempy.differentiable.advanced`:

```text
objective_solution_rhs_diagnostics
collect_material_chain_diagnostics
finite_difference_material_E_gradient
apply_finite_difference_gradient_fallback
run_polyfem_bridge_step
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

It validates the user map before PolyFEM is called:

```text
return value must be a torch.Tensor
shape must match base_vertices
dtype/device must match base_vertices
values must be finite
returned vertices must stay connected to differentiable parameters when grad is enabled
```

These checks belong here instead of in experiment scripts so every
parameterized-shape workflow gets the same contract and error messages.

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

Current API behavior:

```text
make_von_mises_loss(...) accepts both shape results and material results.
Material results are detected from their lambda/mu tensors and routed to the elastic objective bridge.
make_material_von_mises_loss(...) remains available for explicit advanced/compatibility use.
body=... and time=... are user-facing aliases for volume_selection=... and time_aggregation=....
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

Advanced lower-level builder:

```text
prepare_parameterized_shape_optimization_problem(...)
  -> accepts prebuilt ParameterizedVertexDesign or custom parameter_map/project
  -> loads the fixed-topology mesh
  -> constructs ParameterizedShapeOptimizationProblem
```

New user examples should call `prepare_parameterized_shape_problem(...)` unless
they specifically need that lower-level control.

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

### `material_config.py`

Small parsing helpers for reading scalar material data from `cfg.materials`.

It owns:

```text
material id lookup
Young's modulus value/unit parsing
Poisson ratio parsing
fixed non-design material inference
```

This keeps `material_optimization.py` from carrying low-level config parsing
logic directly.

### `material_optimization.py`

Reusable optimization helpers for scalar Young's modulus optimization.

Main class:

```text
ScalarMaterialOptimizationProblem
```

It owns:

```text
log_E: torch.nn.Parameter for the compatibility softplus path
E_parameter: torch.nn.Parameter for the newer physical-E path
slot_mask for selected body
nu
other material constants
solver
```

Compatibility solve path:

```text
log_E
  -> softplus(log_E) + e_floor
  -> E
  -> unit conversion
  -> lambda/mu through youngs_to_lame
  -> solve_differentiable_material_from_youngs(...)
  -> DifferentiableMaterialResult
```

User-facing physical-E solve path:

```text
E_parameter
  -> optional bounds projection after optimizer step
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
optional physical-E bounds projection
```

This is scalar material optimization:

```text
design variable == physical E_parameter, or compatibility log_E
PolyFEM-supported tensor == lambda/mu
```

Diagnostics and finite-difference fallback utilities were moved out of this
file. This module should stay focused on scalar material problem setup,
per-iteration solves, optimizer creation, and reports.

### `material_diagnostics.py`

Advanced/debug helpers for scalar material work:

```text
objective_solution_rhs_diagnostics(...)
collect_material_chain_diagnostics(...)
finite_difference_material_E_gradient(...)
apply_finite_difference_gradient_fallback(...)
```

These remain available from `polyfempy.differentiable` for compatibility, but
new user-facing examples should not need them.

### `optimization_problem.py`

Thin public dispatcher for preparing optimization problems.

Main functions:

```text
prepare_optimization_problem(cfg=..., kind="shape"|"parameterized_shape"|"material", ...)
```

Dispatch rules:

```text
kind="shape" or "geometry"
  -> prepare_shape_optimization_problem(...)

kind="parameterized_shape" or "parametric_shape"
  -> prepare_parameterized_shape_problem(...)
  -> prepare_parameterized_shape_optimization_problem(...)

kind="material", "e", or "youngs"
  -> prepare_material_optimization_problem(...)

kind="material", "e", or "youngs" with explicit E_parameter
  -> prepare_scalar_youngs_material_problem(...)
```

This file should not own optimizer-loop details. Those live in
`optimization_runner.py`.

### `optimization_runner.py`

Runtime helpers for already-prepared optimization problems:

```text
prepare_optimization_baseline_simulation(...)
report_optimization_baseline(...)
make_optimizer(problem, ...)
run_optimization(problem, ...)
OptimizationRunResult
```

It still has type-specific defaults for output files and shape-only gradient
artifacts, but that logic is separated from the problem-construction
dispatcher.

Compatibility behavior:

```text
run_optimization(...) returns list[Step] by default
run_optimization(..., return_result=True) returns OptimizationRunResult
```

Stable result fields:

```text
run.steps
run.final_step
run.final_loss
run.summary()
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

Legacy / experiment-facing bridge utilities for running differentiable PolyFEM steps and optimizer
probes.

It overlaps conceptually with the newer `solve_diff.py`, `shape_optimization.py`, and
`optimization_problem.py` stack. Treat it as compatibility / experiment support, not the cleanest
new API surface. Prefer importing it through `polyfempy.differentiable.advanced` when a low-level
probe is actually needed.

## Chain 1: Direct Shape Optimization

User script:

```python
problem = prepare_optimization_problem(cfg=cfg, kind="shape")
optimizer = make_optimizer(problem)
loss_fn = make_von_mises_loss(body=1, time="smooth_max")
run = run_optimization(
    problem,
    steps=1,
    optimizer=optimizer,
    loss_fn=loss_fn,
    return_result=True,
)
print(run.final_loss)
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
E_lattice = make_parameter("E_lattice_MPa", 20.0, bounds=(1.0, None))
problem = prepare_scalar_youngs_material_problem(
    cfg=cfg,
    body_id=1,
    E_parameter=E_lattice,
    E_unit="MPa",
)
optimizer = make_optimizer(problem)
loss_fn = make_von_mises_loss(body=1, time="smooth_max")
run = run_optimization(
    problem,
    steps=1,
    optimizer=optimizer,
    loss_fn=loss_fn,
    return_result=True,
)
print(run.final_loss)
```

Code flow:

```text
new_api_experiment02_von_mises_E_optimization.py
  -> E_lattice = make_parameter(...)
  -> prepare_scalar_youngs_material_problem(...)
  -> prepare_material_optimization_problem(...)
  -> ScalarMaterialOptimizationProblem(E_parameter=E_lattice)
  -> run_optimization(...)
  -> run_scalar_material_optimization(...)
  -> ScalarMaterialOptimizationProblem.optimize(...)
  -> ScalarMaterialOptimizationProblem.solve()
  -> current_E = E_lattice
  -> youngs_value_to_internal(...)
  -> solve_differentiable_material_from_youngs(...)
  -> build_lame_from_youngs(...)
  -> PolyFEMPerElementMaterialFunction.apply(solver, lam, mu, ...)
  -> make_von_mises_loss(result) dispatches to the elastic objective bridge
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
  accumulates gradient into E_lattice.grad
```

Final optimized variable:

```text
E_lattice
```

Compatibility path:

```text
prepare_material_optimization_problem(...) still optimizes log_E:
  log_E -> softplus(log_E) + e_floor -> E -> lambda/mu
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
  -> h = make_parameter("h", INITIAL_H, bounds=(H_MIN, H_MAX))
  -> theta_deg = make_parameter("theta_deg", INITIAL_THETA_DEG, bounds=(...))
  -> prepare_parameterized_shape_problem(...)
  -> ParameterizedVertexDesign(
         parameters=[h, theta_deg],
         parameter_map=_h_theta_design_value,
         vertex_map=_h_theta_vertex_map,
     )

run_h_theta_optimization(...)
  -> h_theta = problem.design.design_value()
  -> vertices = problem.design.vertex_map(h_theta, base_vertices, context)
  -> PolyFEMFunction.apply(solver, vertices, "shape", ...)
  -> make_von_mises_loss(result)
  -> loss.backward()
  -> h_theta.grad
  -> optimizer.step() updates h and theta_deg
```

The paper optimizer uses per-parameter learning rates derived from
`H_SCALE` and `THETA_SCALE_DEG` so the physical update sizes match the previous
scaled-design implementation.

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
params is a dict keyed by make_parameter(...) names unless an advanced parameter_map is provided
base_vertices is the fixed-topology mesh loaded from cfg
context is for cached masks, index sets, or non-differentiable metadata
return shape must match the mesh vertices shape
mesh connectivity/topology must stay fixed
use torch operations for differentiable math
do not detach or convert differentiable values to numpy in the gradient path
do not remesh inside the differentiable forward
discrete changes like nx jumps are not represented by this gradient
```

The parameter names are not special. `h`, `theta_deg`, `lattice_y_scale`,
`block_shift_x`, or neural-network weights are all valid user parameters if
their PyTorch map returns fixed-topology vertices.

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

4. Objective selection is mostly unified, but the older explicit material helper remains.

```text
make_von_mises_loss
make_material_von_mises_loss
```

`make_von_mises_loss` now inspects the result type and routes material results to the elastic
objective bridge internally. `make_material_von_mises_loss` is kept for compatibility and explicit
advanced use.

5. Diagnostics are still exported alongside production API.

```text
finite-difference fallback and RHS probes now live in material_diagnostics.py,
but __init__.py still re-exports them for compatibility. New code should import those helpers
through polyfempy.differentiable.advanced.
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
  design variable = physical E_parameter or compatibility log_E
  map = E -> lambda/mu
  status = implemented for scalar E inside ScalarMaterialOptimizationProblem

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
   h/theta, vertices, physical E, or compatibility log_E are ordinary torch Parameters.

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

3. Scalar material optimization can now use a physical E_parameter through
   prepare_scalar_youngs_material_problem(...), while the older log_E path is
   still available for compatibility.
```

The next cleanup step is to make direct vertex shape and scalar E look like the
same kind of design object, so the optimization loop can stop knowing about
every special case.
