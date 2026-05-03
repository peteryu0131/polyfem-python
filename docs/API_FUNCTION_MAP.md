# API Function Map

This note documents what each paper-facing helper contains, what it calls, and
which pieces are shared reusable API versus demo-specific code. It is meant to
make the API surface inspectable without reading every implementation file.

## Layers

| Layer | Location | Shared? | Purpose |
| --- | --- | --- | --- |
| Guided config API | `polyfempy/api/guided_sections.py` | Yes | Converts Python sections into a `SimulationConfig`. |
| Forward solve API | `polyfempy/api/solve.py` and `polyfempy/api/_solve_pipeline.py` | Yes | Runs PolyFEM and returns structured result fields. |
| Differentiable API | `polyfempy/differentiable/` | Yes | Connects PolyFEM solves, PyTorch autograd, losses, and optimization loops. |
| Paper demo helpers | `experiment/paper_experiment/common.py` | No | Paths, constants, workspaces, output setup, and optional reporting. |
| Paper geometry map | `experiment/paper_experiment/08_h_theta_shape_optimization.py` | Demo-specific | The explicit `h/theta -> vertices` parameterization for the lattice example. |

## Forward API

### `build_config(template, workspace)`

Location: `polyfempy/api/guided_sections.py`

What it does:

1. Creates an empty `SimulationConfig`.
2. Copies high-level problem fields from `ExperimentTemplate`.
3. Adds all bodies with `add_body_from_section(...)`.
4. If bodies are array-backed, stores the mesh payload in
   `cfg.extras["_mesh_array_mode"]`.
5. Builds solver, time, space, output, and contact sections.
6. Returns a `SimulationConfig`.

What it does not do:

- It does not run PolyFEM.
- It does not build a loss.
- It does not do differentiation.
- It does not update design variables.

Shared status: shared public config helper.

### `solve(vertices=None, cells=None, cfg=None, ...)`

Location: `polyfempy/api/solve.py`

What it does:

1. Accepts either explicit `vertices/cells` or a config with geometry.
2. Calls the staged pipeline in `polyfempy/api/_solve_pipeline.py`.
3. Returns a `Result` object with fields such as `u`, `von_mises`, and mesh
   metadata when requested.

Internal pipeline:

```text
normalize_cfg
  -> build_full_json
  -> resolve_runtime_options
  -> normalize_mesh_inputs
  -> build_solver
  -> configure_solver
  -> apply_sidesets
  -> run_solver_stage
  -> extract_native_outputs
  -> apply_sampled_vtu_fallback
  -> finalize_result
```

Shared status: shared public forward-solve entry point. The heavy numerical
solve is in the PolyFEM backend; the Python function is the orchestration layer.

## Differentiable Problem Builders

### `prepare_optimization_problem(cfg, kind=..., **kwargs)`

Location: `polyfempy/differentiable/optimization_problem.py`

What it does:

1. Normalizes the user-facing `kind`.
2. Dispatches to one of three problem builders:
   - `kind="shape"` -> `prepare_shape_optimization_problem(...)`
   - `kind="parameterized_shape"` -> `prepare_parameterized_shape_problem(...)`
   - `kind="material"` -> scalar Young's modulus material problem
3. Returns a reusable problem object with `.solve()` and `.make_optimizer()`.

What it hides:

- Only the dispatch decision. It does not contain the optimization loop or the
  geometry parameterization.

Shared status: shared public convenience dispatcher.

### `prepare_parameterized_shape_problem(...)`

Location: `polyfempy/differentiable/shape_optimization.py`

What it does:

1. Receives plain `torch.nn.Parameter` objects from the user.
2. Normalizes parameter names and bounds with `normalize_design_parameters(...)`.
3. Builds a dictionary-style parameter map with
   `make_named_parameter_map(...)`, e.g. `{"h": h, "theta_deg": theta_deg}`.
4. Builds a bounds projector with `make_bounds_projector(...)`.
5. Delegates to `prepare_parameterized_shape_optimization_problem(...)`.
6. Returns a `ParameterizedShapeOptimizationProblem`.

What it does not do:

- It does not define how `h/theta` changes the lattice.
- It does not remesh.
- It does not call `optimizer.step()`.

Shared status: shared public parameterized-shape helper.

### `prepare_parameterized_shape_optimization_problem(...)`

Location: `polyfempy/differentiable/shape_optimization.py`

What it does:

1. Loads the fixed-topology mesh from the config.
2. Extracts base vertices, cells, body ids, boundary ids, and solver settings.
3. Creates a `ParameterizedVertexDesign` if the caller did not provide one.
4. Returns a `ParameterizedShapeOptimizationProblem`.

What it guarantees:

- The reference mesh connectivity is fixed.
- Later vertex maps must return vertices with the same shape as the base mesh.

Shared status: lower-level shared builder used by the public wrapper.

## Parameterized Shape Objects

### `ParameterizedVertexDesign`

Location: `polyfempy/differentiable/design.py`

What it stores:

- The PyTorch parameters.
- The user `vertex_map`.
- The fixed `base_vertices`.
- Optional context for precomputed non-differentiable masks.
- Optional projection logic for bounds.

What `vertices()` does:

1. Gets the current PyTorch parameters.
2. Converts them to the user-facing design value, usually a dict.
3. Calls the user `vertex_map(...)`.
4. Validates that the returned vertices are a Torch tensor.
5. Validates same shape, dtype, device, finite values, and gradient connection.
6. Returns the new vertex tensor.

Shared status: shared adapter between low-dimensional parameters and PolyFEM
shape differentiation.

### `ParameterizedShapeOptimizationProblem.solve()`

Location: `polyfempy/differentiable/shape_problem.py`

What it does:

1. Calls `self.design.vertices()` to get the current mesh vertices.
2. Checks that the vertex count and shape match the reference mesh.
3. Runs the differentiable PolyFEM solve through either:
   - the cached solver plus `PolyFEMFunction.apply(...)`, or
   - `prepare_differentiable_simulation(...)`.
4. Returns a `DifferentiableResult`.

Why this matters:

- PolyFEM gives gradients with respect to vertices.
- PyTorch then continues the chain through `vertex_map(...)` back to `h` and
  `theta_deg`.

Shared status: shared problem object used by parameterized shape optimization.

## Loss And Optimization

### `make_von_mises_loss(result=None, ...)`

Location: `polyfempy/differentiable/objective_bridge.py`

What it does:

1. Resolves body selection and time aggregation.
2. If `result is None`, returns a reusable `loss_fn(result)` builder.
3. If the result is a material-differentiation result, routes to the material
   objective bridge.
4. Otherwise calls `make_polyfem_autograd_loss(...)`.
5. Returns a Torch scalar loss, or `(loss, info)` when `return_info=True`.

Common paper setting:

```text
body=1
time_aggregation="smooth_max"
```

Shared status: shared objective helper.

### `make_optimizer(problem, **kwargs)`

Location: `polyfempy/differentiable/optimization_runner.py`

What it does:

1. Checks the problem type.
2. Calls the problem-specific `.make_optimizer(...)`.
3. Returns a standard PyTorch optimizer such as `torch.optim.Adam`.

Shared status: shared convenience helper. The optimizer itself is PyTorch.

### `run_optimization(problem, steps, optimizer, loss_fn, ...)`

Location: `polyfempy/differentiable/optimization_runner.py`

What it does:

1. Checks whether the problem is shape, parameterized shape, or material.
2. Sets optional default summary/output paths when a workspace is provided.
3. Dispatches to the type-specific loop.
4. Returns either a list of step records or an `OptimizationRunResult`.

For parameterized shape, the loop is:

```text
optimizer.zero_grad()
result = problem.solve()
loss = loss_fn(result)
loss.backward()
optimizer.step()
problem.design.project_()
record parameter values and vertex update size
result.release_solver()
```

Shared status: shared optimization-loop helper.

### `shape_gradient_for_body(result, body_id=...)`

Location: `polyfempy/differentiable/shape_mask.py`

What it does:

1. Reads `result.shape_gradient`.
2. Builds a vertex mask for the selected body.
3. Returns the same gradient with non-selected vertices zeroed.

Shared status: shared diagnostic/helper function for shape gradients.

## Paper Demo Functions

### `experiment/paper_experiment/common.py`

This file is intentionally not the core API. It contains:

- `new_workspace(...)`: create an output folder.
- `configure_output(...)`: attach log/output paths to a config.
- `write_summary(...)`: JSON reporting for longer experiment scripts.
- `scalar_from_snapshot(...)`: read scalar values from optimization snapshots.
- `design_step_label(...)`: name before/after report folders.

These functions are shared only inside the paper demos. They do not solve,
differentiate, build losses, or optimize.

### `h_theta_vertex_map(params, base_vertices, context)`

Location: `experiment/paper_experiment/08_h_theta_shape_optimization.py`

What it does:

1. Reads `h` and `theta_deg` from the parameter dict.
2. Uses precomputed static mesh metadata from `context`.
3. Moves vertices with Torch operations.
4. Returns a vertex tensor with the same shape as `base_vertices`.

What it does not do:

- It does not change triangle connectivity.
- It does not call PolyFEM.
- It does not compute the loss.
- It does not update `h` or `theta_deg`.

Shared status: demo-specific geometry formula, intentionally kept in the demo
file so the parameterization is visible.

## Clean H/Theta Demo Chain

The clean paper demo `experiment/paper_experiment/08_h_theta_shape_optimization.py`
uses the shared API like this:

```text
build_config(...)
  -> h = torch.nn.Parameter(...)
  -> theta_deg = torch.nn.Parameter(...)
  -> prepare_parameterized_shape_problem(
         parameters=[h, theta_deg],
         parameter_names=["h", "theta_deg"],
         bounds={...},
         vertex_map=h_theta_vertex_map,
     )
  -> make_optimizer(...)
  -> make_von_mises_loss(...)
  -> run_optimization(...)
```

The differentiable chain is:

```text
h, theta_deg
  -> h_theta_vertex_map(...)
  -> vertices
  -> PolyFEM differentiable solve
  -> smooth-max von Mises loss
  -> loss.backward()
  -> h.grad, theta_deg.grad
  -> optimizer.step()
```

This is the part to show when the question is whether the example hides a large
amount of custom optimization code. The custom paper logic is only the explicit
geometry map; the solve, loss bridge, and optimization loop are shared reusable
API functions.
