# PolyFEM Python API Guide

This guide describes the public `polyfempy.api` layer in this repository. It is
not a full PolyFEM theory reference. It focuses on the practical questions a
Python user needs to answer:

- how to launch a simulation from Python
- what can be passed as `cfg`
- how JSON configuration and Python-side edits interact
- what fields are available on `Result`
- how to request `stress` and `von_mises`
- how file output differs from Python result extraction

For differentiable simulation, optimization, and PyTorch autograd workflows,
see `polyfempy.differentiable`.

## API Layers

Use `polyfempy.api` for forward solves, configuration objects, mesh IO, result
inspection, and non-differentiable post-processing:

```python
from polyfempy.api import SimulationConfig, solve

cfg = SimulationConfig.from_json_file("config.json")
result = solve(cfg=cfg)
```

Use `polyfempy.differentiable` when a solve must participate in a PyTorch loss
or optimization loop:

```python
from polyfempy.differentiable import (
    make_von_mises_loss,
    prepare_differentiable_simulation,
)

result = prepare_differentiable_simulation(cfg=cfg, derivative_type="shape")
loss = make_von_mises_loss(result=result, body=1, time="smooth_max")
loss.backward()
```

## `solve(...)`

The main forward entry point is:

```python
from polyfempy.api import solve

result = solve(cfg=...)
```

`cfg` can be provided in three common forms.

### JSON Path

```python
result = solve(cfg="config.json")
```

This is the closest form to the native PolyFEM configuration workflow and is
usually the safest option for complex scenes.

### Python `dict`

```python
cfg = {
    "pde": "LinearElasticity",
    "materials": [{"type": "LinearElasticity", "E": 2100.0, "nu": 0.3}],
    "geometry": [{"mesh": "mesh.msh"}],
    "boundary_conditions": {
        "dirichlet_boundary": [{"id": 4, "value": [0.0, 0.0]}],
    },
}

result = solve(cfg=cfg)
```

This is useful for small scripts that generate or modify configuration values
programmatically.

### `SimulationConfig`

```python
from polyfempy.api import (
    BoundaryConditions,
    Geometry,
    GeometryMesh,
    LinearSolver,
    Material,
    Output,
    SimulationConfig,
    Solver,
    Time,
    solve,
)

bc = BoundaryConditions()
bc.add_dirichlet(id=4, value=[0.0, 0.0])

cfg = SimulationConfig(
    pde="LinearElasticity",
    materials=Material(E=2100.0, nu=0.3),
    boundary_conditions=bc,
    geometry=Geometry(meshes=[GeometryMesh(mesh="mesh.msh")]),
    solver=Solver(linear=LinearSolver(solver_type="Eigen::SparseLU")),
    time=Time(tend=0.1, dt=0.01),
    output=Output(directory="out"),
)

result = solve(cfg=cfg)
```

This form is better for IDE completion and for reusable Python scene builders.

## JSON And Python Override Semantics

When a config is loaded from JSON and then edited in Python, the later Python
object state is the state used by `solve(cfg=cfg)`.

```python
from polyfempy.api import SimulationConfig, solve

cfg = SimulationConfig.from_json_file("config.json")
cfg.materials[0]["E"] = {"value": 50, "unit": "MPa"}

result = solve(cfg=cfg)
```

In this example the solve uses `E = 50 MPa`, even if the original JSON file
contained `E = 20 MPa`. This rule is intentional: JSON preserves the original
case, while the Python object represents the current effective case.

For round-trippable export, prefer the explicit full-JSON methods:

```python
payload = cfg.to_full_json_dict()
text = cfg.to_full_json_str()
cfg2 = SimulationConfig.from_full_json_dict(payload)
```

The older minimal JSON helpers remain for compatibility, but they do not
represent every solver/configuration field.

## Mesh Inputs

The normal path is to let the configuration load meshes from disk:

```python
result = solve(cfg="config.json")
```

For in-memory workflows, `solve(...)` also supports array-backed mesh data:

```python
result = solve(vertices=V, cells=C, cfg=cfg)
```

Array-backed mode is useful when meshes are generated in Python or by another
library before PolyFEM is called. If a mesh has no usable boundary or sideset
tags, a `sidesets_func` may be needed to define boundary ids geometrically.

## `Result`

`solve(...)` returns a `Result` object. Common fields are:

```python
result.u
result.vertices
result.cells
result.stress
result.strain
result.von_mises
result.history
result.meta
```

Inspect available fields before assuming a particular quantity is present:

```python
print(result.summary())
print(sorted(result.field_names()))
```

For arbitrary fields:

```python
field = result.get_field("von_mises")
```

`Result.to_torch(include_mesh=True)` converts stored arrays to torch tensors,
but it does not retroactively make a forward solve differentiable. Use
`polyfempy.differentiable` for autograd-backed solves.

## Stress And Von Mises Fields

The preferred approach is to request fields through the config/result-output
helpers before solving. The public examples use:

```python
from polyfempy.api.guided import results_section

results = results_section(requested_fields=["u", "stress", "von_mises"])
```

When native solver output already includes `stress`, `Result.von_mises` can be
computed directly from that field if a stored von Mises field is not present.

If native fields are unavailable, the API can use temporary VTU output and
`meshio` as a fallback. Treat fallback `stress` and `von_mises` as sampled
visualization fields. Their array length and mesh semantics may differ from
`result.u` and `result.vertices`.

Safe rule:

- use `u` and `vertices` as the primary solution-space fields
- use fallback `stress` and `von_mises` for visualization, statistics, and
  diagnostics
- do not assume fallback stress values are nodal fields aligned one-to-one with
  `result.vertices`

## Output Configuration

There are two output concerns:

1. solver-facing file output, such as Paraview/VTU files and logs
2. Python-facing result extraction, such as requested fields and fallback
   behavior

For examples and scripts, prefer the runtime helpers:

```python
from polyfempy.api.runtime import result_output, terminal_log

terminal_log(cfg, level="debug", file_level="debug", path="polyfem.log")
result_output(cfg, directory="runs/example", save_vtu=False)
```

This keeps output behavior explicit without forcing every script to edit nested
JSON dictionaries by hand.

## Recommended Minimal Workflow

```python
from polyfempy.api import SimulationConfig, solve
from polyfempy.api.runtime import result_output, terminal_log

cfg = SimulationConfig.from_json_file("config.json")

# Python-side edits override the original JSON values.
cfg.time.tend = 0.1
cfg.time.dt = 0.01

terminal_log(cfg, level="debug", file_level="debug", path="polyfem.log")
result_output(cfg, directory="runs/case_001", save_vtu=False)

result = solve(cfg=cfg)

print(result.summary())
print(result.von_mises)
```

## Current Limitations

- `solve(...)` still owns several responsibilities: config normalization,
  mesh loading, solver setup, solve execution, and result extraction. Treat it
  as the public convenience entry point, not as the internal architecture
  boundary.
- Fallback visualization fields may not share the same mesh as native solution
  fields.
- `force`, `reaction`, and contact-force access are not yet stable public
  result fields. They should receive explicit semantics before being exposed as
  first-class `Result` fields.

## Public Examples

The top-level `examples/` directory contains the recommended short tutorials:

```bash
python examples/01_forward_solve.py
python examples/02_result_fields.py
```

Differentiable examples require PyTorch:

```bash
python examples/03_shape_gradient.py
python examples/04_scalar_E_gradient.py
python examples/05_parameterized_vertex_map.py
```
