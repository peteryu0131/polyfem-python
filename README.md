# PolyFEM-Python

PolyFEM-Python provides a Python and PyTorch interface to high-fidelity finite
element simulation with nonlinear elasticity and contact. It exposes PolyFEM
simulations through programmatic configuration objects, structured result
fields, and differentiable operators for scientific machine learning workflows.

Many ML-oriented simulators are convenient from Python but limited in
contact-rich deformable mechanics. PolyFEM and IPC-style solvers provide strong
mechanics, but their native interfaces are C++-centric and hard to integrate
into ML pipelines. This repository bridges that gap with:

- a stable Python simulation entry point: `solve(cfg=...)`
- typed configuration objects and guided config builders
- structured outputs such as `Result.u`, `Result.vertices`, and
  `Result.von_mises`
- PyTorch differentiable simulation and optimization helpers
- public examples, tests, and paper reproduction scripts

## Quickstart

Run a contact-rich deformable simulation from a JSON config:

```python
from polyfempy.api import SimulationConfig, solve

cfg = SimulationConfig.from_json_file("examples/configs/contact_impact.json")
result = solve(cfg=cfg)

print(result.u.shape)
print(result.vertices.shape)
print(result.available_fields())
```

Run the same idea as a script:

```bash
python examples/01_forward_solve.py
```

Inspect fields and export a VTU file:

```python
u = result.require_field("u", namespace="point_data")
von_mises = result.require_field("von_mises")
print(result.field_info("von_mises"))
result.write("result_fields.vtu")
```

or run:

```bash
python examples/02_result_fields.py
```

Run differentiable shape-gradient and vertex-map examples:

```bash
python examples/03_shape_gradient.py
python examples/05_parameterized_vertex_map.py
```

Generated outputs go under `examples/runs/`, which is ignored by git.

## Public API

The recommended user-facing surface is intentionally small. Lower-level config
classes and reporting helpers remain available, but they are treated as
advanced or compatibility APIs rather than the first path for new users.

Forward simulation:

```python
from polyfempy.api import solve, SimulationConfig, Result
```

Guided config construction:

```python
import polyfempy.api.guided as g

template = g.simulation_template(
    bodies=g.bodies_section(
        g.body_section(
            name="body",
            mesh="mesh.msh",
            material=g.material_section(model="NeoHookean", E=20.0, nu=0.45),
        )
    ),
)
cfg = g.build_config(template, workspace)
```

Differentiable simulation:

```python
from polyfempy.differentiable import (
    prepare_differentiable_simulation,
    make_von_mises_loss,
    shape_gradient_for_body,
)
```

`solve_differentiable(...)` remains available as a lower-level compatibility
entry point, but new scripts should usually start with
`prepare_differentiable_simulation(...)`.

Optimization helpers:

```python
from polyfempy.differentiable import (
    prepare_optimization_problem,
    prepare_parameterized_shape_problem,
    make_von_mises_loss,
    run_optimization,
)
```

Recommended optimization runs request a stable result object:

```python
run = run_optimization(..., return_result=True)
print(run.final_loss, run.best_loss)
print(run.summary())
```

### Differentiable Stability

The recommended differentiable paths for new users are:

- shape gradients through `prepare_differentiable_simulation(..., derivative_type="shape")`
- scalar Young's modulus optimization through `prepare_optimization_problem(..., kind="material")`
- parameterized fixed-topology shape optimization through `prepare_parameterized_shape_problem(...)`

Advanced or experimental paths include initial-condition/initial-velocity
gradients, raw `derivative_type="material"` use, and finite-difference
diagnostics. Prefer the public helper functions above unless you are extending
the differentiable backend itself.

Reporting and runtime helpers are available for scripts that need logs or
history artifacts, but they are optional helpers rather than the core API path.
For the current API surface decision and import audit, see:

- `docs/API_PUBLIC_SURFACE_DECISION.md`
- `docs/API_INTERNAL_IMPORT_AUDIT.md`

## Examples

The top-level `examples/` directory contains short user-facing tutorials:

- core: `examples/01_forward_solve.py`, `examples/02_result_fields.py`
- advanced: `examples/03_shape_gradient.py` through `examples/06_dataset_one_case.py`
- paper reproduction: `experiment/paper_experiment/`

These examples use checked-in meshes under `examples/assets/impact/`, so they
do not require Gmsh or a cluster environment.

For the examples-to-capability matrix, see:

- `docs/EXAMPLES_MATRIX.md`

## Tests

Run the Python test suite:

```bash
python -m pytest tests
```

The GitHub Actions workflow runs the same command. The tests include public
import smoke tests and config/result/pipeline checks. Some tests avoid requiring
the compiled C++ backend so the Python API boundary remains testable in
lightweight environments. A backend smoke test is also included: it is skipped
when the compiled backend is unavailable and runs a one-step `solve(cfg=...)`
when the backend is present.

## Paper Artifact

The teacher-facing Experiment 02 API demos live in:

- `experiment/paper_experiment/README.md`

The shortest h/theta shape-optimization demo is:

- `experiment/paper_experiment/08_h_theta_shape_optimization.py`

For reviewer-facing artifact instructions, see:

- `ARTIFACT.md`
- `docs/REVIEWER_QUICKSTART.md`
- `docs/TOMS_REVIEW_CHECKLIST.md`
- `docs/ARTIFACT_REPRODUCIBILITY.md`
- `docs/TEST_MATRIX.md`
- `docs/SOLVE_CONTRACT.md`
- `docs/DIFFERENTIABLE_CONTRACT.md`

For current paper optimization and Compute Canada reproduction scripts, see:

- `experiment/paper_experiment/README.md`

For an advisor-oriented guide to the clean API demos and repository layout, see:

- `docs/TEACHER_REVIEW_GUIDE.md`

The public examples are the recommended quick verification path. The paper
experiment scripts are heavier reproduction runs.

## Installation And Build

Real simulations require the compiled PolyFEM C++ backend. Build notes,
nanobind details, and platform-specific installation commands are documented
separately:

- `BUILD.md`

Quick backend check:

```bash
python -c "import polyfempy as pf; print(pf.cpp_backend_available()); print(pf.cpp_backend_error())"
```

Expected status for real solves:

```text
True
None
```

## Scope

This repository is infrastructure for simulation and scientific ML workflows.
It does not propose a new ML model. The primary artifact is the Python/PyTorch
interface layer around PolyFEM, together with examples, tests, result
extraction, differentiable operators, and reproducibility scripts.
