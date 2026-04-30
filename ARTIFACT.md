# Artifact Guide

This guide is intended for artifact reviewers and users who want to verify the
repository as a NeurIPS infrastructure-style artifact.

The main contribution of this repository is Python/PyTorch infrastructure for
PolyFEM. It is not a new machine-learning model. The artifact provides a public
Python configuration API, guided simulation helpers, differentiable PyTorch
interfaces, result extraction utilities, small user-facing examples, and paper
experiment scripts built on top of the PolyFEM C++ backend.

## 1. What This Repository Provides

This repository contains:

- `polyfempy.api`: public Python APIs for building and running simulations.
- `polyfempy.api.guided`: higher-level guided config sections such as bodies,
  materials, contact, time stepping, and output.
- `polyfempy.differentiable`: PyTorch-facing differentiable simulation and
  optimization helpers.
- `examples/`: short user-facing examples that do not depend on Compute Canada.
- `tests/`: import, config, result, runtime, and pipeline tests.
- `experiment/experiment_api_solve/paper_experiments/`: paper-facing
  optimization/reproduction scripts.
- `experiment/compute_canada_training_cases/`: larger HPC dataset-generation
  workflows.

The intended use cases are:

- Running PolyFEM simulations from Python.
- Building simulation configs without hand-writing large JSON files.
- Differentiating simulation losses through PyTorch.
- Inspecting result fields such as displacement, stress, and von Mises stress.
- Reproducing paper experiments and generating training-style data.

## 2. Installation

### Basic Python Dependencies

From the repository root:

```bash
python -m pip install --upgrade pip
python -m pip install numpy pytest meshio
```

For differentiable examples:

```bash
python -m pip install torch
```

### Full Source Build

Real solves require the compiled PolyFEM C++ backend. For a source checkout,
install build tools and build the extension:

```bash
python -m pip install --upgrade pip setuptools wheel cmake nanobind
python -m pip install -e . --no-build-isolation
```

Verify the backend:

```bash
python -c "import polyfempy as pf; print(pf.cpp_backend_available()); print(pf.cpp_backend_error())"
```

Expected backend status for real simulations:

```text
True
None
```

If the backend is not available, pure-Python import/config tests can still be
useful, but `solve(...)`, contact simulation, and differentiable examples will
not run.

## 3. 3-Minute Quickstart

Run these from the repository root after the C++ backend is available:

```bash
python examples/01_forward_solve.py
python examples/04_differentiable_shape_gradient.py
python examples/06_parameterized_vertex_map.py
```

These examples write timestamped outputs under:

```text
examples/runs/
```

The generated run folders are ignored by git.

## 4. Public API Overview

Forward simulation:

```python
from polyfempy.api import solve, SimulationConfig, Result
```

Guided config construction:

```python
from polyfempy.api.guided import (
    body_section,
    material_section,
    contact_section,
    build_config,
)
```

Differentiable simulation:

```python
from polyfempy.differentiable import (
    prepare_differentiable_simulation,
    solve_differentiable,
    make_von_mises_loss,
)
```

Optimization helpers:

```python
from polyfempy.differentiable import (
    make_parameter,
    prepare_scalar_youngs_material_problem,
    prepare_parameterized_shape_problem,
    run_optimization,
)
```

Differentiable stability labels:

- Stable: shape gradients with `derivative_type="shape"`.
- Stable: scalar Young's modulus optimization through
  `prepare_scalar_youngs_material_problem(...)`.
- Stable: fixed-topology parameterized shape optimization through
  `prepare_parameterized_shape_problem(...)` and a user `vertex_map`.
- Advanced / experimental: initial-condition or initial-velocity gradients,
  raw `derivative_type="material"` use, and low-level torch-bridge diagnostics.

Typical forward flow:

```python
cfg = build_config(template, workspace)
result = solve(cfg=cfg)
```

Typical differentiable flow:

```python
result = prepare_differentiable_simulation(cfg=cfg, derivative_type="shape")
loss = make_von_mises_loss(result=result, body=1, time="smooth_max")
loss.backward()
shape_grad = result.shape_gradient
```

## 5. How To Run Examples

All public examples live in `examples/`:

```bash
python examples/01_forward_solve.py
python examples/02_contact_impact.py
python examples/03_result_fields.py
python examples/04_differentiable_shape_gradient.py
python examples/05_scalar_E_optimization.py
python examples/06_parameterized_vertex_map.py
python examples/07_dataset_one_case.py
```

Example coverage:

- `01_forward_solve.py`: guided config plus `solve(cfg=...)`.
- `02_contact_impact.py`: two-body transient contact impact.
- `03_result_fields.py`: `Result` fields and VTK-compatible export.
- `04_differentiable_shape_gradient.py`: `d loss / d vertices`.
- `05_scalar_E_optimization.py`: scalar Young's modulus optimization.
- `06_parameterized_vertex_map.py`: user-defined `vertex_map` for named shape
  parameters.
- `07_dataset_one_case.py`: one local training-sample export.

These examples use checked-in meshes under `examples/assets/impact/`, so they do
not require Gmsh or a Slurm/Compute Canada environment.

## 6. How To Run Tests

Run the repository test suite:

```bash
python -m pytest tests
```

The current minimal CI workflow also runs:

```bash
python -m pytest tests
```

The tests include public import smoke tests:

```python
from polyfempy.api import solve, SimulationConfig, Result
from polyfempy.api.guided import body_section, material_section, contact_section, build_config
from polyfempy.differentiable import solve_differentiable, prepare_differentiable_simulation
```

Many tests are designed to avoid requiring the compiled backend, so they can
still check the Python API boundary in lightweight environments.

The suite also contains a backend smoke test:

```bash
python -m pytest tests/test_backend_smoke.py
```

If the compiled C++ backend is unavailable, this test is skipped. If the backend
is available, it runs a one-step forward `solve(cfg=...)` using the public
example contact config and asserts that displacement and mesh arrays are
returned.

## 7. How To Reproduce Paper Experiments

The paper-facing optimization scripts live in:

```text
experiment/experiment_api_solve/paper_experiments/
```

Run from the repository root after activating a working environment:

```bash
python -m experiment.experiment_api_solve.paper_experiments.paper_E_opt
python -m experiment.experiment_api_solve.paper_experiments.paper_shape_opt
python -m experiment.experiment_api_solve.paper_experiments.paper_h_theta_autograd_opt
```

The h/theta vertex-map API comparison runs are:

```bash
python -m experiment.experiment_api_solve.paper_experiments.paper_h_theta_autograd_opt
python -m experiment.experiment_api_solve.paper_experiments.paper_h_theta_helper_vertex_map_test
python -m experiment.experiment_api_solve.paper_experiments.paper_h_theta_inline_vertex_map_test
```

For Compute Canada / Slurm runs, see:

```text
experiment/experiment_api_solve/paper_experiments/README.md
```

The Slurm scripts are intentionally separate from the public examples:

```bash
sbatch experiment/experiment_api_solve/paper_experiments/sbatch_paper_E_opt.sh
sbatch experiment/experiment_api_solve/paper_experiments/sbatch_paper_shape_opt.sh
sbatch experiment/experiment_api_solve/paper_experiments/sbatch_paper_h_theta_autograd_opt.sh
sbatch experiment/experiment_api_solve/paper_experiments/sbatch_paper_h_theta_helper_vertex_map_test.sh
sbatch experiment/experiment_api_solve/paper_experiments/sbatch_paper_h_theta_inline_vertex_map_test.sh
```

These paper runs are heavier than the public examples. Use the public examples
for quick API verification and the paper experiment scripts for reproduction.

## 8. Expected Outputs

Public examples write to:

```text
examples/runs/<example_name>_<timestamp>/
```

Common outputs include:

- `polyfem.log`
- `energy.csv`
- `stats.csv`
- `impact_stats.json`
- optional `.vtu` / `.pvd` visualization files
- `training_sample/metadata.json`
- `training_sample/*.npy`

Paper experiments write to:

```text
experiment/experiment_api_solve/paper_experiments/runs/<run_name>_<timestamp>/
```

Common paper-run outputs include:

- `before_opt/`
- `first_time_opt/`
- `history_summary.txt`
- `history_bundle.json`
- `optimization_metrics.csv`
- `optimization_metrics.json`
- `optimization_comparison.txt`
- gradient arrays such as `E_gradient.npy`, `shape_gradient.npy`, or
  `h_theta_gradient.npy`

## 9. Known Limitations

- The compiled C++ backend is required for real simulation solves.
- PyTorch is required for differentiable simulation and optimization examples.
- The repository is still an alpha-stage Python binding layer; API details may
  change.
- Some result fields, especially stress/von Mises fields, depend on backend
  history extraction or sampled-output fallback paths.
- Contact simulation is exposed through the guided Python config helpers, but it
  still depends on backend contact capabilities and solver settings.
- Parameterized `vertex_map` optimization uses fixed mesh topology. It
  differentiates through `params -> vertices -> loss`; it does not differentiate
  through Gmsh remeshing or changes in integer mesh connectivity.
- Guided array-backed mesh bodies currently have limited scope: they do not
  support mixing file-backed and array-backed bodies in one guided config, and
  advanced geometry transforms are not yet supported on array-backed bodies.
- Raw material derivative modes and initial-condition/initial-velocity
  differentiable paths are available only for advanced users and are not the
  recommended scalar-E or shape-optimization entry points.
- Paper experiments are heavier than the public examples and may need longer
  wall time or an HPC environment.
- Compute Canada scripts are provided for reproduction and dataset generation,
  not as the first user-facing API tutorial.
