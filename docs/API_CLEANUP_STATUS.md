# API Cleanup Status

This note summarizes the current API cleanup work for the NeurIPS
infrastructure-style artifact. It is intended as an engineering handoff note for
future ChatGPT/Codex sessions, the project author, or an advisor reviewing the
repository structure.

## Short Report

PolyFEM-Python is being cleaned up into a paper-facing infrastructure artifact:
a Python and PyTorch interface to high-fidelity PolyFEM simulation, with
programmatic configuration, structured result objects, contact examples,
differentiable simulation helpers, and reproducible paper experiments.

The main direction is to make the public API look like a usable library rather
than a collection of experiments. The important user-facing imports are now:

```python
from polyfempy.api import solve, SimulationConfig, Result
import polyfempy.api.guided as g
from polyfempy.differentiable import (
    make_von_mises_loss,
    prepare_differentiable_simulation,
    prepare_optimization_problem,
    prepare_parameterized_shape_problem,
    run_optimization,
)
```

The cleaned public examples live in `examples/`, while research and HPC scripts
remain under `experiment/`.

Phase 3 adds TOMS-facing contract documentation:

- `docs/API_STABILITY.md`: stable public API vs compatibility vs internal.
- `docs/GUIDED_API.md`: guided section authoring contract.
- `docs/CONFIG_CONTRACT.md`: `SimulationConfig` and JSON semantics.
- `docs/RESULT_CONTRACT.md`: `Result` fields/history/sampled-data semantics.
- `docs/EXAMPLES_MATRIX.md`: examples-to-capability matrix.
- `docs/TOMS_REVIEW_CHECKLIST.md`: reviewer-style API/artifact checklist.
- `docs/ARTIFACT_REPRODUCIBILITY.md`: minimal reproducibility command path.
- `docs/TEST_MATRIX.md`: cleanup-slice test subset matrix.

Phase 3 also keeps `Result` lightweight but clearer for users:

```python
result.field("von_mises")          # merged lookup: point -> cell -> sampled
result.point_field("u")            # native point namespace only
result.cell_field("material_id")   # native cell namespace only
result.sampled_field("stress")     # sampled/probe namespace only
result.available_fields()          # names grouped by namespace
```

The current teacher-facing paper demos live in
`experiment/paper_experiment/`.  The clean h/theta shape optimization demo is
`experiment/paper_experiment/08_h_theta_shape_optimization.py`; the longer
`07_h_theta_fix06_global_affine_vertex_map.py` is the experiment driver with
reporting, early stopping, and mesh snapshots.

## What Was Done

### Public API Packaging

- Moved guided section helpers into the package boundary:
  - `polyfempy/api/guided_sections.py`
- Updated:
  - `polyfempy/api/guided.py`
- The guided API no longer imports from `experiment.*`.
- The old experiment helper path is no longer required for package imports.

### Legacy API Removal

- Removed the `polyfempy/legacy/` package from the public library.
- Migrated still-needed problem helper classes into:
  - `polyfempy/api/problems.py`
- Updated:
  - `polyfempy/api/config.py`
- `SimulationConfig.to_settings()` now resolves predefined problem helpers from
  `polyfempy.api.problems`, not from a legacy namespace.
- Preserved predefined problem support:
  - `Franke`
  - `GenericScalar`
  - `Gravity`
  - `Torsion`
  - `TorsionElastic` alias
  - `GenericTensor`
  - `Flow`
  - `DrivenCavity`
  - `FlowWithObstacle`
- Fixed the old `TorsionElastic` lookup issue by mapping it to `Torsion`.
- Preserved `Flow` compatibility with both `inflow_amout` and `inflow_amount`.

### CI And Tests

- Replaced stale GitHub Actions test paths with a simple workflow:

```bash
python -m pytest tests
```

- Added public import smoke tests:
  - `tests/test_import_public_api.py`
- Added predefined problem helper tests:
  - `tests/test_api_problems.py`
- Added a backend smoke test with graceful skip:
  - `tests/test_backend_smoke.py`
- Verified:

```text
227 passed when the compiled backend is available
226 passed, 1 skipped when the compiled backend is unavailable
```

### Public Examples

Added a clean top-level `examples/` directory:

- `examples/01_forward_solve.py`
- `examples/02_result_fields.py`
- `examples/03_shape_gradient.py`
- `examples/04_scalar_E_gradient.py`
- `examples/05_parameterized_vertex_map.py`
- `examples/06_dataset_one_case.py`
- `examples/configs/contact_impact.json`
- `examples/assets/impact/*.msh`

These examples do not depend on Compute Canada or private paper experiment
paths. Generated outputs go under `examples/runs/`, which is ignored by git.

### Documentation

Added or rewrote:

- `README.md`
- `BUILD.md`
- `ARTIFACT.md`
- `examples/README.md`
- `experiment/paper_experiment/README.md`
- `experiment/paper_experiment/CLEAN_API_WALKTHROUGH.md`
- `docs/TEACHER_REVIEW_GUIDE.md`

The README now starts with the library value proposition instead of build notes.
Build details live in `BUILD.md`. Artifact/reproduction guidance lives in
`ARTIFACT.md`.

## Current API Shape

Forward solve:

```python
from polyfempy.api import SimulationConfig, solve

cfg = SimulationConfig.from_json_file("examples/configs/contact_impact.json")
result = solve(cfg=cfg)
print(result.u.shape)
print(result.vertices.shape)
```

Guided config:

```python
import polyfempy.api.guided as g

template = g.simulation_template(...)
cfg = g.build_config(template, workspace)
```

Differentiable shape:

```python
result = prepare_differentiable_simulation(cfg=cfg, derivative_type="shape")
loss = make_von_mises_loss(result=result, body=1, time="smooth_max")
loss.backward()
```

Parameterized shape:

```python
problem = prepare_parameterized_shape_problem(cfg=cfg, vertex_map=my_vertex_map)
```

Scalar Young's modulus optimization:

```python
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
```

Stable differentiable paths:

- shape gradients through `prepare_differentiable_simulation(..., derivative_type="shape")`
- scalar Young's modulus optimization through `prepare_optimization_problem(..., kind="material")`
- fixed-topology parameterized shape optimization through
  `prepare_parameterized_shape_problem(...)` and a user `vertex_map`

Teacher-facing paper demos:

- `experiment/paper_experiment/01_forward_von_mises.py`: forward `solve(cfg=cfg)`
- `experiment/paper_experiment/02_shape_diff.py`: shape gradient chain
- `experiment/paper_experiment/03_E_diff.py`: scalar `E` gradient chain
- `experiment/paper_experiment/04_x_shape_optimization.py`: raw vertex optimization
- `experiment/paper_experiment/08_h_theta_shape_optimization.py`: clean h/theta parameterized shape optimization

Advanced or experimental differentiable paths:

- raw `derivative_type="material"` use
- initial-condition or initial-velocity gradients
- low-level torch-bridge and finite-difference diagnostics

## API Questions To Decide Next

These are the most important open design questions if the goal is a polished
NeurIPS artifact.

1. Should `SimulationConfig` remain the main public entry point, or should the
   guided helpers become the recommended first user path?

2. Should `build_config(...)` accept only guided templates, or should it also
   accept a plain dict-style schema for advanced users?

3. Should contact be exposed mainly through `SimulationConfig.contact`, through
   `contact_section(...)`, or both equally?

4. Should `Result.von_mises`, `Result.stress`, and `Result.strain` always return
   arrays, or should missing/derived fields require explicit `result.field(...)`
   calls so users understand fallback behavior?

5. Should `Result.to_torch()` convert all numeric fields by default, or should
   users choose which fields to convert?

6. Should differentiable modes be split into named functions instead of a
   `derivative_type` string? For example:
   - `prepare_shape_gradient_problem(...)`
   - `prepare_material_gradient_problem(...)`
   - `prepare_parameterized_shape_problem(...)`

7. Should experimental differentiable paths such as initial velocity gradients
   be hidden under `polyfempy.differentiable.advanced` until they are stable?

8. Should examples write into timestamped run folders by default, or should they
   use deterministic output directories for easier artifact checking?

9. Should the public package include example meshes, or should example assets be
   repository-only and not part of the installable wheel?

10. Should old predefined problems such as `Gravity`, `Franke`, and `Flow` be
    advertised as public API, or treated as compatibility helpers only?

## Remaining Risks

### P0

- Commit hygiene: make sure unrelated generated files, local logs, and `.codex`
  state are not accidentally committed.
- Confirm that deleting `polyfempy/legacy/` is intentional and that downstream
  code should use `polyfempy.api.problems` if it needs predefined problem
  helpers.

### P1

- Decide whether `polyfempy.api.problems` should be public or internal.
- Clarify result-field provenance: direct backend arrays vs VTU/meshio fallback.
- Keep the backend smoke test lightweight enough for local artifact validation.

### P2

- Move old generator/prototype material out of the root if it is not part of the
  artifact story.
- Consider moving Compute Canada notes under `docs/` after paper-facing docs are
  stable.
- Add deterministic "artifact smoke" commands for reviewers.

## Commands To Recheck

```bash
python -m pytest tests
```

Expected:

```text
227 passed if the compiled backend is available
226 passed, 1 skipped if the compiled backend is unavailable
```

Public import check:

```bash
python -m pytest tests/test_import_public_api.py tests/test_api_problems.py tests/test_backend_smoke.py
```

Backend quickstart check:

```bash
python examples/01_forward_solve.py
python examples/02_result_fields.py
python examples/03_shape_gradient.py
```

## Suggested Next Step

Do a final pre-commit review of `git status --short` and split the cleanup into
intentional commits. Avoid mixing generated logs, local run outputs, or editor
state into the artifact cleanup commit.
