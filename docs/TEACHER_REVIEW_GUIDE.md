# Teacher Review Guide

This repository has two layers:

1. The reusable Python/PyTorch API under `polyfempy/`.
2. Paper-facing demos and reproduction scripts under `experiment/`.

If the goal is to inspect the clean API surface, start with the files listed
below.  The older experiment folders remain in the repository for provenance,
but they are not the recommended API entry point.

## Recommended Reading Order

1. `README.md`
   - High-level project scope and public API summary.
2. `polyfempy/README.md`
   - Package-level map of the small recommended API and larger implementation
     layers.
3. `examples/`
   - Small standalone tutorials that do not require Compute Canada.
4. `experiment/paper_experiment/README.md`
   - Paper-facing Experiment 02 demos.
5. `experiment/paper_experiment/CLEAN_API_WALKTHROUGH.md`
   - Function-by-function explanation of the clean demo scripts.
6. `docs/API_FUNCTION_MAP.md`
   - Public helper call map: what each function contains, what it calls, and
     which pieces are shared API versus demo code.
7. `experiment/paper_experiment/08_h_theta_shape_optimization.py`
   - Clean h/theta parameterized shape optimization demo.

## Clean API Files

The clean paper demos are:

| File | Purpose |
| --- | --- |
| `01_forward_von_mises.py` | Forward solve and structured result fields. |
| `02_shape_diff.py` | Shape differentiation: `loss.backward()` gives `dL/dX`. |
| `03_E_diff.py` | Material differentiation: scalar Young's modulus parameter gets `E.grad`. |
| `04_x_shape_optimization.py` | Raw vertex-position shape optimization. |
| `08_h_theta_shape_optimization.py` | Parameterized shape optimization through explicit `h/theta -> vertices` map. |

The longer `07_h_theta_fix06_global_affine_vertex_map.py` is the experiment
driver. It keeps reporting, early stopping, and mesh snapshots for Compute
Canada and result inspection. It is useful for reproduction, but `08` is the
clean API version to read first.

## What The Clean Main Functions Do

The clean demos intentionally keep `main()` short.

Forward solve:

```text
build_config(...)
configure_output(...)
solve(cfg=cfg)
read Result fields
```

Shape differentiation:

```text
prepare_optimization_problem(cfg=cfg, kind="shape")
problem.solve()
make_von_mises_loss(...)
loss.backward()
shape_gradient_for_body(...)
```

Scalar material differentiation:

```text
E_lattice = torch.nn.Parameter(...)
prepare_optimization_problem(kind="material", E_parameter=E_lattice)
problem.solve()
make_von_mises_loss(...)
loss.backward()
E_lattice.grad
```

Parameterized h/theta shape optimization:

```text
h = torch.nn.Parameter(...)
theta_deg = torch.nn.Parameter(...)
prepare_parameterized_shape_problem(..., vertex_map=h_theta_vertex_map)
make_optimizer(...)
make_von_mises_loss(...)
run_optimization(...)
```

The h/theta differentiable chain is:

```text
h, theta_deg
  -> h_theta_vertex_map(...)
  -> vertices
  -> PolyFEM solve
  -> smooth-max von Mises loss
  -> backward
  -> h.grad, theta_deg.grad
```

## What Is Not Hidden API

`experiment/paper_experiment/common.py` only contains:

- paths and shared constants,
- workspace creation,
- PolyFEM output/log configuration,
- reporting helpers used by longer experiment scripts.

It does not define the solve, the loss, the backward pass, the optimizer loop,
or the h/theta geometry map used in the clean demo.

In `08_h_theta_shape_optimization.py`, the geometry map is explicit:

- `_build_static_geometry_data(...)` preprocesses the fixed reference mesh.
- `h_theta_vertex_map(...)` maps `h/theta` to vertices using Torch operations.
- `prepare_parameterized_shape_problem(...)` connects that map to the solver.

## Directory Map

| Path | Meaning |
| --- | --- |
| `polyfempy/api/` | Forward solve API, config objects, guided config builders. |
| `polyfempy/differentiable/` | PyTorch bridge, losses, optimization problem helpers. |
| `examples/` | Lightweight public examples. |
| `experiment/paper_experiment/` | Current paper-facing demos and Compute Canada scripts. |
| `experiment/prepare_paper/` | Notes and assets for manuscript preparation. |
| `experiment/archive/` | Local archived experiment material; ignored by git. |

## Push Hygiene

Before committing, avoid `git add -A` while there are unrelated tracked
deletions in `experiment/`.  Stage only the intended files.

Generated outputs should stay local:

- `runs/`
- `slurm_logs/`
- `outputs/`
- `training_data/`
- `zip_parts/`

These are ignored by `.gitignore`; already tracked files still require explicit
Git cleanup if the project decides to remove them from GitHub.
