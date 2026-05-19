# polyfempy Package Map

This package has a small recommended user surface and a larger implementation
layer underneath it.

The current public-surface decision is documented in
`../docs/API_PUBLIC_SURFACE_DECISION.md`. The corresponding import audit is in
`../docs/API_INTERNAL_IMPORT_AUDIT.md`.

## Recommended Imports

Forward simulations:

```python
from polyfempy.api import SimulationConfig, Result, solve
```

Guided config construction:

```python
import polyfempy.api.guided as g
```

Differentiable solves and optimization:

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

Use plain PyTorch parameters for optimized scalar values:

```python
import torch

E_lattice = torch.nn.Parameter(torch.tensor(20.0))
h = torch.nn.Parameter(torch.tensor(0.04))
theta_deg = torch.nn.Parameter(torch.tensor(90.0))
```

## Main Layers

| Path | Role |
| --- | --- |
| `api/solve.py` | Thin public forward-solve entry point. |
| `api/_solve_pipeline.py` | Internal staged solver pipeline used by `solve(...)`. |
| `api/config.py` | Typed Python wrappers for PolyFEM JSON/config fields. |
| `api/guided.py` | Public guided section facade. |
| `api/guided_builders.py` | Guided section factory functions. |
| `api/_guided_config.py` | Guided template to `SimulationConfig` translation. |
| `differentiable/runtime/` | Differentiable solve runtime, settings contract, autograd bridge, result objects. |
| `differentiable/objectives/` | Differentiable von Mises/stress objective builders. |
| `differentiable/optimization/` | Public optimization dispatcher, runner, result, reports, summaries. |
| `differentiable/shape/` | Shape and parameterized-shape problems, masks, and geometry maps. |
| `differentiable/material/` | Scalar material optimization, config parsing, diagnostics, and material parameters. |
| `differentiable/data/` | Training-sample export helpers. |
| `differentiable/design.py` | Adapter for `parameters -> vertex_map -> vertices`. |
| `differentiable/advanced.py` | Diagnostics and legacy compatibility helpers. |

## What Is Implementation Detail

Large files such as `api/config.py` are not hidden optimization logic. They
handle JSON schema coverage. Guided config construction and backend solve
orchestration live behind focused internal modules such as `_guided_config.py`
and `_solve_pipeline.py`.

The paper-facing optimization demos should not call those internals directly.
They should stay on:

```text
g.build_config / solve
prepare_optimization_problem
prepare_parameterized_shape_problem
make_von_mises_loss
run_optimization
```

For new optimization scripts, prefer:

```python
run = run_optimization(..., return_result=True)
run.summary()
```

Lower-level helpers remain importable for older experiments, but the docs and
examples should treat them as compatibility or advanced APIs.
