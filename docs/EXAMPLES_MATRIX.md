# Examples Matrix

这个文件把 `examples/` 目录里的用户入口和 API capability 对齐。目标是让
TOMS reviewer 快速看到：

- 每个 example 展示什么软件能力；
- 它使用哪些 public API；
- 需要哪些额外依赖；
- 对应哪些测试或 smoke check。

所有 top-level examples 都应该使用推荐 public imports：

```python
from polyfempy.api import solve
import polyfempy.api.guided as g
```

需要 differentiable path 时，再 import：

```python
from polyfempy.differentiable import ...
```

## Layering

`examples/` 不需要覆盖所有 paper experiment。它应该保持少而清楚：

| Layer | 例子 | 目的 |
| --- | --- | --- |
| core | `01_forward_solve.py`, `02_result_fields.py` | 新用户第一入口，展示 forward solve 和 `Result`。 |
| advanced | `03_shape_gradient.py` 到 `06_dataset_one_case.py` | 展示 differentiable solve、parameterized shape、dataset export。 |
| paper reproduction | `experiment/paper_experiment/` | 论文实验、报告、HPC、长 run，可随论文需求变化。 |

## Matrix

| Example | Layer | Capability | Public API Used | Extra Dependency | Suggested Check |
| --- | --- | --- | --- | --- | --- |
| `examples/01_forward_solve.py` | core | Guided config + forward solve。 | `solve`, `g.simulation_template`, `g.build_config` | backend only | `python examples/01_forward_solve.py` |
| `examples/02_result_fields.py` | core | Structured `Result` fields + VTK export。 | `solve`, `Result.available_fields`, `Result.point_field`, `Result.field`, `Result.write` | `meshio` | `python examples/02_result_fields.py` |
| `examples/03_shape_gradient.py` | advanced | `d loss / d vertices` shape sensitivity。 | `prepare_optimization_problem(kind="shape")`, `make_von_mises_loss`, `shape_gradient_for_body` | PyTorch | `python examples/03_shape_gradient.py` |
| `examples/04_scalar_E_gradient.py` | advanced | scalar Young's modulus gradient `dL/dE`。 | `prepare_optimization_problem(kind="material")`, `make_von_mises_loss` | PyTorch | `python examples/04_scalar_E_gradient.py` |
| `examples/05_parameterized_vertex_map.py` | advanced | named parameters -> vertex map -> gradient。 | `prepare_parameterized_shape_problem`, user `vertex_map` | PyTorch | `python examples/05_parameterized_vertex_map.py` |
| `examples/06_dataset_one_case.py` | advanced | one local supervised training sample export。 | `save_training_sample`, shape gradient path | PyTorch | `python examples/06_dataset_one_case.py` |

## What Each Example Should Keep Visible

The examples should keep the core API steps visible instead of hiding them in
example-only wrappers.

Forward solve examples should show:

```text
g.simulation_template(...)
g.build_config(...)
solve(cfg=cfg)
read Result fields
```

Differentiable examples should show:

```text
prepare_optimization_problem(...)
problem.solve()
make_von_mises_loss(...)
loss.backward()
read gradient
```

Parameterized shape examples should show:

```text
torch.nn.Parameter(...)
vertex_map(params, base_vertices, ...)
prepare_parameterized_shape_problem(...)
run solve / backward / optimization
```

## Relationship To Paper Experiments

`examples/` 是 small public tutorial layer。

`experiment/paper_experiment/` 是 paper-facing reproduction layer。它可以包含：

- longer reporting；
- environment variable overrides；
- Compute Canada helper scripts；
- before/after figures；
- mesh snapshots；
- result summaries。

但 paper experiments should still use the same public API surface:

```python
from polyfempy.api import solve
import polyfempy.api.guided as g
from polyfempy.differentiable import ...
```

This keeps the paper demos inspectable: reviewer can see which parts are
library API and which parts are paper-specific geometry/reporting code.

## Dependencies

| Capability | Dependency |
| --- | --- |
| forward solve | compiled PolyFEM backend |
| mesh export/import | `meshio` |
| differentiable solve / optimization | PyTorch |
| local examples | checked-in meshes under `examples/assets/impact/` |
| Compute Canada runs | cluster environment and job scripts under `experiment/` |

The top-level examples intentionally use checked-in meshes so they do not
depend on Gmsh or a cluster environment.

## Tests That Protect The Examples

Examples are not a replacement for tests. The nearby test coverage is:

| Test | Protects |
| --- | --- |
| `tests/test_import_public_api.py` | public imports, guided namespace, `simulation_template` alias。 |
| `tests/test_config_typed_blocks.py` | typed config blocks used by guided builders。 |
| `tests/test_geometry_transformations.py` | geometry helpers used by guided body construction。 |
| `tests/test_pipeline_normalize.py` | `solve(...)` input normalization。 |
| `tests/test_pipeline_extract_outputs.py` | native output extraction into `Result`。 |
| `tests/test_pipeline_sampled_fallback.py` | history/sampled fallback behavior。 |
| `tests/test_result_*` | `Result` fields, history, meshio round-trip, reporting。 |

When the compiled backend is available, examples themselves can be run as
integration smoke checks.
