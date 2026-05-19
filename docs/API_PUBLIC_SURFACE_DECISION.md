# API Public Surface 决策

这个文件记录 Phase 2 的 public API 决策。目标是让 TOMS reviewer、
advisor、未来用户和我们自己都能快速看懂：

- 推荐用户 import 什么；
- 哪些名字只是兼容或高级用法；
- 哪些模块是内部实现；
- 当前为什么先不大拆代码。

这个决策偏保守。原则是先稳定软件契约，再考虑文件拆分。

## 总结

推荐 public surface 保持小：

```python
from polyfempy.api import solve, SimulationConfig, Result
```

guided config 使用独立 facade：

```python
import polyfempy.api.guided as g

template = g.simulation_template(
    problem=g.problem_section(...),
    bodies=g.bodies_section(...),
    solver=g.solver_section(...),
    results=g.results_section(...),
)
cfg = g.build_config(template, workspace)
```

differentiable API 保持在 `polyfempy.differentiable`：

```python
from polyfempy.differentiable import (
    make_von_mises_loss,
    prepare_differentiable_simulation,
    prepare_optimization_problem,
    prepare_parameterized_shape_problem,
    run_optimization,
)
```

核心判断：

- `polyfempy.api` 顶层不应该继续被描述成“什么都推荐 import”。
- `polyfempy.api.__all__` 只保留推荐小 API；兼容/高级名字仍可显式 import。
- `guided.py` 应该保留为 guided API 的 public facade。
- `_solve_pipeline.py`、`_guided_array_mesh.py`、`_runtime.py` 是 internal implementation。
- `config.py` 暂时不拆，先用文档和测试保护语义。

## 为什么这样符合 TOMS

TOMS 关心 mathematical software 的 development、evaluation、use。对这个仓库来说，reviewer 需要看到的是：

- 软件入口清楚；
- 配置、求解、结果对象的契约清楚；
- 内部 pipeline 和用户 API 分开；
- examples 和 tests 能证明这些入口可用；
- artifact/reproduction 路径能跑。

所以当前最重要的不是把文件拆得很细，而是让 public contract 很明确。

## 推荐 Public API

### 顶层 API

这三个名字是首页级推荐入口：

| 名字 | 路径 | 推荐状态 | 说明 |
| --- | --- | --- | --- |
| `solve` | `polyfempy.api` | 推荐 public | forward simulation 的唯一主入口。 |
| `SimulationConfig` | `polyfempy.api` | 推荐 public | solver-facing config contract。 |
| `Result` | `polyfempy.api` | 推荐 public | `solve(...)` 返回的 structured result object。 |

推荐写法：

```python
from polyfempy.api import SimulationConfig, Result, solve
```

### Guided API

这些是推荐用户用来构造 `SimulationConfig` 的 guided authoring layer：

| 名字 | 推荐状态 | 说明 |
| --- | --- | --- |
| `problem_section` | 推荐 public | PDE / problem-level 设置。 |
| `units_section` | 推荐 public | 单位系统。 |
| `material_section` | 推荐 public | material model 和参数。 |
| `body_section` | 推荐 public | 单个 body 的 mesh/material/BC。 |
| `bodies_section` | 推荐 public | 多个 body 的组合。 |
| `fixed_surface_section` | 推荐 public | 常用 Dirichlet fixed region。 |
| `loads_section` | 推荐 public | RHS / body force。 |
| `space_section` | 推荐 public | discretization。 |
| `time_section` | 推荐 public | transient time settings。 |
| `solver_section` | 推荐 public | solver settings。 |
| `contact_section` | 推荐 public | contact settings。 |
| `output_section` | 推荐 public | output policy。 |
| `results_section` | 推荐 public | requested result fields。 |
| `simulation_template` | 推荐 public | 把 sections 组合成 generic simulation template。 |
| `build_config` | 推荐 public | template -> `SimulationConfig`。 |
| `experiment_template` | 兼容 alias | 旧 examples 名字，继续可用但新文档不推荐。 |

`polyfempy.api.guided.__all__` 只包含推荐的 section factory / builder
functions。Section dataclasses、Literal type aliases 和 `experiment_template`
仍然是显式可访问的 compatibility/advanced attributes，但不进入推荐
star-import surface。

推荐写法：

```python
import polyfempy.api.guided as g

body = g.body_section(...)
template = g.simulation_template(bodies=g.bodies_section(body))
cfg = g.build_config(template, workspace)
```

### Runtime / Reporting Helpers

这些可以保留，但不要作为核心 API 主推：

| 名字 | 路径 | 推荐状态 | 说明 |
| --- | --- | --- | --- |
| `make_timestamped_workspace` | `polyfempy.api.runtime` | 脚本 helper | examples/paper demos 方便建 workspace。 |
| `terminal_log` | `polyfempy.api.runtime` | 脚本 helper | 配置日志。 |
| `result_output` | `polyfempy.api.runtime` | 脚本 helper | 配置 JSON/VTU/history 输出。 |
| `solve_with_timing` | `polyfempy.api.runtime` | reusable helper | 返回 `(result, elapsed_seconds)`，比 demo wrapper 更适合复用。 |
| `solve_and_report` | `polyfempy.api.runtime` | 脚本 helper | 方便 demo，不是核心求解契约。 |
| `summarize_result` | `polyfempy.api.report` | report helper | 对 `Result` 做 summary。 |
| `format_result_summary` | `polyfempy.api.report` | report helper | 文本 summary。 |
| `summarize_history_bundle` | `polyfempy.api.report` | report helper | transient history summary。 |

建议文档写法：

```python
from polyfempy.api.runtime import make_timestamped_workspace, result_output, terminal_log
```

`polyfempy.api.runtime.__all__` 和 `polyfempy.api.report.__all__` 表示模块自己的
advanced reusable surface；`emit_history_bundle` / `solve_and_report` 继续支持显式
import，但不作为 `runtime` star-import 推荐入口。

不要把这些写成“必须 import 的核心 API”。

## 兼容/高级导出

`polyfempy.api.__all__` 现在只导出 `CORE_API`。typed config blocks 和
runtime/reporting helpers 仍然是 `polyfempy.api` 的 module attributes，所以旧脚本
里的显式 import 仍然可用；只是它们不再进入 star-import surface。

这些名字可以继续显式 import，但文档上降级为高级或兼容：

```text
Quantity
Material classes
Boundary condition classes
Geometry classes
Solver / Time / Output / Contact classes
Selection
Mesh / read_mesh
configure_windows_runtime
```

推荐规则：

- 首页和 examples 主推 `solve`, `SimulationConfig`, `Result`。
- 需要细粒度 config 时，引导用户去 `polyfempy.api.config`。
- 需要 guided config 时，引导用户去 `polyfempy.api.guided`。
- 需要 runtime/report 时，引导用户去 `polyfempy.api.runtime` 或 `polyfempy.api.report`。
- 不推荐用户依赖 `from polyfempy.api import *`；如果使用 star import，它只代表
  `solve`, `SimulationConfig`, `Result`。

## Internal-Only API

这些模块不应该被普通用户 import：

| 名字 | 状态 | 说明 |
| --- | --- | --- |
| `polyfempy.api._solve_pipeline` | internal | staged solve implementation。 |
| `polyfempy.api._guided_array_mesh` | internal | guided array-backed mesh payload builder。 |
| `polyfempy.api._runtime` | internal | Windows runtime shim implementation。 |

tests 可以直接 import internal helper，这是为了保护 pipeline 语义，不代表用户应该 import。

## `solve.py` 兼容 alias 决策

`solve.py` 现在只把 `solve` 放进 `__all__`。旧 helper imports 仍然显式可用，
但 compatibility map 已移到 `_solve_compat.py`，避免 public facade 继续显得像
内部工具集合。

`_solve_compat.py` 维护这些 backward-compatible aliases：

```text
_process_json_config
_clean_json_for_cpp
_merge_user_cfg_over_full_json
_extract_runtime_output_request
_extract_additional_fields
_maybe_fill_result_from_temp_vtu
_finalize_result_output
_reconstruct_sampled_cauchy_stress
_extract_meshio_array
_field_available
```

当前 repo 内没有真实内部 caller 依赖这些 alias。tests 现在直接 import
`polyfempy.api._solve_pipeline` 里的 helper。

Phase 3 决策：

- 保留这些 alias，标记为 compatibility-only。
- 在 `_solve_compat.py` 中用 `COMPATIBILITY_ALIAS_TARGETS` 显式记录每个旧名字指向
  staged solve internals 的哪个 current implementation。
- 不在用户文档中推荐。
- 不新增对这些 alias 的使用。
- 当前不删除、不加 deprecation warning，避免无谓打扰旧脚本；未来如果要删除，
  需要单独做 migration note。

## `guided_sections.py` 决策

Phase 2 先不拆 `guided_sections.py`；Phase 3 已经把真实职责拆出来，
但保留旧 import path。

理由：

- `guided.py` 已经提供清楚 public facade。
- `guided_sections.py` 现在是 compatibility facade，继续 re-export factories/types/config helpers。
- `guided_builders.py` 只负责创建 section objects。
- `guided_types.py` 只负责 dataclasses 和 type aliases。
- `_guided_config.py` 负责 `build_config(...)` 和 template -> `SimulationConfig` translation。

当前规则：

- 文档上明确推荐哪些 section factory；
- 继续保持 `guided.py.__all__` 为 factory-only 推荐 surface；
- 不让 public facade 反向依赖 `polyfempy.api` 顶层 facade；
- 所有旧 `polyfempy.api.guided_sections` import path 继续可用。

## `config.py` 决策

`config.py` 继续作为稳定 facade，旧 import path 不变。第一步只把 solver /
time / output typed blocks 拆到 `config_solver.py` / `config_time.py` /
`config_output.py`，并由 `config.py` re-export。

Phase 4 已经做的低风险整理：

- 在 `config.py` 顶部说明这个文件为什么暂时保持为宽文件；
- `Solver` / `LinearSolver` / `NonlinearSolver` typed blocks 移到 `config_solver.py`；
- `Time` / integrator typed blocks 移到 `config_time.py`；
- `Output*` typed blocks 移到 `config_output.py`；
- 给 helper、materials、boundary/initial conditions、body/constraint/space/input、
  `SimulationConfig`、geometry、solver、time、contact 等区块加结构标记；
- 不改 import path，不改 JSON 语义。

必须先保护这些语义：

- `to_full_json_*` / `from_full_json_*`
- `to_minimal_json_*` / `from_minimal_json_*`
- `from_json_str(..., kind="auto")`
- `from_json_file(...)` 的 `_root_path`
- `to_dict()` 从 `_full_json_config` overlay Python-side edits
- `add_body(...)` 对齐 material id 和 geometry `volume_selection`
- `solver.nonlinear.Newton.residual_tolerance` 等 solver method blocks

未来如果拆，必须保持这些 import path：

```python
from polyfempy.api import SimulationConfig
from polyfempy.api.config import SimulationConfig, Output, Solver, Time
```

## Test Matrix

只改 public import/export：

```bash
python -m pytest tests/test_import_public_api.py
```

改 `solve.py` / `_solve_pipeline.py`：

```bash
python -m pytest \
  tests/test_import_public_api.py \
  tests/test_pipeline_normalize.py \
  tests/test_pipeline_runtime_options.py \
  tests/test_pipeline_clean_json.py \
  tests/test_pipeline_extract_outputs.py \
  tests/test_pipeline_helpers.py \
  tests/test_pipeline_sampled_fallback.py
```

改 guided API：

```bash
python -m pytest \
  tests/test_import_public_api.py \
  tests/test_config_typed_blocks.py \
  tests/test_geometry_transformations.py
```

改 `config.py`：

```bash
python -m pytest \
  tests/test_config_json_io.py \
  tests/test_config_typed_blocks.py \
  tests/test_config_validate.py \
  tests/test_solver_method_blocks.py \
  tests/test_pipeline_normalize.py \
  tests/test_pipeline_runtime_options.py
```

改 `result.py` / `report.py`：

```bash
python -m pytest \
  tests/test_result_history.py \
  tests/test_result_meshio_roundtrip.py \
  tests/test_result_report.py \
  tests/test_result_sampled_data.py
```

backend 可用时补：

```bash
python -m pytest tests/test_backend_smoke.py
```

## 文档写法约定

后续 README / examples / paper docs 应该统一成这句话：

```text
推荐用户入口是 solve, SimulationConfig, Result。
guided sections 是配置 authoring layer。
_solve_pipeline 等下划线模块是 internal implementation。
```

这套写法比“所有 helper 都是 public API”更适合 TOMS，因为它让软件边界、使用方式、测试目标都更清楚。
