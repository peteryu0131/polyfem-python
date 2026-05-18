# API Cleanup Phase 2 计划

这个文件是 Phase 2 的详细执行计划。Phase 1 已经完成了
`polyfempy/api` 的函数地图，见 `docs/API_FUNCTION_MAP.md`。Phase 2 的目标不是马上大规模重构，而是先把 API cleanup 的决策做扎实：

- 哪些名字应该成为推荐 public API；
- 哪些名字只是兼容导出；
- 哪些文件可以整理；
- 哪些文件不能先动；
- 每一个 cleanup slice 后应该跑哪些测试；
- 哪些语义必须保留，不能为了“看起来干净”而删掉。

一句话：Phase 2 是从“我知道每个函数在哪”推进到“我知道哪些能动、怎么动、动完怎么证明没坏”。

## 当前执行状态

本轮已经按稳定优先的方式执行前三个方案：

1. 方案 1：public surface 文档、import audit、README 统一。
   已新增 `API_PUBLIC_SURFACE_DECISION.md` 和
   `API_INTERNAL_IMPORT_AUDIT.md`，并在 README/docs 索引中连接。

2. 方案 2：只清 facade，确保用户入口小而清楚。
   已在 `polyfempy/api/__init__.py` 中明确 `CORE_API` 和
   `ADVANCED_COMPAT_API` 的区别，在 `solve.py` 中标出 compatibility
   aliases，在 `guided.py` 中明确它是 public guided facade。

3. 方案 3：内部注释、测试保护、轻量整理。
   已给 `_solve_pipeline.py` 加 internal-module 说明，并在
   `tests/test_import_public_api.py` 中加入 public surface / guided imports /
   solve compatibility aliases 的保护测试。

4. 本轮继续收口 guided/batch public surface。
   已新增 `simulation_template(...)` 作为 generic 推荐名，保留
   `experiment_template(...)` 作为 compatibility alias；examples 和 paper
   demos 改成 `import polyfempy.api.guided as g`；`batch_solve` 已从
   top-level facade 移除。

5. Phase 2 final cleanup：internal import boundary。
   `tests/test_pipeline_sampled_fallback.py` 已改成直接 import
   `polyfempy.api._solve_pipeline`，`guided_sections.py` 已改成从 `.config`
   和 `._guided_array_mesh` 做相对 import，不再反向依赖 package facade。

还没有执行的部分：

- 没有拆 `config.py`。
- 没有拆 `guided_sections.py`。
- 没有删除 `solve.py` 里的 compatibility aliases。
- 已审计并整理 `solve.py` compatibility aliases；它们现在通过
  `COMPATIBILITY_ALIAS_TARGETS` 明确指向 `_solve_pipeline` targets。
- 没有改变 `solve(...)`、`SimulationConfig`、`Result` 的行为语义。

## Phase 2 的核心目标

Phase 2 要完成三件事。

第一，明确 public surface：

```text
用户推荐 import 什么
  vs
为了兼容还导出什么
  vs
内部实现不应该被用户 import 什么
```

第二，明确 cleanup 顺序：

```text
低风险文档/导出整理
  -> solve pipeline 小整理
  -> guided API 小整理
  -> Result/report 小整理
  -> config.py 拆分方案
```

第三，明确验证矩阵：

```text
每改一个区域
  -> 跑对应 tests
  -> 确认 examples/paper demos 没有断 import
  -> 确认核心语义没变
```

## Phase 2 不是做什么

Phase 2 暂时不做这些事：

- 不直接删 public import。
- 不直接拆 `config.py`。
- 不直接改 `SimulationConfig` 的 full/minimal JSON 语义。
- 不直接改 `Result` 的 field lookup 语义。
- 不直接把 guided API 和 `SimulationConfig` 合并。
- 不直接改 differentiable objective 或 optimization 逻辑。
- 不为了让文件变短而移动大量代码。

原因很简单：这些区域都牵涉到论文 API、examples、paper demos、tests 和 backward compatibility。先决策，再小步改。

## 当前 API 分层

Phase 1 得出的当前结构是：

```text
public facade:
  polyfempy/api/__init__.py
  polyfempy/api/solve.py
  polyfempy/api/guided.py

core data contracts:
  polyfempy/api/config.py
  polyfempy/api/result.py

internal pipeline:
  polyfempy/api/_solve_pipeline.py
  polyfempy/api/_guided_array_mesh.py
  polyfempy/api/_runtime.py

experiment/report convenience:
  polyfempy/api/runtime.py
  polyfempy/api/report.py

support/compatibility:
  polyfempy/api/io.py
  polyfempy/api/tensor.py
  polyfempy/api/selection.py
  polyfempy/api/problems.py
```

Phase 2 要围绕这几个层做决策，而不是把所有文件混在一起重构。

## Phase 2 输出物

Phase 2 完成后，应该至少有这些可检查结果：

1. `docs/API_PUBLIC_SURFACE_DECISION.md`
   记录推荐 public API、兼容导出、内部 API。
   当前已建立，用来固定 Phase 2 的 public-surface 决策。

2. `docs/API_INTERNAL_IMPORT_AUDIT.md`
   记录当前 examples/tests/experiments 里到底 import 了哪些 API 名字。
   当前已建立，用来区分推荐 public API、internal test target 和兼容导出。

3. `docs/API_CLEANUP_PHASE2_PLAN.md`
   也就是本文件，作为执行计划。

4. 可选：`docs/API_REFACTOR_DECISIONS.md`
   如果 Phase 2 中做了很多设计选择，可以把 decision log 单独拆出去。

5. 小范围代码 cleanup PR / commit。
   但只有在 public surface 和 test matrix 清楚以后再做。

## 工作流 A：Public Surface 决策

### 要回答的问题

先回答这些问题：

1. `from polyfempy.api import ...` 推荐用户用哪些名字？
2. `polyfempy.api.guided` 推荐用户用哪些名字？
3. 哪些名字必须保留导出，但文档里不推荐？
4. 哪些名字应该变成 internal-only？
5. 如果暂时不能删，是否需要加 deprecated note？

### 当前推荐 public API 候选

最小核心入口：

```python
from polyfempy.api import solve, SimulationConfig, Result
```

guided config 入口：

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

常用 runtime/report helper：

```python
from polyfempy.api.runtime import (
    make_timestamped_workspace,
    terminal_log,
    result_output,
    solve_and_report,
)
```

这个候选集合还不是最终决定。Phase 2 要做 import audit 后再定。

### 需要审计的地方

用 `rg` 搜这些目录：

```bash
rg -n "from polyfempy\\.api|import polyfempy\\.api" examples tests experiment polyfempy
```

重点记录：

- examples 用了哪些 API；
- paper demos 用了哪些 API；
- tests 用了哪些 private helper；
- differentiable package 有没有 import `polyfempy.api` 的 internal symbol；
- 有没有外部文档推荐了不该推荐的路径。

### 输出表格格式

建议在 `docs/API_INTERNAL_IMPORT_AUDIT.md` 里写这样的表：

| Import name | Import path | 使用位置 | 推荐状态 | 处理方式 |
| --- | --- | --- | --- | --- |
| `solve` | `polyfempy.api` | examples/tests/paper demos | 推荐 public | 保留 |
| `SimulationConfig` | `polyfempy.api` | examples/tests/diff | 推荐 public | 保留 |
| `Result` | `polyfempy.api` | tests/report | 推荐 public | 保留 |
| `_solve_pipeline.normalize_cfg` | private module | pipeline tests | internal test target | 保留 internal，不写进用户文档 |
| `_process_json_config` | `solve.py` alias | 已审计，无内部业务 caller | compatibility-only | 保留旧 import path，用 `COMPATIBILITY_ALIAS_TARGETS` 记录 target |

## 工作流 B：`solve.py` 和 `_solve_pipeline.py`

### 当前判断

`solve.py` 已经是薄 wrapper：

```text
solve(...)
  -> _solve_pipeline.run_pipeline(...)
```

真正复杂逻辑在 `_solve_pipeline.py`：

```text
normalize_cfg
build_full_json
resolve_runtime_options
normalize_mesh_inputs
build_solver
configure_solver
apply_sidesets
run_solver_stage
extract_native_outputs
apply_sampled_vtu_fallback
finalize_result
```

这个结构本身是对的。Phase 2 不应该把 pipeline 又塞回 `solve.py`。

### 要搞明白什么

重点搞明白这些边界：

1. `normalize_cfg(...)`
   用户输入 `dict/path/SimulationConfig` 怎么统一。

2. `build_full_json(...)`
   什么情况走 JSON mode，什么情况走 array mode。

3. `normalize_mesh_inputs(...)`
   显式 `vertices/cells`、guided array payload、JSON geometry 谁优先。

4. `configure_solver(...)`
   Python config 什么时候变成 C++ solver state。

5. `extract_native_outputs(...)`
   backend 返回不同 shape 时怎么变成 `Result`。

6. `apply_sampled_vtu_fallback(...)`
   history 和 exported VTU 怎么补字段。

### Phase 2 可以做的低风险 cleanup

可以做：

- 把 `solve.py` 的 backward-compatible aliases 做 import audit。
- 如果 aliases 没有人用，保留为 compatibility-only，不在新代码中使用。
- 给 `_solve_pipeline.py` 顶部加更清楚的 stage overview。
- 给每个 pipeline dataclass 写清楚用途。
- 改局部变量名，但不改 behavior。

不建议做：

- 不改 `solve(...)` signature。
- 不改 `cfg` 三种输入形式。
- 不改 array mode / JSON mode 的优先级。
- 不改 result extraction strategy。
- 不改 fallback semantics。

### 测试子集

改 `solve.py` 或 `_solve_pipeline.py` 后，至少跑：

```bash
python -m pytest \
  tests/test_import_public_api.py \
  tests/test_pipeline_normalize.py \
  tests/test_pipeline_runtime_options.py \
  tests/test_pipeline_clean_json.py \
  tests/test_pipeline_extract_outputs.py \
  tests/test_pipeline_sampled_fallback.py
```

如果 backend 可用，再跑：

```bash
python -m pytest tests/test_backend_smoke.py
```

## 工作流 C：Guided API

### 当前判断

`guided.py` 是 public facade；`guided_sections.py` 是真实实现。

这点应该保留：

```text
用户 import:
  polyfempy.api.guided

真实实现:
  polyfempy.api.guided_sections
```

### 要搞明白什么

重点搞明白：

1. 哪些 section factory 是用户应该直接用的？
2. 哪些 section dataclass 只是返回类型，不需要用户手动实例化？
3. `body_section(...)` 的 mesh-file 和 array-backed 两条路径怎么走？
4. `build_config(...)` 里面哪些 helper 是 builder 内部函数？
5. `impact_template(...)` 是不是 paper/demo-specific 太强？已解决：从 package public guided API 移除，examples 直接写 `simulation_template(...)`。

### `body_section(...)` 的关键语义

必须保留：

```text
exactly one geometry source:
  mesh="..."
  或
  vertices=... + cells/faces=...
```

必须保留：

- `faces` 是 `cells` 的 alias；
- array-backed body 不允许和 file-backed body 混用；
- array-backed body 当前不支持 `transformation`、`advanced`、`n_refs`；
- array-backed payload 存在 `cfg.extras["_mesh_array_mode"]`。

### Phase 2 可以做的低风险 cleanup

可以做：

- 给 `guided_sections.py` 增加 section 分组注释。
- 把 public section factories 在文档里分成推荐/高级/内部 builder。
- 审计 `guided.py.__all__`，看是否导出了太多 type aliases。
- 写 `docs/API_PUBLIC_SURFACE_DECISION.md` 说明哪些 guided names 推荐用户用。
- 保持 `guided_sections.py` 使用相对 import，避免 implementation 依赖
  `polyfempy.api` 顶层 facade。

暂时不建议做：

- 不马上拆 `guided_sections.py`。
- 不改 `body_section(...)` signature。
- 不改 `build_config(...)` 行为。
- `impact_template(...)` 已移除；demo 不再通过 package preset 隐藏配置。

### 如果后续要拆 `guided_sections.py`

可选拆法：

```text
guided_types.py
  section dataclasses
  Literal type aliases

guided_factories.py
  problem_section(...)
  body_section(...)
  material_section(...)
  ...

guided_builders.py
  build_material(...)
  add_body_from_section(...)
  build_config(...)
```

但是 Phase 2 先不要做这个拆分。原因是 import path 和 `guided.py` export 面太大，直接拆容易引入不必要风险。

### 测试子集

改 guided API 后，至少跑：

```bash
python -m pytest tests/test_import_public_api.py
python -m pytest tests/test_config_typed_blocks.py
python -m pytest tests/test_geometry_transformations.py
```

再跑 examples 的 import/syntax 检查：

```bash
python -m py_compile \
  examples/01_forward_solve.py \
  examples/02_result_fields.py \
  examples/03_shape_gradient.py \
  examples/04_scalar_E_gradient.py \
  examples/05_parameterized_vertex_map.py \
  examples/06_dataset_one_case.py
```

## 工作流 D：`config.py`

### 当前判断

`config.py` 是最大、最乱、也最危险的文件。它同时负责：

- typed material/config blocks；
- `SimulationConfig`；
- full JSON import/export；
- minimal JSON import/export；
- old settings/problem compatibility；
- body/material/geometry id alignment；
- output/runtime options；
- solver/time/contact schema blocks。

所以 Phase 2/4 不应该直接拆 `config.py`。Phase 4 的实际动作是先加文件级说明
和分组注释，让 reviewer 能看懂结构，同时保持所有 import path 和序列化语义不变。

### 必须保留的语义

这些不能动坏：

1. `to_full_json_*` / `from_full_json_*`
   必须是 full round-trip config。

2. `to_minimal_json_*` / `from_minimal_json_*`
   必须保持 legacy subset。

3. `from_json_str(..., kind="auto")`
   是 compatibility shim，auto mode warning 要保留。

4. `from_json_file(...)`
   必须保留 `_root_path`，否则相对 mesh path 可能断。

5. `to_dict()`
   必须从 `_full_json_config` 出发，然后 overlay Python-side edits。

6. `add_body(...)`
   必须对齐 material id 和 geometry `volume_selection`。

7. `solver.nonlinear.Newton.residual_tolerance`
   这类 solver method block 语义不能被 cleanup 丢掉。

### Phase 2 要做的 audit

先把 `config.py` 的 class 分组：

```text
basic helpers:
  Quantity
  unit wrappers
  validation helpers

material blocks:
  Material
  NeoHookean
  LinearElasticity
  ...

boundary blocks:
  BoundaryConditions
  DirichletBoundary
  NeumannBoundary
  ...

geometry blocks:
  Geometry
  GeometryMesh
  GeometryPlane
  GeometryGround
  ...

solver/time/output/contact blocks:
  Solver
  Time
  Output
  Contact
  ...

main contract:
  SimulationConfig
```

Phase 4 已经把这些分组作为源码里的 section markers。之后如果真的要拆文件，
必须先确认下面的测试子集全过。

### 未来可能拆分方案

只作为候选，不在 Phase 2 直接执行：

```text
polyfempy/api/config/
  __init__.py
  simulation_config.py
  materials.py
  boundary.py
  geometry.py
  solver.py
  output.py
  contact.py
  json_io.py
```

必须满足：

```python
from polyfempy.api import SimulationConfig
from polyfempy.api.config import SimulationConfig, NeoHookean, Output
```

这些 import path 都不破。

### 测试子集

只要动 `config.py`，至少跑：

```bash
python -m pytest \
  tests/test_config_json_io.py \
  tests/test_config_typed_blocks.py \
  tests/test_config_validate.py \
  tests/test_solver_method_blocks.py \
  tests/test_pipeline_normalize.py \
  tests/test_pipeline_runtime_options.py
```

如果动了 geometry/body 相关，还要跑：

```bash
python -m pytest tests/test_geometry_transformations.py
```

## 工作流 E：`Result`

### 当前判断

`Result` 是非常重要的 output contract。不能为了看起来简单就把所有 field 混在一个 dict 里。

当前三个 namespace 是有意义的：

```text
point_data
cell_data
sampled_data
```

lookup 顺序是：

```text
point_data -> cell_data -> sampled_data
```

### 必须搞明白什么

1. `result.u` 是怎么从 `field("u")` 来的。
2. `result.stress` / `result.von_mises` 可能来自 native field，也可能来自 sampled/history fallback。
3. `sampled_data` 不一定和 native `vertices/cells` 对齐。
4. `to_meshio()` 不应该把 `sampled_data` 假装 attach 到 native mesh。
5. `history` 是 transient per-step data，不是普通 static field。

### Phase 2 可以做的 cleanup

可以做：

- 补 `Result` 文档，明确 namespace 和 fallback。
- 把 `summary/report/history_bundle` 的边界写清楚。
- 给 tests 增加一个“sampled_data 不进 meshio”的直接测试，如果还没有。

不建议做：

- 不改 `field(...)` lookup 顺序。
- 不改 `von_mises` fallback 逻辑。
- 不改 `HistoryView` shape convention。
- 不把 `sampled_data` 合并进 `point_data`。

### 测试子集

改 `result.py` / `report.py` 后，至少跑：

```bash
python -m pytest \
  tests/test_result_history.py \
  tests/test_result_meshio_roundtrip.py \
  tests/test_result_report.py \
  tests/test_result_sampled_data.py
```

如果改了 pipeline 的 result extraction，还要加：

```bash
python -m pytest tests/test_pipeline_extract_outputs.py
python -m pytest tests/test_pipeline_sampled_fallback.py
```

## 工作流 F：Runtime / Report Helpers

### 当前判断

`runtime.py` 和 `report.py` 主要服务 examples 和 paper experiments。它们有用，但不是核心 solver contract。

它们的 cleanup 优先级应该低于：

```text
public surface
solve pipeline
config semantics
Result contract
```

### 可以清的内容

可以做：

- 统一函数命名；
- 明确哪些是 examples convenience；
- 把 old aliases 标出来；
- 把 report artifact 输出路径写清楚；
- 减少和 paper-specific naming 的耦合。

不建议做：

- 不先删 `solve_and_report(...)`；
- 不先删 `emit_history_bundle(...)`；
- 不改变默认输出 artifact 名字，除非 examples/docs 一起改。

### 测试子集

```bash
python -m pytest tests/test_result_report.py
python -m pytest tests/test_pipeline_runtime_options.py
```

## 工作流 G：Differentiable Boundary

### 当前判断

differentiable 代码在 `polyfempy/differentiable/`，不是 `polyfempy/api/` 内部。但 API cleanup 不能破坏它，因为它依赖：

```text
SimulationConfig
Result-like fields
guided configs
array-backed mesh plumbing
```

### 必须保留的概念边界

三种东西不能混：

```text
objective:
  smooth-max von Mises loss

design variable:
  E
  X
  h/theta

optimization step logic:
  optimizer / line search / candidate acceptance
```

API cleanup 不应该改 objective 或 optimization logic。那些属于 differentiable/experiment 层。

### 需要检查的 import

```bash
rg -n "from polyfempy\\.api|import polyfempy\\.api" polyfempy/differentiable
```

要确认 differentiable 层有没有 import API internal symbol。如果有，先记录，不急着改。

### 测试/检查

如果 API cleanup 影响 differentiable import，至少跑：

```bash
python -m pytest tests/test_import_public_api.py
python -m py_compile \
  examples/03_shape_gradient.py \
  examples/04_scalar_E_gradient.py \
  examples/05_parameterized_vertex_map.py
```

如果 backend 可用，再跑对应 examples 的 smoke。

## 建议执行顺序

### Step 0：建立 baseline

先记录当前状态：

```bash
git status --short
python -m pytest tests/test_import_public_api.py
python -m pytest tests/test_config_json_io.py tests/test_pipeline_normalize.py
```

如果 backend 不可用，不要卡在 backend smoke；记录 skip 即可。

### Step 1：Public import audit

产出：

```text
docs/API_INTERNAL_IMPORT_AUDIT.md
```

内容：

- examples import 了什么；
- tests import 了什么；
- experiment/paper_experiment import 了什么；
- differentiable package import 了什么；
- 哪些是 public；
- 哪些是 private-but-tested；
- 哪些是 legacy alias。

### Step 2：Public surface decision

产出：

```text
docs/API_PUBLIC_SURFACE_DECISION.md
```

内容分三类：

```text
Recommended public API
Compatibility exports
Internal-only APIs
```

这一步先写文档，不改代码。

### Step 3：低风险代码整理

优先顺序：

1. `solve.py` aliases audit 和注释整理；已完成，保留 compatibility-only mapping。
2. `_solve_pipeline.py` stage 注释和局部命名；
3. `guided.py.__all__` 文档化；
4. `runtime.py/report.py` 文档和命名说明；
5. `config.py` 只做分组注释，不拆。

### Step 4：同步 docs/examples

确认这些文档都讲同一套 public API：

```text
README.md
polyfempy/README.md
docs/API_FUNCTION_MAP.md
docs/API_CLEANUP_STATUS.md
docs/TEACHER_REVIEW_GUIDE.md
examples/README.md
ARTIFACT.md
```

### Step 5：决定是否进入 Phase 3

只有满足这些条件，才进入 Phase 3 真正拆文件：

- public surface 已定；
- import audit 已完成；
- tests matrix 已明确；
- config full/minimal JSON 语义已有保护测试；
- guided API import path 有保护测试；
- examples 可以 py_compile；
- backend smoke 至少在可用环境中过一次。

## 每次 cleanup 的最小记录模板

每次改代码前，在 commit message 或临时 notes 里写清楚：

```text
改动范围:
  例如 solve.py / _solve_pipeline.py 注释和 alias cleanup

保留语义:
  solve(cfg=dict/path/SimulationConfig)
  JSON mode / array mode priority
  Result field lookup

验证:
  python -m pytest ...

风险:
  是否影响 public import
  是否影响 examples
  是否影响 differentiable
```

## Phase 2 完成标准

Phase 2 可以认为完成，当这些事情都完成：

- `docs/API_INTERNAL_IMPORT_AUDIT.md` 写好；
- `docs/API_PUBLIC_SURFACE_DECISION.md` 写好；
- `docs/API_FUNCTION_MAP.md` 和实际代码一致；
- public/import tests 通过；
- config/pipeline/result 的关键 test subset 通过；
- examples 至少 py_compile 通过；
- 明确 Phase 3 是否要拆 `guided_sections.py` 或 `config.py`。

## 最重要的原则

API cleanup 的目标不是让文件数量最少，也不是让单个文件最短。目标是让 reviewer 和未来用户能看懂：

```text
我应该 import 什么
这个函数到底干什么
它后面调用什么
哪些语义是稳定 contract
哪些只是内部实现
动完以后怎么证明没坏
```

只要这个目标没达成，就不要急着大拆代码。
