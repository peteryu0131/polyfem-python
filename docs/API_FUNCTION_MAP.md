# API 函数地图

这个文件是 `polyfempy/api` 的 Phase 1 代码地图。目的不是马上重构，而是先把当前 API 文件夹看清楚：

- 哪些东西是 public API；
- 每个文件负责什么；
- 每个入口函数后面调用了什么；
- 哪些是核心逻辑、内部 pipeline、兼容层、report/helper。

这是一份当前实现地图，不是最终设计文档。

## Phase 1 总结

现在 `polyfempy/api` 主要有四条主流程。

第一条：guided sections 生成配置，然后 forward solve：

```text
guided sections
  -> build_config(...)
  -> SimulationConfig
  -> solve(cfg=...)
  -> Result
```

第二条：用户直接给 JSON / dict / `SimulationConfig` 进 `solve(...)`：

```text
JSON path / dict / SimulationConfig
  -> solve(cfg=...)
  -> _solve_pipeline.run_pipeline(...)
  -> C++ Solver / State
  -> Result
```

第三条：`Result` 作为 forward solve 的稳定输出对象：

```text
Result
  -> field accessors
  -> history accessors
  -> report/runtime helpers
```

第四条：differentiable API 复用 `SimulationConfig`：

```text
SimulationConfig
  -> polyfempy.differentiable
  -> prepare_optimization_problem(...)
  -> loss.backward()
```

对 cleanup 来说，最重要的是先分清这些层：

- `solve.py`、`guided.py`、`__init__.py`：public facade，用户主要从这里 import。
- `_solve_pipeline.py`、`_guided_array_mesh.py`、`_runtime.py`：内部实现，不应该当成普通用户入口。
- `config.py`、`result.py`：核心 data contract。
- `runtime.py`、`report.py`：实验和报告 convenience layer。
- `selection.py`、`problems.py`、`io.py`、`tensor.py`：support / compatibility helper。

## 文件职责

| 文件 | 层级 | 是否 public | 主要职责 |
| --- | --- | --- | --- |
| `polyfempy/api/__init__.py` | package facade | 是 | 暴露 top-level API；import 时处理 Windows runtime shim。 |
| `polyfempy/api/solve.py` | forward solve facade | 是 | 提供薄的 `solve(...)` 入口；`__all__` 只推荐 `solve`。 |
| `polyfempy/api/_solve_compat.py` | solve compatibility | 内部 | 给旧的 `polyfempy.api.solve` helper imports 安装 compatibility-only aliases。 |
| `polyfempy/api/_solve_pipeline.py` | forward solve orchestration | 内部 | 按顺序调用 contract、backend、outputs，并构造最终 `Result`。 |
| `polyfempy/api/_solve_contract.py` | forward solve contract | 内部 | 统一 cfg 输入、mesh source selection、canonical backend settings。 |
| `polyfempy/api/_solve_backend.py` | backend adapter | 内部 | 创建 C++ solver、应用 settings、attach mesh、assemble/solve。 |
| `polyfempy/api/_solve_outputs.py` | output adapter | 内部 | native output extraction、history、sampled fallback、result finalize。 |
| `polyfempy/api/config.py` | config data model | 是 | typed config blocks；`SimulationConfig`；full/minimal JSON 语义。 |
| `polyfempy/api/guided.py` | guided API facade | 是 | guided section 的稳定 import path；`__all__` 只推荐 factory/builder functions。 |
| `polyfempy/api/guided_sections.py` | guided compatibility facade | 内部 | re-export guided factories/types/config helpers for older internal imports。 |
| `polyfempy/api/guided_builders.py` | guided section factories | 内部 | 创建 user-authored section objects，不创建 `SimulationConfig`。 |
| `polyfempy/api/guided_types.py` | guided section schema | 内部 | dataclasses 和 Literal type aliases。 |
| `polyfempy/api/_guided_config.py` | guided config translation | 内部 | `build_config(...)` 和 template -> `SimulationConfig` helpers。 |
| `polyfempy/api/_guided_array_mesh.py` | guided array mesh internals | 内部 | 检查并合并 `vertices/cells` body，生成 array mesh payload。 |
| `polyfempy/api/result.py` | result data model | 是 | `Result`、`HistoryView`、field namespace、meshio conversion、report hooks。 |
| `polyfempy/api/runtime.py` | runtime/report convenience | 是 | workspace、logging/output setup、solve-and-report helper。 |
| `polyfempy/api/report.py` | reporting helpers | 是 | `Result` 和 transient history 的结构化 summary。 |
| `polyfempy/api/io.py` | mesh I/O helper | 是 | 用 `meshio` 读取 mesh，转成 normalized arrays。 |
| `polyfempy/api/tensor.py` | tensor conversion helper | support | NumPy/Torch/JAX detection 和转换，主要服务 array-mode solve/result。 |
| `polyfempy/api/selection.py` | geometric selection helper | 是 | 用 sphere/box/plane 生成 body/sideset selection payload。 |
| `polyfempy/api/problems.py` | predefined problem compatibility | 是 | `SimulationConfig.to_settings()` 还会用到的小 problem class。 |
| `polyfempy/api/_runtime.py` | Windows runtime shim | 内部/导出 helper | Windows 下 UTF-8/OpenMP 的可选 runtime tweak。 |

## Public Surface

顶层 `from polyfempy.api import ...` 现在暴露这些组。

| 组 | 名字 | 意义 |
| --- | --- | --- |
| Core | `solve`, `SimulationConfig`, `Result` | 推荐的最小 public surface。 |
| I/O | `Selection`, `Mesh`, `read_mesh` | mesh、selection helper。 |
| Reporting | `summarize_result`, `format_result_summary`, `summarize_history_bundle`, `format_history_bundle_txt`, `write_history_bundle_txt` | 给 `Result` 做人读/脚本读的 summary。 |
| Runtime | `make_timestamped_workspace`, `terminal_log`, `result_output`, `format_history_summary`, `write_history_artifacts`, `report_history_bundle`, `emit_history_bundle`, `solve_and_report` | 实验脚本常用 convenience utilities。 |
| Config | `Quantity`、material classes、geometry classes、solver/time/output/contact blocks、problem param blocks | 较底层的 typed config surface 和兼容导出。 |
| Runtime shim | `configure_windows_runtime` | 显式 Windows runtime 配置。 |

`polyfempy.api.guided` 是另一套 public facade。它通过 `guided_sections.py`
兼容 facade 暴露 section factories、dataclasses 和 type aliases，但 `g.__all__`
只包含推荐 factory/builder functions；types 和 compatibility names 仍可显式访问。

## 主流程 1：`solve(cfg=...)`

位置：`polyfempy/api/solve.py`

`solve.py` 现在是很薄的一层：

```text
solve(...)
  -> _solve_pipeline.run_pipeline(...)
```

完整 pipeline：

```text
run_pipeline(...)
  -> prepare_canonical_solve_input(...)
       -> normalize_config(...)
       -> choose_mesh_source(...)
       -> build_canonical_solver_settings(...)
  -> resolve_runtime_options(cfg, full_json, sampled_vtu_fallback)
  -> _inputs_from_mesh_source(canonical.mesh_source)
  -> build_solver()
  -> configure_solver(solver, cfg, full_json, inputs)
  -> apply_sidesets(solver, sidesets_func, ctx)
  -> run_solver_stage(solver, full_json)
  -> extract_native_outputs(ret, solver, inputs)
  -> _collect_solver_history(solver, full_json)
  -> Result(...)
  -> apply_sampled_vtu_fallback(...)
  -> finalize_result(result, runtime)
```

### `normalize_cfg(cfg)`

位置：`_solve_pipeline.py`

接受三种 public cfg 形式：

- `dict`：调用 `SimulationConfig.from_json_dict(...)`。
- JSON path `str`：调用 `SimulationConfig.from_json_file(...)`。
- `SimulationConfig`：原样通过。

拒绝：

- `None`；
- 其他类型。

这里是三种用户输入统一成内部 `SimulationConfig` 的地方。

### `build_full_json(cfg)`

位置：`_solve_pipeline.py`

作用：

- 如果能走 JSON mode，返回 full JSON dict；
- 如果应该走 array mode，返回 `None`。

来源优先级：

1. `cfg.extras["_full_json_config"]`，也就是从 JSON 文件/JSON dict 加载进来的原始 full config；
2. `cfg.to_dict()`，前提是里面有 `geometry` block。

重要语义：

- 从 JSON 读进来以后，用户在 Python 侧改 `cfg`，这些改动会 overlay 原来的 full JSON。
- guided array-backed body 会跳过 full JSON mode。
- `_root_path` 会提升成 `root_path`，让相对 mesh path 能继续解析。

### `resolve_runtime_options(...)`

位置：`_solve_pipeline.py`

读取 runtime-only 选项，来源是：

1. `cfg.output.runtime_options()`；
2. `full_json["output"]`；
3. `solve(..., sampled_vtu_fallback=...)` 显式参数。

输出 `RuntimeOptions`：

```text
requested_fields
strict
fallback_mode
temp_storage
keep_temp_files
```

这些控制 result extraction / fallback / strict check，不是物理模拟配置。

### `normalize_mesh_inputs(...)`

位置：`_solve_pipeline.py`

决定执行模式：

- 用户显式传 `vertices/cells`：array mode；
- `cfg.extras["_mesh_array_mode"]` 里有 guided array mesh：array mode；
- `full_json` 里有 `geometry` 且没有显式 arrays：JSON mode；
- 其他情况报错。

array mode 会 normalize：

```text
vertices -> NumPy array
cells -> int32 NumPy array
body_ids -> optional int32 array
boundary_ids -> optional int32 array
original backend -> "numpy" / "torch" / "jax"
```

### `configure_solver(...)`

位置：`_solve_pipeline.py`

分两支。

JSON mode：

```text
process_json_config(...)
  -> clean_json_for_cpp(...)
  -> solver.set_settings(...)
  -> solver.load_mesh_from_settings()
```

Array mode：

```text
cfg.to_dict()
  -> 如果没有 geometry，插入 placeholder geometry
  -> promote materials to list
  -> solver.set_settings(...)
  -> set_mesh / set_mesh_data / load_mesh_from_points
  -> 如果 backend 支持，设置 body_ids / boundary_ids
  -> retouch boundary conditions
```

这里是 Python config 变成 C++ solver state 的边界。

### `run_solver_stage(...)`

位置：`_solve_pipeline.py`

如果 backend 有这些方法，就按顺序调用：

```text
solver.build_basis()
solver.assemble()
solver.solve(log_level=...)
```

如果 backend 暴露的是 `run(...)` 而不是 `solve(...)`，就退到 `run(...)`。

### `extract_native_outputs(...)`

位置：`_solve_pipeline.py`

根据 solver return shape 选择输出提取策略：

1. solver 返回 `_result_bundle` dict，里面已经有 mesh/fields；
2. solver 返回 tuple/list，通常是 `(u, p)`；
3. solver 有 `get_sampled_solution()`；
4. solver 有 direct getters，比如 `get_solution`、`get_displacement`、`get_u`。

额外 field probes：

```text
stress:   get_stress / get_cauchy_stress / stress
strain:   get_strain / strain
energy:   get_energy / energy / total_energy
pressure: get_pressure / pressure
velocity: get_velocity / velocity
stats:    get_stats / stats / get_log
```

### `apply_sampled_vtu_fallback(...)`

位置：`_solve_pipeline.py`

当前 fallback 数据源只有两个：

1. `solver.solution_frames` 生成的 in-memory `result.history`；
2. 用户已经导出的 `impact_step_*.vtu` 文件。

旧的临时 `export_vtu()` 路径已经不在当前 flow 里。

### `finalize_result(...)`

位置：`_solve_pipeline.py`

做 requested-field bookkeeping：

- 把 `requested_fields` 写进 `result.meta`；
- 记录 `missing_requested_fields`；
- 只有 `strict=True` 且字段缺失时才 raise。

## 主流程 2：Guided Config

public import path：

```text
import polyfempy.api.guided as g
```

主链路：

```text
problem_section / body_section / material_section / ...
  -> simulation_template(...)
  -> build_config(template, workspace)
  -> SimulationConfig
```

### 常用 Section Factories

| 函数 | 返回 | 意义 |
| --- | --- | --- |
| `problem_section(...)` | `ProblemSection` | PDE 和可选 predefined problem type。 |
| `units_section(...)` | `UnitsSection` | 单位系统。 |
| `material_section(...)` | `MaterialSection` | material model，可用 `young_poisson` 或 `lame` mode。 |
| `fixed_surface_section(...)` | `FixedSurfaceSection` | position/sphere/box/plane fixed region。 |
| `body_section(...)` | `BodySection` | 一个 mesh-file 或 array-backed body，加 material/BC/initial conditions。 |
| `bodies_section(...)` | `list[BodySection]` | 打包多个 body section。 |
| `loads_section(...)` | `LoadsSection` | RHS/body force。 |
| `space_section(...)` | `SpaceSection` | discretization/basis choices。 |
| `time_section(...)` | `TimeSection` | 时间区间和 integrator。 |
| `solver_section(...)` | `SolverSection` | linear/nonlinear/contact solver choices。 |
| `contact_section(...)` | `ContactSection` | contact model/options。 |
| `output_section(...)` | `OutputSection` | output files、ParaView、fallback、requested result fields。 |
| `results_section(...)` | `ResultsSection` | guided output 中的 result-field request。 |
| `simulation_template(...)` | `SimulationTemplate` | 把所有 sections 组合起来；新文档推荐名。 |
| `experiment_template(...)` | `ExperimentTemplate` | compatibility alias，旧脚本继续可用，但不在 `g.__all__`。 |

### `body_section(...)`

位置：`guided_builders.py`

geometry contract：必须且只能给一种 geometry source。

文件 mesh：

```text
mesh="..."
```

或者 array-backed mesh：

```text
vertices=..., cells=...
vertices=..., faces=...   # faces 是 cells 的 alias
```

重要检查：

- `cells` 和 `faces` 不能同时传；
- 传 `faces` 时内部转成 `cells`；
- array-backed body 必须同时有 `vertices` 和 `cells/faces`；
- array-backed body 目前只支持 `extract="volume"`；
- array-backed body 暂时不支持 `transformation`、`advanced`、`n_refs`。

输出是 `BodySection`。它还没有创建 solver body；真正创建发生在 `build_config(...)`。

### `build_config(template, workspace)`

位置：`_guided_config.py`

调用链：

```text
build_config(...)
  -> SimulationConfig()
  -> add_body_from_section(...) for each BodySection
       -> build_material(...)
       -> cfg.add_body(...)
       -> build_surface_selection(...) for fixed surfaces
       -> body.fix_surface(...)
       -> set initial velocity/solution/acceleration
  -> build_guided_array_mesh_payload(...) when array-backed bodies exist
  -> build_geometry_extra(...)
  -> cfg.set_rhs(...)
  -> build_solver(...)
  -> build_time(...)
  -> build_space(...)
  -> build_output(...)
  -> build_contact(...)
  -> SimulationConfig
```

重要语义：

- `build_config(...)` 至少需要一个 body；
- 当前不允许在同一个 guided template 里混用 mesh-file bodies 和 array-backed bodies；
- 如果有 array-backed bodies，会把合并后的 mesh 存到 `cfg.extras["_mesh_array_mode"]`；
- 它不运行 PolyFEM；
- 它不构造 loss；
- 它不处理 optimization variable。

### `build_guided_array_mesh_payload(...)`

位置：`_guided_array_mesh.py`

作用：

- 检查每个 array-backed body；
- 把多个 body 的 vertices/cells merge 成一个 global mesh；
- 给 cells 加 vertex offset；
- 用 `body.volume_id` 给每个 cell 写 `body_ids`。

输出：

```text
{
  "vertices": merged_vertices,
  "cells": merged_cells,
  "body_ids": merged_body_ids,
}
```

后续由 `_solve_pipeline.normalize_mesh_inputs(...)` 读取。

## 主流程 3：`SimulationConfig`

位置：`config.py`

`SimulationConfig` 是 solver-facing configuration object。它可以直接构造，也可以从 JSON 读入，也可以由 guided sections 生成。

`config.py` 暂时不拆文件。Phase 4 只加源码 section markers，把它分成：

- internal normalization / compatibility helpers；
- units and material config classes；
- boundary and initial-condition classes；
- body / constraint / space / input helpers；
- problem parameter compatibility classes；
- root `SimulationConfig` contract；
- geometry / solver / time / output / contact blocks。

这样 reviewer 可以看懂职责边界，但旧 import path 保持不变。

### 重要方法

| 方法 | 作用 |
| --- | --- |
| `to_dict()` | 返回当前 full config dict；优先从 `_full_json_config` 开始，然后 overlay Python-side edits。 |
| `to_full_json_dict()` / `to_full_json_str()` | 显式 full round-trip export。 |
| `to_minimal_json_dict()` / `to_minimal_json_str()` | legacy minimal subset export。 |
| `to_json_str()` | deprecated alias，等价于 minimal export。 |
| `from_json_dict(...)` | full PolyFEM JSON import；原始 JSON 存到 `extras["_full_json_config"]`。 |
| `from_json_file(...)` | 从磁盘读 full JSON；额外存 `_root_path`。 |
| `from_full_json_dict(...)` / `from_full_json_str(...)` | 显式 full import。 |
| `from_minimal_json_dict(...)` / `from_minimal_json_str(...)` | 显式 legacy minimal import。 |
| `from_json_str(..., kind="auto")` | compatibility shim；auto mode 会 warning。 |
| `validate()` | 轻量 type/shape 检查，不是物理有效性检查。 |
| `add_body(...)` | 自动对齐 material id 和 geometry `volume_selection`。 |
| `set_material(...)`, `set_dirichlet_boundary(...)`, `set_neumann_boundary(...)`, `set_rhs(...)` | convenience mutator。 |
| `to_settings()` | 旧 backend settings path / compatibility path。 |

### Full JSON 和 Minimal JSON 的区别

Full JSON：

```text
to_full_json_dict / to_full_json_str
from_full_json_dict / from_full_json_str
```

保留这些内容：

```text
geometry
time
output
solver
contact
units
space
tests
input
initial_conditions
constraints
problem_type
problem_params
```

Minimal JSON：

```text
to_minimal_json_dict / to_minimal_json_str
from_minimal_json_dict / from_minimal_json_str
```

只保留：

```text
pde
discr_order
materials
boundary_conditions
public extras
```

cleanup 时不要随便合并这两套语义。full/minimal 的区别是当前 API compatibility 的关键。

## 主流程 4：`Result`

位置：`result.py`

`Result` 是 `solve(...)` 返回的稳定输出对象。

### Field Namespace

field lookup 优先级：

```text
point_data -> cell_data -> sampled_data
```

含义：

- `point_data`：native mesh 的 per-vertex fields；
- `cell_data`：native mesh 的 per-cell fields；
- `sampled_data`：history/fallback/probe mesh 上的数据，不一定和 native `vertices/cells` 对齐。

常用 accessors：

```text
result.vertices
result.cells
result.V
result.u
result.p
result.stress
result.strain
result.von_mises
result.body_ids
result.history
```

### 重要方法

| 方法 | 作用 |
| --- | --- |
| `field(name)` | 兼容 lookup：在 point/cell/sampled namespace 里找 field，缺失时返回 `None`。 |
| `has_field(name, namespace=None)` | 检查 field 是否存在，可限制到某个 namespace。 |
| `require_field(name, namespace=None)` | strict lookup：缺失时抛 `KeyError` 并列出 available fields。 |
| `field_info(name, namespace=None)` | 返回 namespace/source/shape/dtype/derived metadata，适合写 JSON summary。 |
| `set_field(name, value)` | 存 native mesh-aligned point/cell data。 |
| `set_sampled_field(name, value)` | 存 sampled/probe data。 |
| `field_by_body(name)` | 用 `body_ids` 把 field 按 body 分开。 |
| `get_von_mises_numpy()` | 复用已有 `von_mises`/`von_mises_avg`，或者从 stress 计算。 |
| `get_percentile_from_von_mises(...)` | 计算 von Mises percentile。 |
| `to_torch(...)` | 把 fields 和可选 mesh 转成 Torch。 |
| `to_backend(...)` | 转回原 backend。 |
| `to_meshio()` / `from_meshio(...)` / `read(...)` / `write(...)` | meshio 互操作。 |
| `summary()` | shape-level summary。 |
| `report()` / `format_summary()` | 调到 `report.py`。 |
| `history_bundle()` / `format_history_bundle_txt()` / `write_history_bundle_txt()` | 调到 history reporting helpers。 |

### `HistoryView`

`HistoryView` 包装 transient per-step frames，来源可以是 `solver.solution_frames` 或导出的 VTU。

重要字段：

```text
history.u
history.vm
history.vm_avg
history.stress
history.points
history.connectivity
history.body_ids
history.times
```

shape convention：

```text
u:      (n_steps, n_sampled, dim)
vm:     (n_steps, n_sampled)
stress: (n_steps, n_sampled, tensor_width)
```

## Runtime 和 Reporting Layer

这些文件对 examples 和 paper experiments 很有用，但不是核心 solver contract。

### `runtime.py`

| 函数 | 调用 | 作用 |
| --- | --- | --- |
| `make_timestamped_workspace(...)` | filesystem only | 创建 run workspace。 |
| `terminal_log(cfg, ...)` | `_ensure_output(...)`, `Output.set_log(...)` | 配置 terminal/file logging。 |
| `result_output(cfg, ...)` | `_ensure_output(...)`, output config methods | 配置 JSON/ParaView/time-sequence outputs。 |
| `format_history_summary(...)` | formatting only | 渲染 history bundle。 |
| `write_history_artifacts(...)` | `summarize_history_bundle(...)`, CSV/JSON writers | 写 history summary artifacts。 |
| `report_history_bundle(...)` | `write_history_artifacts(...)` | 写并可选打印 report artifacts。 |
| `emit_history_bundle(...)` | `report_history_bundle(...)` | backward-compatible alias。 |
| `solve_with_timing(cfg=...)` | `solve(cfg=cfg)` | 跑 solve 并记录时间。 |
| `solve_and_report(cfg=..., workspace=...)` | `solve_with_timing(...)`, `report_history_bundle(...)` | examples 用 convenience wrapper。 |

### `report.py`

| 函数 | 作用 |
| --- | --- |
| `summarize_result(...)` | field availability、shape、origin、可选 history/body stats。 |
| `format_result_summary(...)` | 把 `summarize_result(...)` 渲染成人读文本。 |
| `summarize_history_bundle(...)` | transient per-step metrics 和 per-body summaries。 |
| `format_history_bundle_txt(...)` | history 的 TSV-style text bundle。 |
| `write_history_bundle_txt(...)` | 写 TSV bundle。 |

## Support 和 Compatibility 文件

### `io.py`

```text
read_mesh(path)
  -> meshio.read(path)
  -> Mesh(vertices, cells, point_data, cell_data)
```

这是外部 mesh 的 convenience path，需要 `meshio`。

### `tensor.py`

被 array-mode solve 和 result backend conversion 使用。

重要函数：

```text
detect_backend(x)
as_numpy(x, dtype=None)
from_numpy(arr, backend)
to_backend(arr, backend)
```

注意：Torch tensor 会先 `detach()` 再转 NumPy。这个文件不是 differentiable bridge，只是 API 层的数据转换工具。

### `selection.py`

`Selection` 生成 plain dict / JSON payload：

```text
select_body_with_sphere(...)
select_body_with_box(...)
select_body_with_axis_plane(...)
select_body_with_plane(...)
select_sideset_with_sphere(...)
select_sideset_with_box(...)
select_sideset_with_axis_plane(...)
select_sideset_with_plane(...)
to_dict()
to_json_str()
```

用于不想手写 sideset/body id 的场景。

### `problems.py`

提供 `SimulationConfig.to_settings()` 还会用到的 predefined problem classes：

```text
Problem
Franke
GenericScalar
Gravity
Torsion / TorsionElastic
GenericTensor
Flow
DrivenCavity
FlowWithObstacle
get_problem_class(name)
available_problem_names()
```

主要是旧 settings/problem construction 的 compatibility support。

### 已移除：`batch.py`

`batch_solve(jobs)` 只是顺序循环调用 `solve(...)`，repo 内没有真实 caller。
Phase 2 已从 public facade 中移除。用户脚本需要批量运行时，可以直接写清楚的
Python loop。

### `_runtime.py`

内部 Windows runtime shim：

```text
should_auto_configure_windows()
configure_windows_runtime(force=False)
```

`__init__.py` import 时可能会自动调用。设置 `POLYFEMPY_SKIP_WINDOWS_AUTOCONFIG=1` 可以跳过。

## Differentiable API 怎么接上来

虽然 differentiable 代码不在 `polyfempy/api` 文件夹里，但它依赖这里的两个 contract：

```text
SimulationConfig
Result / DifferentiableResult-style fields
```

关键链路：

```text
SimulationConfig
  -> prepare_optimization_problem(cfg=..., kind=...)
  -> problem.solve()
  -> make_von_mises_loss(...)
  -> loss.backward()
```

三种 design variable 要分开理解：

| variant | design variable | gradient | 意义 |
| --- | --- | --- | --- |
| material | lattice Young's modulus `E` | `dL/dE` | 材料敏感度。 |
| free-form shape | mesh vertices `X` | `dL/dX` | 直接移动 mesh vertex 的形状敏感度。 |
| parameterized shape | `h`, `theta_deg` | `dL/dh`, `dL/dtheta_deg` | 通过 `X(h, theta_deg)` vertex map 做 chain rule。 |

这部分 cleanup 时要注意：API 文件夹不应该知道具体优化逻辑，但必须保留 differentiable 层需要的 config/result 语义。

## Paper Demo 相关 helper

`experiment/paper_experiment/common.py` 不是核心 API。它主要放 paper demo 内部复用的小工具：

```text
new_workspace(...)
configure_output(...)
write_summary(...)
scalar_from_snapshot(...)
design_step_label(...)
```

这些函数不 solve、不 differentiate、不 build loss、不 optimize。

`experiment/paper_experiment/08_h_theta_shape_optimization.py` 里的 `h_theta_vertex_map(...)` 是 demo-specific geometry formula：

```text
h, theta_deg
  -> h_theta_vertex_map(...)
  -> vertices
```

它不改变 connectivity，不 call PolyFEM，不计算 loss，不更新 `h/theta`。它只是显式写出这个 paper demo 的参数化几何。

## 测试覆盖地图

| 区域 | 测试 |
| --- | --- |
| Public imports | `tests/test_import_public_api.py` |
| Config full/minimal JSON semantics | `tests/test_config_json_io.py`, `tests/test_config_typed_blocks.py`, `tests/test_config_validate.py`, `tests/test_solver_method_blocks.py` |
| Solve cfg/mesh normalization | `tests/test_pipeline_normalize.py` |
| Runtime output options | `tests/test_pipeline_runtime_options.py` |
| JSON cleanup for C++ | `tests/test_pipeline_clean_json.py` |
| Native output extraction | `tests/test_pipeline_extract_outputs.py` |
| Sampled/history fallback | `tests/test_pipeline_sampled_fallback.py` |
| Result field/meshio/history behavior | `tests/test_result_history.py`, `tests/test_result_meshio_roundtrip.py`, `tests/test_result_report.py`, `tests/test_result_sampled_data.py` |
| Problem helpers | `tests/test_api_problems.py` |
| Windows runtime shim | `tests/test_runtime_windows_shim.py` |
| Backend smoke | `tests/test_backend_smoke.py` |

## Cleanup 结论

1. `solve.py` 应该保持薄。它现在已经只是 `_solve_pipeline.run_pipeline(...)` 的 public wrapper。
2. `_solve_pipeline.py` 应该保持 internal。它已经有清楚 stage 和测试；cleanup 应该主要改命名/结构，不要把 public 语义搬进去。
3. `guided.py` 应该保留为 guided API 的 public facade。`guided_sections.py` 现在是 compatibility facade；factory functions 在 `guided_builders.py`，template translation 在 `_guided_config.py`。
4. 不要把 guided authoring layer 和 `config.py` 合并。guided sections 是 authoring layer，`SimulationConfig` 是 solver-facing contract。
5. `config.py` 是最高风险 cleanup 对象，因为它同时承载 full JSON、minimal JSON、typed blocks、旧 problem factories、backend settings conversion。
6. `Result` 是稳定 contract。`point_data/cell_data/sampled_data` 的区别不能随便拍平。
7. `runtime.py` / `report.py` 可以等核心 map 稳定以后再清，因为它们主要是实验 ergonomics。
8. differentiable code 在 `polyfempy/api` 外面，但依赖 `SimulationConfig` 和 `Result`，所以 API cleanup 必须保留这两个 contract。

## Phase 2 前要回答的问题

1. 顶层 `polyfempy.api.__all__` 到底应该保留哪些名字？
   已决定：只保留 `solve`, `SimulationConfig`, `Result`。typed config、
   runtime/reporting helper 仍然可以显式 import，但不进入 star-import surface。
2. guided implementation 是否还要继续拆？
   已决定：先保留当前清晰边界；`guided_types.py` 放 dataclasses/types，
   `guided_builders.py` 放 factories，`_guided_config.py` 放 template translation。
3. `solve.py` 里的 backward-compatible aliases 还有没有内部 caller 在用？
   已审计：当前 repo 内业务代码没有 caller；tests 直接使用 `_solve_pipeline`
   helper。Phase 3 保留这些 alias 作为 compatibility-only，并用
   `COMPATIBILITY_ALIAS_TARGETS` 记录每个 alias 的 current pipeline target。
4. `config.py` 能不能安全拆分，并且不破坏现有 import paths？
   当前决定：周一前不拆。Phase 4 只加结构标记和文档，后续真拆必须保留
   `from polyfempy.api.config import SimulationConfig, Solver, Output, Time`。
5. 每一个 cleanup slice 后必须跑哪些 test subset？

建议的下一步不是马上大改，而是先做 public surface decision：确定哪些是推荐入口，哪些只是兼容导出。之后再动 `solve.py/_solve_pipeline.py` 这种低风险结构层。
