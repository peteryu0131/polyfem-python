# SimulationConfig Contract

`SimulationConfig` 是 `polyfempy.api` 的 solver-facing config contract。
它的职责是把 Python-friendly configuration 表达成 PolyFEM backend 能消费的
settings/json 结构。

推荐 import：

```python
from polyfempy.api import SimulationConfig
```

## 三种进入 `solve(...)` 的 cfg 形式

`solve(cfg=...)` 接受三种主要形式：

| 用户输入 | 语义 | 适合场景 |
| --- | --- | --- |
| `SimulationConfig` | 已经构造好的 config object。 | guided API、typed config、程序化改配置。 |
| `dict` | full JSON-style config dict。 | 从脚本中直接生成完整配置。 |
| JSON path `str` | 从磁盘加载 JSON config。 | 复用 PolyFEM JSON 文件或 examples/configs。 |

内部会先 normalize 成 `SimulationConfig`，然后再决定走 JSON mode 还是
array-backed mode。

## Full JSON vs Minimal JSON

当前 API 明确分成两套 JSON 语义。

### Full JSON

推荐用于 round-trip：

```python
cfg = SimulationConfig.from_full_json_str(json_str)
json_str2 = cfg.to_full_json_str()
```

或 dict 形式：

```python
cfg = SimulationConfig.from_full_json_dict(d)
d2 = cfg.to_full_json_dict()
```

full JSON 目标是保留完整 solver-facing config，包括：

- geometry；
- materials；
- boundary conditions；
- solver；
- time；
- output；
- contact；
- units；
- space；
- tests/input/constraints 等 JSON-only fields。

从 JSON 文件或 full JSON dict 读进来的配置会保留在：

```python
cfg.extras["_full_json_config"]
```

`cfg.to_dict()` 会先从这个 full JSON snapshot 开始，然后 overlay 当前
Python-side edits。这是为了既保留 JSON-only 细节，又允许用户在 Python 侧修改
`cfg.solver`、`cfg.output` 等字段。

### Minimal JSON

legacy compatibility path：

```python
cfg = SimulationConfig.from_minimal_json_str(json_str)
json_str2 = cfg.to_minimal_json_str()
```

minimal JSON 只保存旧 API 的小子集：

- `pde`
- `discr_order`
- `materials`
- `boundary_conditions`
- public `extras`

它不保证保留 geometry/time/output/contact 等完整 solver config。新代码如果需要
round-trip，应使用 full JSON。

### Compatibility Shim

`from_json_str(..., kind="auto")` 是 compatibility helper：

```python
cfg = SimulationConfig.from_json_str(json_str, kind="auto")
```

更推荐显式写：

```python
SimulationConfig.from_full_json_str(json_str)
SimulationConfig.from_minimal_json_str(json_str)
```

原因：`auto` 只能根据 JSON shape 猜用户意图，不如 explicit API 清楚。

## `from_json_file(...)` 的 root path 语义

JSON 文件路径会影响 mesh 相对路径解析。

当调用：

```python
cfg = SimulationConfig.from_json_file("configs/case.json")
```

加载后的 config 会记录：

```python
cfg.extras["_root_path"]
```

并把 root path 信息保留到 full JSON 中。这样 JSON 里的相对 mesh path 在
`solve(...)` normalization 阶段仍然能正确解析。

## Guided API 如何接到 SimulationConfig

guided API 不替代 `SimulationConfig`。它只是 config authoring layer：

```text
g.problem_section(...)
g.body_section(...)
g.simulation_template(...)
  -> g.build_config(...)
  -> SimulationConfig
```

所以这两个对象语义不同：

| 对象 | 语义 |
| --- | --- |
| guided sections/template | 用户写配置时的 authoring layer。 |
| `SimulationConfig` | solver-facing config contract。 |

`g.build_config(...)` 返回以后，后续所有 solve/differentiable path 都应该围绕
`SimulationConfig` 工作。

## Array-Backed Mesh Contract

guided array-backed bodies 会把 mesh payload 存在：

```python
cfg.extras["_mesh_array_mode"]
```

这个字段是 internal bridge，连接：

```text
g.body_section(vertices=..., cells=...)
  -> g.build_config(...)
  -> cfg.extras["_mesh_array_mode"]
  -> solve(...) mesh normalization
```

用户不应该手写这个 payload，应该通过 `g.body_section(...)` 构造。

## `to_dict()` Contract

`cfg.to_dict()` 是当前 config object 的 full dictionary representation。

重要语义：

1. 如果 `extras["_full_json_config"]` 存在，先复制这个 full JSON；
2. 然后 overlay 当前 Python fields；
3. private extras，例如以下划线开头的 key，不作为 public extras 输出；
4. public extras 会写进 `"extras"`；
5. 一些 backend 兼容参数会按规则提升到 top level。

这意味着：

```python
cfg = SimulationConfig.from_json_file("case.json")
cfg.solver = new_solver
d = cfg.to_dict()
```

`d` 仍然保留原 JSON 中 Python typed fields 没覆盖的细节，同时反映新的 solver。

## 不应该破坏的语义

后续拆 `config.py` 前，必须保护：

- `from_full_json_*` / `to_full_json_*` round-trip；
- `from_minimal_json_*` / `to_minimal_json_*` legacy subset；
- `from_json_str(..., kind="auto" | "full" | "minimal")` compatibility；
- `from_json_file(...)` 的 `_root_path`；
- `to_dict()` overlay full JSON snapshot 的行为；
- guided array-backed mesh 的 `_mesh_array_mode` bridge；
- solver method blocks，例如 `solver.nonlinear.Newton.residual_tolerance`。

## 源码组织说明

`config.py` 目前仍然是单文件 contract。Phase 4 只增加 section markers，不拆
module，原因是这些 import path 已经是用户可见契约：

```python
from polyfempy.api import SimulationConfig
from polyfempy.api.config import SimulationConfig, Solver, Output, Time
```

源码里现在按职责分成 helper、material、boundary/initial condition、body/space、
problem params、`SimulationConfig`、geometry、solver、time、output、contact 等区块。
这些标记是为了降低阅读成本，不表示行为改变。

## 推荐测试

改 `SimulationConfig` 或 config JSON 语义后，至少跑：

```bash
python -m pytest \
  tests/test_config_json_io.py \
  tests/test_config_typed_blocks.py \
  tests/test_config_validate.py \
  tests/test_solver_method_blocks.py \
  tests/test_pipeline_normalize.py \
  tests/test_pipeline_runtime_options.py
```

如果改的是 guided config 到 `SimulationConfig` 的转换，还要跑：

```bash
python -m pytest tests/test_import_public_api.py tests/test_geometry_transformations.py
```
