# Original JSON Class Generator Review

这个文件解释 `Original_Class-Generation-From-Json-main/` 里的 original
generator 是干什么的、它能怎么帮助当前 `polyfempy` API、哪些现有代码现在先不用动、
它本身有什么不足，以及如果以后要真的接入，需要改哪些地方。

## 一句话结论

这个 original generator 是一个 **JSON specification -> Python 配置类** 的原型。
它的目标不是求解、不是 differentiable FEM、也不是 mesh generator，而是把一份描述
配置层级和约束的 schema/specification 转成带 setter、类型检查、范围检查和
`as_dict()` 的 Python class。

对当前 repo 来说，它最适合作为：

- 配置 schema / typed config class 的生成工具原型；
- 检查 PolyFEM JSON 配置字段覆盖率的工具；
- 将来减少手写 `SimulationConfig` typed blocks 的参考。

它现在 **不适合直接接到生产 solve path**。当前 `solve(...)`、`SimulationConfig`、
guided API、differentiable runtime 和实验脚本先保持现状。

## 当前 repo 里的状态

相关目录：

```text
Original_Class-Generation-From-Json-main/
  README.md
  JsonToTreeClass.py
  Untitled-1.json
  generated_class.py
  class_test.py
  extentions_name.json
```

我检查到的当前状态：

- `JsonToTreeClass.py` 是 generator 主脚本。
- `Untitled-1.json` 是示例 specification。
- `generated_class.py` 是 generator 生成出来的 Python class 文件。
- `class_test.py` 更像早期手写/测试用草稿。
- 当前 package 代码没有 import 这个 generator 或 `generated_class.py`。
- 当前 `git status --short` 显示整个 `Original_Class-Generation-From-Json-main/`
  目录还是 untracked，因此它还不是已提交的正式 package 组成部分。

所以它现在是一个放在 repo 里的 prototype，不是 `polyfempy.api` 的正式路径。

## 它到底干啥

### 输入

输入是一份 JSON array。每个 entry 描述一个配置字段或配置节点，例如：

```json
{
  "pointer": "/geometry/gamma",
  "type": "float",
  "default": 0.5,
  "min": 0,
  "max": 1,
  "doc": "Newmark gamma"
}
```

常见字段含义：

| 字段 | 作用 |
| --- | --- |
| `pointer` | 字段在 JSON tree 里的路径，例如 `/time/dt`。 |
| `type` | 字段类型，例如 `object`, `int`, `float`, `list`, `string`, `bool`, `file`。 |
| `required` | object 节点下必须出现的 child fields。 |
| `optional` | object 节点下可选的 child fields。 |
| `default` | Python class 初始化时使用的默认值。 |
| `min`, `max` | 数值范围约束。 |
| `options` | 字符串枚举选项。 |
| `extensions` | 文件路径允许的扩展名。 |
| `type_name` | list/polymorphic entry 的类型名字，例如材料列表里的 `NeoHookean`。 |
| `doc` | 生成到 property docstring 里的说明。 |

### 内部流程

`JsonToTreeClass.py` 做了三件事：

1. 读 `Untitled-1.json`。
2. 根据 `pointer` 把所有字段挂到一个 tree 上。
3. 递归生成 Python class 字符串，最后写到 `generated_class.py`。

核心对象大概是：

```text
JsonToTreeClass
  -> 存一个 config node 的 name/type/default/doc/min/max
  -> 存 required children
  -> 存 optional children
  -> 找到 pointer 对应的 tree node
  -> 递归生成 nested classes

ClassGenerator
  -> 生成 __init__
  -> 生成 property setter
  -> 生成 enum
  -> 生成 check_required()
  -> 生成 as_dict()
```

### 输出

输出是类似这样的 Python 用法：

```python
from generated_class import Root

root = Root(string1="hello")
root.geometry.gamma = 0.5
root.geometry.mesh_sequence = "mesh.msh"
root.check_required()

data = root.as_dict()
```

它的目标是让用户不要直接手写 JSON，而是通过 Python object 构造配置，最后
`as_dict()` 变回 JSON-compatible dict。

## 它和你现在 API 的关系

当前 `polyfempy` 已经有几层配置相关代码：

```text
guided sections/templates
  -> g.build_config(...)
  -> SimulationConfig
  -> solve(cfg=...)
  -> shared solve contract
  -> backend runtime
  -> Result
```

关键文件：

- `polyfempy/api/config.py`
  - `SimulationConfig` 是 solver-facing config contract。
  - 它负责 full/minimal JSON、typed blocks、`to_dict()`、`validate()`。
- `polyfempy/api/guided.py`
  - 推荐给用户的 guided config namespace。
- `polyfempy/api/guided_sections.py`
  - compatibility facade，把 guided builders/types/config 组织起来。
- `polyfempy/api/_solve_contract.py`
  - shared solve contract，统一 config normalization、mesh source、backend settings。
- `docs/CONFIG_CONTRACT.md`
  - 记录 `SimulationConfig` 的 full/minimal JSON 语义。
- `docs/GUIDED_API.md`
  - 记录 guided API 作为 authoring layer 的语义。
- `docs/SOLVE_CONTRACT.md`
  - 记录 solve path 在进 C++ backend 前怎么统一 config/mesh/settings。

original generator 和这些代码的关系是：

| 层 | 现在是谁负责 | generator 能不能替代 |
| --- | --- | --- |
| 用户语义入口 | `solve`, `SimulationConfig`, `Result`, `polyfempy.api.guided` | 现在不能。 |
| 配置 authoring | guided sections + typed config blocks | 将来可以辅助生成一部分 typed blocks。 |
| JSON round-trip | `SimulationConfig.to_full_json_*` / `from_full_json_*` | 现在不能直接替代。 |
| minimal legacy JSON | `to_minimal_json_*` / `from_minimal_json_*` | 不应该由 generator 直接接管。 |
| mesh source 选择 | `_solve_contract.choose_mesh_source(...)` | generator 不处理。 |
| backend settings cleanup | `_solve_contract.build_canonical_solver_settings(...)` | generator 不处理。 |
| solve result | `Result` / `_solve_outputs.py` | generator 不相关。 |
| differentiable solve | `polyfempy.differentiable.*` | generator 不相关。 |

所以它更像 **配置类生产工具**，不是新的 solver architecture。

## 它可以怎么帮你

### 1. 减少手写 typed config class 的维护量

现在 `config.py`、`config_time.py`、`config_solver.py`、`config_output.py` 里有很多
手写 dataclass 和 `to_dict()`。这些代码的好处是可控、稳定、已经和当前 solve
contract 配合起来了；坏处是字段多了以后容易重复、漏字段、文档和代码不同步。

如果将来有一份完整 PolyFEM config schema，generator 可以自动生成一批 typed
classes，让这些字段来源更统一。

适合生成的候选区域：

- time block；
- solver block；
- output block；
- material parameter block；
- contact block；
- geometry JSON block 的一部分。

不适合马上生成的区域：

- `SimulationConfig` 顶层 contract；
- full/minimal JSON 兼容逻辑；
- guided API 的语义 helpers；
- mesh array mode；
- solve/differentiable runtime。

### 2. 帮你做 JSON 配置覆盖率检查

现在 reviewer 可能会问：你的 Python API 到底覆盖了多少 PolyFEM JSON 字段？

generator 如果配上一份真实 schema，可以反过来做 audit：

```text
schema fields
  vs
current SimulationConfig/to_dict fields
  vs
guided API fields
```

这样能明确说：

- 哪些字段有 typed API；
- 哪些字段只能通过 full JSON passthrough 保留；
- 哪些字段 guided API 暂时不暴露；
- 哪些字段是 internal 或 backend-only。

这对 TOMS/reviewer 很有用，因为它把“我们不是随便包了一层脚本”的证据讲清楚。

### 3. 帮你生成文档和错误提示

`Untitled-1.json` 里已经有 `doc`, `required`, `optional`, `min/max`,
`options`, `extensions`。如果 schema 整理好，可以自动生成：

- 配置字段表；
- required/optional 说明；
- enum values；
- 范围约束；
- example skeleton JSON；
- Python authoring example。

这样文档和代码来自同一个 spec，减少手写 drift。

### 4. 帮实验脚本减少 typo

你的实验里有很多 solver/time/output/material 配置。generator 生成的 class 如果质量够好，
可以让脚本更少写裸 dict 字段，例如：

```python
cfg.solver.nonlinear.Newton.residual_tolerance = 100.0
```

这种方式能减少拼错 key 的风险，尤其是 solver nested fields。

但前提是 generator 必须保留当前 JSON 语义，特别是类似：

```text
solver.nonlinear.Newton.residual_tolerance
```

这种字段不能被重命名、丢失或错误扁平化。

## 当前哪些代码先不用动

如果你的目标是现在继续推进 paper/API/实验，建议这些先不要动：

### 1. 不要把 `generated_class.py` 接到 `solve(...)`

当前 `solve(cfg=...)` 已经支持：

```text
SimulationConfig
dict
JSON path str
```

这三种输入会通过 `_solve_contract.normalize_config(...)` 统一成
`SimulationConfig`。这个路径已经有文档和测试语义。现在把 `Root` 或
`generated_class.py` 加进去，只会增加第四种 config object，短期收益很小，风险比较大。

### 2. 不要替换 `SimulationConfig`

`SimulationConfig` 现在承担的不只是字段容器，还包括：

- full JSON snapshot preservation；
- Python-side edits overlay；
- minimal legacy JSON；
- `from_json_file(...)` 的 root path 语义；
- private/public extras 区分；
- materials normalization；
- solve 前 validate。

original generator 目前只生成普通 nested class 和 `as_dict()`，还没有这些 contract。

### 3. 不要替换 guided API

guided API 不是“把 JSON 字段换成 Python 字段”这么简单。它还有更高层语义：

- `body_section(...)`；
- mesh file vs `vertices/cells` array-backed body；
- workspace 下 mesh path 处理；
- surface selection；
- body/material/load/time/contact/results 的组合；
- `g.build_config(...) -> SimulationConfig`。

generator 现在不理解这些语义，所以不应该替代 guided API。

### 4. 不要动 solve/differentiable/result runtime

generator 不负责：

- mesh source selection；
- backend settings cleanup；
- C++ solver setup；
- history/output extraction；
- VTU fallback；
- `Result` field namespace；
- differentiable material/shape gradient path。

这些路径保持现状。

### 5. 不要为了 generator 去改当前实验脚本

当前 E-diff、shape optimization、h/theta、Compute Canada dataset 脚本依赖的是现有
`SimulationConfig`、guided sections、differentiable runtime 和 experiment helpers。

在 generator 没有完整 schema、没有 adapter、没有测试之前，不建议让实验脚本依赖它。

## 这个 generator 现在的不足

### 1. 它还只是 prototype，不是可复用工具

当前 `JsonToTreeClass.py` 在 import/top-level 直接执行：

- 固定读取 `Untitled-1.json`；
- 固定写 `generated_class.py`；
- 打印 tree/debug 信息；
- 没有 `argparse`；
- 没有 `if __name__ == "__main__"`；
- 没有指定 output package/module；
- 没有单元测试。

这意味着它现在不适合作为 build step 或 package tool。

### 2. schema 太小，不是完整 PolyFEM schema

`Untitled-1.json` 只是示例。它包含少量 time、geometry、materials 字段，但远远不是
当前 solver JSON 的完整 contract。

缺少或不完整的典型区域包括：

- solver nonlinear/linear/contact 细节；
- output/result/fallback 细节；
- contact/collision/adhesion；
- units；
- body/geometry 多 body 语义；
- boundary/initial conditions；
- tests/input/constraints；
- root path 和 relative mesh path；
- JSON-only passthrough fields；
- differentiable runtime patches。

如果用这份 spec 生成生产配置类，会丢很多字段。

### 3. 命名不够稳定

当前 generator 用 `.capitalize()` 生成 class 名。这会破坏一些已有名字：

```text
NeoHookean   -> Neohookean
MooneyRivlin -> Mooneyrivlin
time_steps  -> Time_steps
```

还有 polymorphic object 会变成：

```text
Object3
Object4
Object5
```

这对用户和 reviewer 都不够清楚。生成 API 时必须保留 domain 名字。

### 4. required 检查目前不适合生产

`check_required()` 现在主要是 `print(...)`，不是 raise，也不返回 structured
error list。生产 API 里更需要：

```python
cfg.validate()
```

失败时抛 `ValueError`，并告诉用户具体路径。

此外当前 generated code 里有一些明显问题：

- 错字是 `Requiered`；
- list required 检查逻辑有反向风险；
- 有的 `remove` 代码引用 `self._list`，但实际字段叫 `self._items`；
- polymorphic required check 用 type 和字符串列表比较，语义不可靠；
- enum error message 固定写 `time_steps`，对别的 enum 不准确。

这些都说明它不能直接进入 public API。

### 5. 类型语义和 JSON/PolyFEM 语义还不一致

当前 generator 的 `type_check` 很严格，例如 `float` 必须是 Python `float`，`int`
不一定能通过。JSON 数值和 Python 数值在实际使用里经常需要更宽松的 numeric
规则。

当前 `SimulationConfig.validate()` 也没有做过强的物理范围限制，因为很多字段需要由
backend 或具体 problem 决定。generator 直接按 `min/max` 强制拦截，可能会和真实
solver 语义冲突。

### 6. 不支持当前 `SimulationConfig` 的核心 contract

它现在没有处理：

- `to_full_json_dict()`；
- `from_full_json_dict()`；
- `to_minimal_json_dict()`；
- `from_minimal_json_dict()`；
- `from_json_str(kind="auto")`；
- `_full_json_config` snapshot；
- `_root_path`；
- `_mesh_array_mode`；
- public/private extras；
- Python-side edits overlay full JSON；
- JSON cleanup before C++ backend。

这些是当前 API 的核心语义。没有补齐之前不能替换 `SimulationConfig`。

### 7. 不支持 guided API 的 higher-level 语义

generator 只能表达 tree schema，不知道：

- 一个 body 的 mesh/material/fixed surface 之间应该怎么组合；
- `mesh` 和 `vertices/cells` 必须互斥；
- `faces` 是 `cells` alias；
- array-backed mesh payload 怎么放进 `cfg.extras["_mesh_array_mode"]`；
- workspace path 怎么解析；
- 多 body 的 body_ids/boundary_ids 怎么处理。

这部分仍然应该由 guided API 手写维护。

## 如果以后要用，需要怎么改

我建议分成两个阶段，不要一上来替换当前 API。

## Phase A: 把它变成离线 audit/generation 工具

目标：不影响 `solve(...)`，只让 generator 能稳定运行、生成可检查的代码或文档。

需要改 generator：

1. 移动目录。

   建议从 repo root 移到：

   ```text
   tools/config_generator/
   ```

   或者如果只是研究材料，放到：

   ```text
   docs/prototypes/config_generator/
   ```

2. 加命令行入口。

   例如：

   ```bash
   python tools/config_generator/generate_config_classes.py \
     --schema schemas/polyfem_config_schema.json \
     --output generated/polyfem_config_generated.py
   ```

3. 拆出纯函数。

   ```python
   spec = load_spec(path)
   tree = build_tree(spec)
   source = render_module(tree)
   write_if_changed(output, source)
   ```

   不要在 import 时直接读写文件。

4. 修复命名。

   - 保留 `NeoHookean`, `MooneyRivlin`, `time_steps` 对应的稳定 class/property 名字；
   - polymorphic class 用 schema 里的 `type_name`；
   - 不用 `Object3` 这种名字。

5. 修复 validation。

   - `check_required()` 改成返回 list 或直接 raise `ValueError`；
   - 错误信息带 JSON pointer；
   - list required 逻辑改对；
   - enum error message 使用当前字段 path。

6. 生成代码走 formatter。

   - 生成后跑 `black` 或 repo 接受的 formatter；
   - 生成文件头写明 `# Generated file. Do not edit manually.`；
   - 保证 deterministic output，避免每次生成产生无意义 diff。

7. 加 generator tests。

   至少包括：

   - schema parser test；
   - generated module `py_compile`；
   - required field validation test；
   - enum/range/extension test；
   - polymorphic material list test；
   - golden output snapshot test。

这个阶段完成后，它可以帮助你做 reviewer-facing schema coverage audit，但还不碰
production solve path。

## Phase B: 只把生成类接成 config authoring layer

目标：generated classes 最多作为 authoring layer，不直接接 backend。

推荐路径：

```text
generated config classes
  -> as_dict()
  -> SimulationConfig.from_full_json_dict(...)
  -> solve(cfg=...)
```

也就是说，生成类只负责构造 dict；真正进入 solver 前仍然通过 `SimulationConfig`
和 shared solve contract。

需要改当前代码：

1. 增加 adapter，而不是让 `solve(...)` 接受 generated object。

   例如：

   ```python
   def generated_to_config(obj) -> SimulationConfig:
       return SimulationConfig.from_full_json_dict(obj.as_dict())
   ```

2. 明确 generated API 是 advanced/prototype。

   不要放进：

   ```python
   from polyfempy.api import *
   ```

   推荐先放在：

   ```text
   polyfempy.api.generated_config
   ```

   或者保持在 `tools/`，只给开发者使用。

3. 保留 `SimulationConfig` 的 full/minimal JSON 语义。

   生成类不能绕过这些方法：

   - `from_full_json_dict`
   - `to_full_json_dict`
   - `from_minimal_json_dict`
   - `to_minimal_json_dict`
   - `from_json_file`

4. 保留 guided API。

   generated classes 可以补“完整 JSON 字段覆盖”，guided API 继续负责高层用户体验。

5. 用测试证明没有改变 solve semantics。

   至少跑：

   ```bash
   python -m pytest \
     tests/test_config_json_io.py \
     tests/test_config_typed_blocks.py \
     tests/test_config_validate.py \
     tests/test_pipeline_normalize.py \
     tests/test_pipeline_runtime_options.py \
     tests/test_import_public_api.py
   ```

## 哪些现有代码将来可能被 generator 减少

这是将来的可能方向，不是现在要立刻删。

比较适合逐步生成/减少手写的地方：

- `polyfempy/api/config_time.py`
- `polyfempy/api/config_solver.py`
- `polyfempy/api/config_output.py`
- `polyfempy/api/config.py` 里部分 material/contact/geometry typed block

不建议由 generator 接管的地方：

- `SimulationConfig` 顶层 dataclass；
- `SimulationConfig.to_dict()` overlay semantics；
- JSON file root path；
- minimal JSON compatibility；
- guided API factories；
- solve contract；
- differentiable runtime；
- result extraction；
- experiment scripts。

原因很简单：前者主要是字段结构，generator 擅长；后者主要是 runtime contract 和
domain semantics，generator 不擅长。

## 推荐你现在怎么用

短期建议：

1. 先不要把 original generator 接进 `polyfempy.api`。
2. 把它当成“schema-driven config authoring”原型来读。
3. 如果要向老师/审稿人解释，可以说：

   ```text
   当前正式 API 采用手写稳定 contract：SimulationConfig + guided API + solve contract。
   original generator 代表一个可扩展方向：从 schema 自动生成 typed authoring classes。
   但它还没有覆盖当前 runtime semantics，所以暂时不进入 production path。
   ```

4. 如果你想继续推进 generator，第一步不是改 solve，而是写真实 schema 和 generator
   tests。

## 推荐下一步小任务

如果要把这件事继续往前推，最稳的小任务是：

```text
Add config generator audit mode
```

具体内容：

1. 把 prototype 移到 `tools/config_generator/`。
2. 写一个小 schema，先只覆盖 `time`, `solver`, `output`。
3. 生成一个临时 module 到 `build/generated_config/` 或 `tmp/`。
4. 用 tests 验证 generated dict 能进入 `SimulationConfig.from_full_json_dict(...)`。
5. 生成一份 coverage report，列出 schema fields 和当前 typed API fields 的差异。

这个任务风险低，因为它不改变用户 API，也不碰 solver。

## 最重要的边界

不要把 original generator 理解成“新的 SimulationConfig”。

更准确的分工应该是：

```text
generator
  -> 生成配置 authoring classes / docs / coverage audit

SimulationConfig
  -> 维护 solver-facing config contract

guided API
  -> 提供高层、好用的实验配置入口

solve contract
  -> 统一 config/mesh/backend settings

runtime/result/differentiable
  -> 负责真正求解、输出和梯度
```

这样用，generator 能帮你减少配置字段维护成本；如果直接替换现有 API，反而会破坏
你现在最重要的稳定性。
