# API 稳定性说明

这个文件记录 Phase 3 的 API stability contract。目标是让 TOMS reviewer
和未来用户能快速判断：

- 哪些 import path 是推荐且稳定的；
- 哪些 API 是 advanced / compatibility；
- 哪些模块是 internal implementation，不应该被用户依赖；
- 之后做 cleanup 时哪些边界不能破坏。

这不是说代码以后永远不能变，而是说任何改动都应该先保护这些用户可见的
contract。

## 推荐 Public API

顶层推荐入口保持很小：

```python
from polyfempy.api import solve, SimulationConfig, Result
```

含义：

| 名字 | 稳定性 | 说明 |
| --- | --- | --- |
| `solve` | stable public | forward simulation 的主入口。 |
| `SimulationConfig` | stable public | solver-facing 配置 contract。 |
| `Result` | stable public | `solve(...)` 返回的结构化输出 contract。 |

guided config 的推荐入口是 namespace import：

```python
import polyfempy.api.guided as g
```

推荐用户通过 `g.*` 构造配置：

```python
template = g.simulation_template(
    problem=g.problem_section(...),
    bodies=g.bodies_section(...),
    solver=g.solver_section(...),
    results=g.results_section(...),
)
cfg = g.build_config(template, workspace)
```

## Advanced / Compatibility API

这些 API 可以继续 import，但不作为新用户第一入口：

| 层 | 例子 | 说明 |
| --- | --- | --- |
| typed config blocks | `Material`, `Solver`, `Output`, `Time`, `Contact` | 给 advanced users 或 typed config construction 使用。 |
| runtime helpers | `make_timestamped_workspace`, `result_output`, `terminal_log` | 给 scripts/examples 配置 output/log/workspace。 |
| report helpers | `summarize_result`, `format_result_summary` | 给 reporting，不是 solve contract。 |
| compatibility names | `g.experiment_template` | 旧名字保留，新文档推荐 `g.simulation_template`。 |
| differentiable compatibility | `solve_differentiable`, diagnostic helpers | 旧名字仍可显式 import，但从 `polyfempy.differentiable` lazy-load，不进入推荐 `__all__`。 |

重要规则：

- 文档首页和 examples 不应该把 advanced helpers 写成必要入口。
- 旧脚本可以继续使用 compatibility names。
- 新 examples 应该使用推荐 public API。
- `polyfempy.api.__all__` 只代表推荐入口；advanced / compatibility 名字保留显式 import，不进入 star-import surface。

## Internal-Only API

这些模块可以被 tests 直接 import，但不应该出现在用户文档中：

| 模块 | 状态 | 说明 |
| --- | --- | --- |
| `polyfempy.api._solve_pipeline` | internal | `solve(...)` 的分阶段实现。 |
| `polyfempy.api._solve_contract` | internal | forward / differentiable 共享的 config、mesh source、backend settings contract。 |
| `polyfempy.api._guided_array_mesh` | internal | guided array-backed mesh payload builder。 |
| `polyfempy.api._runtime` | internal | Windows runtime shim implementation。 |

tests 可以直接 import internal module，例如：

```python
import polyfempy.api._solve_pipeline as _p
```

但用户代码应该从 public entry point 进入：

```python
from polyfempy.api import solve
```

## Backward Compatibility Policy

Phase 3 之后的 cleanup 应该遵守：

1. 不删除推荐 public API，除非先有 migration path。
2. `polyfempy.api.__all__` 只保留 `solve`, `SimulationConfig`, `Result`。
3. 不把 demo-only helper 包装成 core API。
4. 不改变 `SimulationConfig` 的 full/minimal JSON 语义。
5. 不改变 `Result` 的 field namespace 语义。
6. 不改变 guided `body_section(...)` 的 mesh-file / array-backed contract。

## 当前可以继续改的地方

低风险：

- 补文档；
- 增加 focused tests；
- 改 examples 的 import 风格；
- 给 internal functions 加注释；
- 把 implementation 内部 import 改成相对 import。

中风险：

- 拆 `guided_sections.py`；
- 拆 `config.py`；
- 重命名 dataclass；
- 移动 advanced compatibility names。

高风险：

- 改 `solve(...)` 参数语义；
- 改 `SimulationConfig.to_dict()` / JSON round-trip；
- 改 `Result.field(...)` 查找顺序；
- 删除 old aliases 而不提供 compatibility path。

## 对 TOMS 的意义

TOMS reviewer 需要看到这个仓库不是一组一次性实验脚本，而是一个有清楚
software contract 的 mathematical software package。这个 stability contract
说明：

- 用户入口小；
- advanced API 和 internal implementation 有边界；
- examples 使用 stable public API；
- tests 覆盖 public imports 和 internal stages；
- 后续 cleanup 有明确风险分级。
