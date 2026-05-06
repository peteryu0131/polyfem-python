# TOMS Reviewer Checklist

这个文件不是投稿说明的替代品，而是给自己用的 reviewer 模拟清单。目标是让
`polyfempy` 看起来像一个清楚、可复用、可验证的 mathematical software
package，而不是一组一次性实验脚本。

参考来源：

- TOMS author information: https://www.cs.kent.ac.uk/projects/toms/Authors.html
- ACM software/data artifacts: https://www.acm.org/publications/artifacts

截至 2026-05-06，TOMS 官方作者页强调：TOMS 关注 mathematical software 的
development、evaluation 和 use；论文会看 originality、relevance 和
presentation；如果是 algorithm/software submission，软件包本身也会被 referee
评价。ACM artifact 页面也强调软件和数据 artifact 应该方便复跑、复用和长期保存。

所以 Phase 3 的重点不是 example 数量，而是 reviewer 能不能快速回答：

```text
这个库怎么用？
哪些 API 是正式入口？
结果怎么拿？
配置怎么复现？
测试保护了什么？
paper scripts 和 library API 的边界在哪里？
```

## 1. Public API Surface

Reviewer 应该能在 1 分钟内看到推荐入口：

```python
from polyfempy.api import solve, SimulationConfig, Result
import polyfempy.api.guided as g
```

检查项：

| 问题 | 当前目标 | 文件 |
| --- | --- | --- |
| 顶层入口是否小？ | 只推荐 `solve`, `SimulationConfig`, `Result`。 | `docs/API_STABILITY.md` |
| guided API 是否有 namespace？ | 新代码推荐 `import polyfempy.api.guided as g`。 | `docs/GUIDED_API.md` |
| compatibility API 是否和 stable API 分开？ | 旧名字可保留，但不作为 README 第一入口。 | `docs/API_PUBLIC_SURFACE_DECISION.md` |
| internal module 是否被隐藏？ | `_solve_pipeline`, `_guided_array_mesh`, `_runtime` 不进用户文档。 | `docs/API_STABILITY.md` |

Reviewer-friendly 的最小例子应该是：

```python
cfg = SimulationConfig.from_json_file("examples/configs/contact_impact.json")
result = solve(cfg=cfg)
print(result.u.shape)
print(result.von_mises)
```

## 2. Configuration Contract

Reviewer 需要看懂 `SimulationConfig` 是 solver-facing config，不是一个模糊的
临时 dict。

检查项：

| 问题 | 应该满足 |
| --- | --- |
| JSON file、dict、`SimulationConfig` 三种输入是否区分清楚？ | `docs/CONFIG_CONTRACT.md` 有明确表格。 |
| full JSON 和 minimal JSON 是否分开？ | round-trip 用 `from_full_json_*` / `to_full_json_*`。 |
| relative mesh path 是否可复现？ | `from_json_file(...)` 记录 `_root_path`。 |
| guided API 是否只是 authoring layer？ | `g.build_config(...) -> SimulationConfig`。 |
| array-backed mesh 是否有明确限制？ | 用户用 `g.body_section(...)`，不手写 `_mesh_array_mode`。 |

Phase 3 不建议现在拆 `config.py`。先让 contract 清楚，再用测试保护以后拆分。

## 3. Result Contract

Reviewer 最容易从 `Result` 判断这个库是否好用。重点不是内部 solver，而是 solve
之后用户怎么拿数据。

推荐表达：

```python
result = solve(cfg=cfg)

u = result.field("u")
vm = result.field("von_mises")
native_u = result.point_field("u")
fields = result.available_fields()
```

检查项：

| 问题 | 应该满足 |
| --- | --- |
| mesh 和 field 是否清楚？ | `vertices`, `cells`, `point_data`, `cell_data`, `sampled_data`。 |
| sampled fallback 是否不伪装成 native data？ | `sampled_data` 不写入 `to_meshio()`。 |
| 常用 field 是否容易拿？ | `u`, `stress`, `strain`, `von_mises`, `field(...)`。 |
| field 来源是否可检查？ | `available_fields()`, `point_field(...)`, `cell_field(...)`, `sampled_field(...)`。 |
| transient history 是否有固定 shape？ | `docs/RESULT_CONTRACT.md` 说明 `history.u/vm/times`。 |

## 4. Examples Layering

不要把 example 数量作为质量标准。Phase 3 应该把 examples 分层：

| 层 | 作用 | 稳定性 |
| --- | --- | --- |
| core examples | 新用户第一入口，展示最稳定 public API。 | 高 |
| advanced examples | 展示 differentiation、parameterization、dataset export。 | 中 |
| paper/experiment scripts | 复现论文实验，可以包含 reporting/HPC 细节。 | 跟论文需求一起变 |

Reviewer 看到 examples 时应该知道：

- 先看 `examples/01_forward_solve.py` 和 `examples/02_result_fields.py`；
- differentiable 能力看 `examples/03` 到 `06`；
- paper reproduction 看 `experiment/paper_experiment/`；
- Compute Canada / long-running scripts 不是新用户 API。

## 5. Tests And Smoke Checks

每个 public contract 都应该有对应测试，而不是只靠 examples。

最低要求：

```bash
python -m pytest \
  tests/test_import_public_api.py \
  tests/test_config_json_io.py \
  tests/test_config_typed_blocks.py \
  tests/test_result_sampled_data.py \
  tests/test_result_history.py \
  tests/test_pipeline_normalize.py
```

更完整的 cleanup subset 见：

```text
docs/TEST_MATRIX.md
```

## 6. Reproducibility

Reviewer 应该能找到一个短路径来确认 artifact 能跑：

1. import public API；
2. run config/result unit tests；
3. run backend smoke test if compiled backend is available；
4. run one core example；
5. optionally run one differentiable example if PyTorch is installed。

具体命令见：

```text
docs/ARTIFACT_REPRODUCIBILITY.md
```

## 7. What Not To Do In Phase 3

现在不建议：

- 为了看起来高级而大拆 `config.py`；
- 为了模仿 DOLFINx 立刻引入 `Problem/Mesh/Material` 新对象体系；
- 为了模仿 PyTorch/JAX 立刻重写 differentiable API；
- 为了 example 数量而把 research scripts 包装成 public examples；
- 删除 compatibility aliases 而没有迁移说明。

这些都是后续 Phase 4/5 的事。Phase 3 的目标是让现有 API 有清楚 contract。

## 8. Reviewer-Style Verdict Template

每次做完一个 cleanup slice，可以用这个模板自查：

```text
Public surface:
- 推荐 import 是否仍然成立？
- README/example 是否仍然只展示 stable API？

Config:
- JSON path / dict / SimulationConfig 是否仍然正常？
- full JSON round-trip 是否仍然保留 solver-only fields？

Result:
- field namespace 是否仍然清楚？
- sampled data 是否没有被误写到 native mesh？

Examples:
- core examples 是否仍然短、可读、无 private path？
- paper scripts 是否仍然和 library API 有边界？

Tests:
- 是否跑了对应 TEST_MATRIX subset？
- 是否记录了不能跑的测试原因？
```
