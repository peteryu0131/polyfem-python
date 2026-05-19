# Shared Solve Contract

这个文件记录当前 `solve(...)` 和 differentiable solve 共同使用的 internal
contract。它面向 reviewer / advisor：重点是说明 config、mesh 和 backend
settings 在进入 C++ backend 之前如何被统一，而不是把实验脚本细节包装成 API。

## One-Sentence Summary

`polyfempy` 的 solve path 应该是：

```text
user cfg + optional arrays
  -> shared config / mesh / settings contract
  -> forward runtime or differentiable runtime
  -> Result or DifferentiableResult
```

也就是说，forward solve 和 differentiable solve 不应该各自猜 config 或 mesh
来源。它们先经过同一套 contract，再进入各自 runtime。

## Shared Internal Module

当前 shared contract 在：

```text
polyfempy/api/_solve_contract.py
```

主要对象和 helper：

| Name | Role |
| --- | --- |
| `normalize_config(...)` | 接受 `SimulationConfig` / `dict` / JSON path，返回 `SimulationConfig`。 |
| `build_full_json(...)` | 保留 full JSON snapshot，并 overlay Python-side edits。 |
| `choose_mesh_source(...)` | 明确本次 solve 使用 JSON mesh、direct array，还是 guided array payload。 |
| `build_canonical_solver_settings(...)` | 生成 C++ backend 消费的 JSON/settings。 |
| `prepare_canonical_solve_input(...)` | 一次性返回 config、mesh source、backend settings 和 metadata。 |
| `MeshSource` | 记录真实 mesh 来源以及 vertices/cells/body_ids/boundary_ids。 |
| `CanonicalSolveInput` | pipeline 入口消费的 canonical payload。 |

这些名字是 internal helper，不是推荐 public API。用户仍然从：

```python
from polyfempy.api import solve, SimulationConfig, Result
```

进入 forward solve。

## Config Normalization

`cfg` 支持三种形式：

| User input | Meaning |
| --- | --- |
| `SimulationConfig` | 已构造好的 solver-facing config object。 |
| `dict` | JSON-style config dict。 |
| JSON path `str` | 从磁盘加载 config，并记录 root path。 |

shared contract 先统一为 `SimulationConfig`。这样后面的 forward pipeline 和
differentiable pipeline 不需要各自写一套：

```text
if cfg is dict
if cfg is str
if cfg is SimulationConfig
```

这个设计保护两件事：

- JSON-only fields 不应该因为 Python wrapper 丢失；
- Python-side edits 应该明确 overlay 到 backend settings。
- 如果 `cfg.to_dict()` 序列化失败，contract 层应该立即报错，不能继续使用旧
  full JSON 或 `None` 掩盖 Python-side edit 没有生效的问题。

## Mesh Source Selection

mesh 来源必须在 runtime 前明确。当前有三种模式：

| Mode | Input | Backend action |
| --- | --- | --- |
| `json` | config 中有 `geometry` mesh 文件 | `solver.load_mesh_from_settings()` |
| `array` | caller 传 `vertices` 和 `cells` | `solver.set_mesh(vertices, cells)` |
| `guided_array` | `g.body_section(vertices=..., cells=...)` 生成 payload | merge 后 `solver.set_mesh(vertices, cells)` |

重要语义：

- 只传 `vertices` 或只传 `cells` 是错误；
- 用户传 array，就不能 silently fallback 到 JSON mesh；
- guided array-backed body 的 fake placeholder 不应该进入 backend settings；
- 最终 `mesh_source` 会进入 metadata / diagnostics，方便 reviewer 判断真实数据来源。

## Canonical Backend Settings

用户写的是 Python-friendly config，但 C++ backend 消费的是 JSON-like settings。
shared contract 的 `build_canonical_solver_settings(...)` 负责把 config 转成 backend
settings：

```text
normalized config
  + selected mesh source
  -> backend_settings
```

它会处理：

- full JSON cleanup，例如移除 Python-only output/result/fallback fields；
- root path preservation/resolution；
- array mode 的 ground geometry fallback；
- material dict/list normalization；
- guided array placeholder cleanup。

runtime layer 不应该再做用户语义决策。它只应该 apply settings、attach mesh、
build basis、assemble、solve、extract outputs。

## Forward Solve Pipeline

`polyfempy.api.solve(...)` 仍然是 forward simulation 的 public entry point。
内部 pipeline 的理想职责是 orchestration：

```text
solve(...)
  -> prepare_canonical_solve_input(...)
  -> build solver
  -> set backend settings
  -> attach/load mesh
  -> apply body/boundary/sideset data
  -> build_basis / assemble / solve
  -> finalize Result
```

这和以前的区别是：config normalization、mesh source selection、backend settings
construction 不再只是 `_solve_pipeline.py` 里的 private ad hoc logic，而是可测试的
shared contract。

## Differentiable Solve Pipeline

differentiable-specific wrapper 在：

```text
polyfempy/differentiable/_solve_settings.py
```

核心 helper：

```text
prepare_differentiable_solve_contract(...)
```

它复用 `prepare_canonical_solve_input(...)`，然后只增加 differentiable-specific
runtime settings：

- runtime patches applied to copied backend settings；
- diagnostics recording；
- no mutation of user `SimulationConfig`；
- same `MeshSource` semantics as forward solve。

`solve_differentiable(...)` 现在消费这个 contract，而不是自己再从
`cfg.extras["_mesh_array_mode"]` 里捞 mesh payload。

## Compatibility Wrapper

旧 helper：

```text
_differentiable_config_and_settings(...)
```

仍然保留给 shape/material optimization 内部和旧 code path 使用。现在它优先委托给
`prepare_differentiable_solve_contract(...)`，只在 settings-only、无 mesh 的旧场景下
保留 fallback。

这保证旧入口不变，但主要 config/mesh/settings 语义不再分叉。

## Reviewer-Facing Invariants

Reviewer 可以重点检查这些不变量：

| Invariant | Why it matters |
| --- | --- |
| partial array input raises | 防止用户以为用了 array，实际跑 JSON mesh。 |
| JSON-only fields are preserved | 防止 Python wrapper 丢 solver config。 |
| guided array placeholders do not reach backend settings | 防止 fake geometry 泄漏到 solver JSON。 |
| `sampled_vtu="never"` disables fallback | 防止 hidden output fallback。 |
| differentiable runtime patches do not mutate user config | 防止 repeated solve / optimization state 被污染。 |
| forward and differentiable paths share mesh source semantics | 防止 two solve paths 对同一 config 得到不同含义。 |

## Recommended Tests

改 shared solve contract 后，至少跑：

```bash
python -m pytest \
  tests/test_pipeline_normalize.py \
  tests/test_pipeline_clean_json.py \
  tests/test_pipeline_runtime_options.py \
  tests/test_pipeline_sampled_fallback.py \
  tests/test_differentiable_solve_settings.py
```

如果改动影响 public facade 或 reviewer docs，再加：

```bash
python -m pytest \
  tests/test_import_public_api.py \
  tests/test_docs_index.py
```

提交或 push 前建议跑完整 suite：

```bash
python -m pytest tests
```
