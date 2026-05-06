# API Internal Import Audit

这个文件记录 Phase 2 的 import 审计。目的不是评价代码好坏，而是回答：

- 当前哪些文件真的 import 了 `polyfempy.api`；
- 哪些 import 是推荐 public API；
- 哪些 import 是 tests 为了覆盖 internal pipeline；
- 哪些 import 是兼容或 paper demo 用法；
- 哪些地方后续 cleanup 时不能随便改。

审计命令：

```bash
rg -n "from polyfempy\\.api|import polyfempy\\.api" examples tests experiment/paper_experiment polyfempy -S
```

## 总结

当前 import 面比较健康：

- examples 和 paper demos 主要使用 `polyfempy.api`、`polyfempy.api.guided`、`polyfempy.api.runtime`。
- tests 会直接 import `_solve_pipeline`、`_runtime` 等 internal helpers，这是为了单元测试 pipeline stage。
- `solve.py` 里的 backward-compatible aliases 当前 repo 内没有真实 caller。
- `config.py`、`result.py`、`report.py` 仍然有直接 imports，主要出现在 tests 和 advanced docs。

结论：

```text
用户文档可以主推小 public surface。
internal helpers 可以继续被 tests 覆盖。
compatibility aliases 先保留，但不再推荐新代码使用。
```

## Examples

### Top-level examples

| 文件 | Import | 分类 | 处理 |
| --- | --- | --- | --- |
| `examples/01_forward_solve.py` | `from polyfempy.api import solve` | 推荐 public | 保留 |
| `examples/01_forward_solve.py` | `import polyfempy.api.guided as g` | 推荐 guided | 保留 |
| `examples/02_result_fields.py` | `from polyfempy.api import solve` | 推荐 public | 保留 |
| `examples/02_result_fields.py` | `import polyfempy.api.guided as g` | 推荐 guided | 保留 |
| `examples/03_shape_gradient.py` | `import polyfempy.api.guided as g` | 推荐 guided | 保留 |
| `examples/04_scalar_E_gradient.py` | `import polyfempy.api.guided as g` | 推荐 guided | 保留 |
| `examples/05_parameterized_vertex_map.py` | `import polyfempy.api.guided as g` | 推荐 guided | 保留 |
| `examples/06_dataset_one_case.py` | `import polyfempy.api.guided as g` | 推荐 guided | 保留 |
| `examples/_common.py` | `from polyfempy.api.runtime import ...` | runtime helper | 保留，但不作为核心 API 主推 |

examples 说明了当前推荐路径：

```text
guided sections -> build_config -> solve / differentiable helpers
```

这符合 public surface 决策。

## Paper Experiments

| 文件 | Import | 分类 | 处理 |
| --- | --- | --- | --- |
| `experiment/paper_experiment/01_forward_von_mises.py` | `from polyfempy.api import solve` | 推荐 public | 保留 |
| `experiment/paper_experiment/01_forward_von_mises.py` | `import polyfempy.api.guided as g` | 推荐 guided | 保留 |
| `experiment/paper_experiment/02_shape_diff.py` | `import polyfempy.api.guided as g` | 推荐 guided | 保留 |
| `experiment/paper_experiment/03_E_diff.py` | `import polyfempy.api.guided as g` | 推荐 guided | 保留 |
| `experiment/paper_experiment/04_x_shape_optimization.py` | `import polyfempy.api.guided as g` | 推荐 guided | 保留 |
| `experiment/paper_experiment/05_h_theta_manual_vertex_map.py` | `import polyfempy.api.guided as g` | 推荐 guided | 保留 |
| `experiment/paper_experiment/07_h_theta_fix06_global_affine_vertex_map.py` | `import polyfempy.api.guided as g` | 推荐 guided | 保留 |
| `experiment/paper_experiment/08_h_theta_shape_optimization.py` | `import polyfempy.api.guided as g` | 推荐 guided | 保留 |
| `experiment/paper_experiment/common.py` | `from polyfempy.api.runtime import ...` | runtime helper | 保留 |

paper demos 没有直接依赖 `_solve_pipeline`。这点很重要：paper-facing code 已经在 public/guided API 上。

## Tests

tests 分成两类。

第一类是 public import smoke tests：

| 文件 | Import | 分类 | 处理 |
| --- | --- | --- | --- |
| `tests/test_import_public_api.py` | `from polyfempy.api import Result, SimulationConfig, solve` | public contract test | 保留 |
| `tests/test_import_public_api.py` | `from polyfempy.api.guided import ...` | guided public contract test | 保留 |
| `tests/test_import_public_api.py` | `from polyfempy.api.problems import ...` | compatibility test | 保留 |
| `tests/test_backend_smoke.py` | `from polyfempy.api import SimulationConfig, solve` | public/backend smoke | 保留 |

第二类是 internal pipeline tests：

| 文件 | Import | 分类 | 处理 |
| --- | --- | --- | --- |
| `tests/test_pipeline_normalize.py` | `from polyfempy.api._solve_pipeline import ...` | internal test target | 保留 |
| `tests/test_pipeline_runtime_options.py` | `from polyfempy.api._solve_pipeline import ...` | internal test target | 保留 |
| `tests/test_pipeline_clean_json.py` | `from polyfempy.api._solve_pipeline import ...` | internal test target | 保留 |
| `tests/test_pipeline_extract_outputs.py` | `from polyfempy.api._solve_pipeline import ...` | internal test target | 保留 |
| `tests/test_pipeline_helpers.py` | `from polyfempy.api._solve_pipeline import ...` | internal test target | 保留 |
| `tests/test_pipeline_sampled_fallback.py` | `import polyfempy.api._solve_pipeline as _p` | internal test target | 保留 |
| `tests/test_result_sampled_data.py` | `from polyfempy.api._solve_pipeline import _field_available` | internal test target | 保留 |
| `tests/test_runtime_windows_shim.py` | `from polyfempy.api._runtime import ...` | internal test target | 保留 |

这些 internal imports 不代表用户 API。它们是为了让 pipeline stage 可以被细粒度测试。

## Package Internals

| 文件 | Import | 分类 | 处理 |
| --- | --- | --- | --- |
| `polyfempy/api/guided_sections.py` | `from .config import ...` | package internal | 保留 |
| `polyfempy/api/guided_sections.py` | `from ._guided_array_mesh import ...` | internal implementation | 保留 |
| `polyfempy/api/guided.py` | `from polyfempy.api.guided_sections import ...` | public facade implementation | 保留 |
| `polyfempy/api/config.py` | `from polyfempy.api.problems import get_problem_class` | compatibility path | 保留 |

`guided_sections.py` 已经改成相对 import，避免内部 implementation 反向依赖
`polyfempy.api` 顶层 facade。

## Documentation Imports

| 文件 | Import | 分类 | 处理 |
| --- | --- | --- | --- |
| `README.md` | `from polyfempy.api import SimulationConfig, solve` | 推荐 public | 保留 |
| `polyfempy/README.md` | `from polyfempy.api import SimulationConfig, Result, solve` | 推荐 public | 保留 |
| `polyfempy/api/API_GUIDE.md` | 多个 `polyfempy.api` imports | user guide | 后续同步 public surface wording |

后续文档 cleanup 应该统一说法：

```text
核心 API 是 solve / SimulationConfig / Result。
guided API 是 config authoring layer。
runtime/report 是脚本 helper。
internal 下划线模块只给实现和 tests 用。
```

## `solve.py` Alias Audit

审计命令覆盖这些 alias：

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

结果：

- repo 内没有业务代码从 `polyfempy.api.solve` import 这些 alias。
- tests 直接使用 `_solve_pipeline` 的 helpers。
- `solve.py` 内这些 alias 是历史兼容层，不是当前内部依赖。

处理建议：

```text
Phase 2:
  保留 alias
  标记 compatibility-only
  不在新代码中使用

Phase 3:
  如果外部旧脚本也无依赖，再删除或 deprecate
```

## 结论

当前 import 面支持这个 public surface 决策：

```text
推荐 public:
  polyfempy.api.solve
  polyfempy.api.SimulationConfig
  polyfempy.api.Result

推荐 guided:
  polyfempy.api.guided.*

脚本 helper:
  polyfempy.api.runtime
  polyfempy.api.report

internal:
  polyfempy.api._solve_pipeline
  polyfempy.api._guided_array_mesh
  polyfempy.api._runtime
```

这意味着 Phase 2 可以先做文档和轻量整理，不需要马上大拆文件。
