# Reviewer Quickstart

这份文档是给 reviewer、advisor、或者自己交稿前快速检查用的入口页。

目标不是跑完所有 paper experiment，而是在较短时间内确认这个 repository 是一个
可 import、可测试、可运行小例子的 software artifact：

- public API 入口清楚；
- `SimulationConfig` / `solve(...)` / `Result` contract 没坏；
- differentiable / optimization helper 有稳定返回对象；
- public examples 和 paper reproduction scripts 有明确边界。

更完整的测试矩阵见 `TEST_MATRIX.md`，更完整的 artifact 说明见
`ARTIFACT_REPRODUCIBILITY.md`。

## 0. 从 Repo Root 开始

所有命令默认从 repository root 运行。

如果使用本机已有环境，可以先激活：

```bash
source "$HOME/polyfem_env/bin/activate"
```

或者显式使用环境里的 Python：

```bash
PYTHON="$HOME/polyfem_env/bin/python"
```

后面的命令可以把 `python` 替换成 `$PYTHON`。

## 1. 先确认 Backend 状态

真实 solve 需要 compiled PolyFEM backend。先看当前环境是否有 backend：

```bash
python -c "import polyfempy as pf; print(pf.cpp_backend_available()); print(pf.cpp_backend_error())"
```

如果输出是：

```text
True
None
```

说明可以跑真实 solver examples。

如果 backend 不可用，仍然可以跑 public import、config、result、pipeline
normalization 等 Python contract tests。`test_backend_smoke.py` 应该 skip，而不应该
让整个 API 检查失败。

## 2. 最短 Reviewer Smoke

这组命令最适合作为 reviewer 第一轮检查。它不要求跑完整 heavy suite：

```bash
python -m pytest \
  tests/test_import_public_api.py \
  tests/test_config_json_io.py \
  tests/test_result_sampled_data.py \
  tests/test_pipeline_normalize.py
```

通过后可以说明：

- 推荐 public imports 没坏；
- `SimulationConfig` JSON semantics 没坏；
- `Result` field namespace / sampled-data contract 没坏；
- `solve(...)` 的输入 normalization 没坏。

## 3. 更完整的 API Contract Check

如果 reviewer 想多看一点，但还不想跑完整 suite，可以跑：

```bash
python -m pytest \
  tests/test_import_public_api.py \
  tests/test_config_json_io.py \
  tests/test_config_typed_blocks.py \
  tests/test_config_validate.py \
  tests/test_solver_method_blocks.py \
  tests/test_result_history.py \
  tests/test_result_meshio_roundtrip.py \
  tests/test_result_report.py \
  tests/test_result_sampled_data.py \
  tests/test_pipeline_normalize.py \
  tests/test_pipeline_clean_json.py \
  tests/test_pipeline_runtime_options.py \
  tests/test_pipeline_extract_outputs.py \
  tests/test_pipeline_sampled_fallback.py \
  tests/test_optimization_run_result.py \
  tests/test_parameterized_shape_problem.py
```

这组测试覆盖四个最重要的 library contracts：

| Contract | 代表内容 |
| --- | --- |
| Public surface | `from polyfempy.api import solve, SimulationConfig, Result` |
| Config | full/minimal JSON、typed blocks、solver sections |
| Result | `point_data` / `cell_data` / `sampled_data`、history、meshio |
| Differentiable optimization | `OptimizationRunResult.summary()`、parameterized vertex map |

## 4. Backend Smoke

如果第 1 步显示 backend 可用，跑：

```bash
python -m pytest tests/test_backend_smoke.py
```

这个测试的意义是确认最小 `solve(cfg=...)` path 能进入 compiled backend。

如果 backend 不可用，预期行为是 skip。记录成 dependency/environment limitation 即可。

## 5. Public Examples

Reviewer 第一次看 library 用法，应该先看 `examples/`，不是先看 long paper scripts。

最小 forward solve：

```bash
python examples/01_forward_solve.py
```

预期输出位置：

```text
examples/runs/01_forward_solve_*/impact_stats.json
examples/runs/01_forward_solve_*/polyfem.log
examples/runs/01_forward_solve_*/impact_step_*.vtu
```

`Result` field inspection 和 VTU export：

```bash
python examples/02_result_fields.py
```

预期输出位置：

```text
examples/runs/02_result_fields_*/result_fields_summary.json
examples/runs/02_result_fields_*/result_fields.vtu
```

`result_fields_summary.json` 里应该能看到类似信息：

```json
{
  "available_fields": {
    "point_data": ["u"],
    "cell_data": [],
    "sampled_data": ["von_mises", "von_mises_avg"]
  },
  "von_mises_info": {
    "available": true,
    "namespace": "sampled_data",
    "source": "history:last_frame"
  }
}
```

如果安装了 PyTorch，可以继续跑 differentiable examples：

```bash
python examples/03_shape_gradient.py
python examples/04_scalar_E_gradient.py
python examples/05_parameterized_vertex_map.py
```

`examples/05_parameterized_vertex_map.py` 会展示：

```python
run = run_optimization(..., return_result=True)
summary = run.summary()
```

也就是 reviewer 可以直接看到 optimization result contract。

## 6. Paper-Facing Demos

Paper-facing scripts 在：

```text
experiment/paper_experiment/
```

推荐阅读顺序：

1. `experiment/paper_experiment/README.md`
2. `experiment/paper_experiment/CLEAN_API_WALKTHROUGH.md`
3. `experiment/paper_experiment/01_forward_von_mises.py`
4. `experiment/paper_experiment/02_shape_diff.py`
5. `experiment/paper_experiment/03_E_diff.py`
6. `experiment/paper_experiment/04_x_shape_optimization.py`
7. `experiment/paper_experiment/08_h_theta_shape_optimization.py`

这里的边界要清楚：

- `examples/` 是 stable library tutorial / smoke layer；
- `experiment/paper_experiment/` 是 paper reproduction layer；
- long-running scripts 可以包含 reporting、early stopping、mesh snapshots、HPC
  settings，不应该被当作新用户第一 API。

## 7. Reviewer 应该重点检查什么

### Public API

推荐入口应该保持小：

```python
from polyfempy.api import solve, SimulationConfig, Result
import polyfempy.api.guided as g
```

Differentiable 入口应该通过 public helper：

```python
from polyfempy.differentiable import (
    prepare_differentiable_simulation,
    prepare_optimization_problem,
    prepare_parameterized_shape_problem,
    make_von_mises_loss,
    run_optimization,
)
```

### Config

Reviewer 应该能区分三种 solve input：

```python
solve(cfg=python_dict)
solve(cfg="path/to/config.json")
solve(cfg=SimulationConfig(...))
```

详细语义见 `CONFIG_CONTRACT.md`。

### Result

Reviewer 应该能看懂 solver output 怎么取：

```python
u = result.require_field("u", namespace="point_data")
vm = result.require_field("von_mises")
info = result.field_info("von_mises")
```

详细语义见 `RESULT_CONTRACT.md`。

### Optimization

Reviewer 应该能看到 optimization loop 返回稳定 summary：

```python
run = run_optimization(..., return_result=True)
print(run.success, run.message)
print(run.final_loss, run.best_loss)
print(run.summary())
```

详细语义见 `DIFFERENTIABLE_CONTRACT.md`。

## 8. 如何解释 Skip 或失败

如果某一步不能跑，先分类，不要立刻改 API：

| 情况 | 应该怎么解释 |
| --- | --- |
| backend unavailable | compiled PolyFEM backend 没装，backend smoke 可以 skip。 |
| `meshio` missing | 只影响 VTU/mesh I/O example，不代表 core API 不能 import。 |
| PyTorch missing | 只影响 differentiable examples/tests。 |
| import smoke failure | public facade 真的坏了，必须修。 |
| config/result contract failure | public semantics 变了，必须判断是 bug 还是 intentional API change。 |
| long paper script failure | 先区分是 HPC/data/path 问题，还是 public API 问题。 |

## 9. 提交或打包前检查

提交前至少跑：

```bash
git diff --check
python -m pytest tests/test_import_public_api.py
```

如果改了 `Result`、config、solve pipeline、或者 differentiable helper，按
`TEST_MATRIX.md` 跑对应 subset。

生成输出不要作为 API cleanup 内容提交：

```text
examples/runs/
experiment/**/runs/
experiment/**/slurm_logs/
experiment/**/training_data/
```

最终 reviewer-facing artifact 应该包含：

- source code；
- docs；
- tests；
- small checked-in example meshes；
- exact commands；
- optional dependency notes；
- paper reproduction entry points。
