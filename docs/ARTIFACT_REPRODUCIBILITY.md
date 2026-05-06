# Artifact Reproducibility

这个文件给自己和 reviewer 一个最短复现路径。它不要求每个 paper experiment 都
跑完，而是先证明 library artifact 的核心 contract 是可安装、可 import、可测试、
可运行一个小例子的。

## 0. 运行前提

推荐在 repo root 运行：

```bash
python -m pytest tests/test_import_public_api.py
```

如果使用本机已有环境，可以显式写：

```bash
~/polyfem_env/bin/python -m pytest tests/test_import_public_api.py
```

依赖分层：

| 能力 | 需要 |
| --- | --- |
| public import / config / result unit tests | Python + package import path |
| forward solve | compiled PolyFEM backend |
| VTU/mesh I/O | `meshio` |
| differentiable examples | PyTorch |
| long-running paper reproduction | paper experiment assets / HPC settings |

## 1. Tier 1: No-Backend Contract Checks

这些测试主要保护 Python API contract，不应该依赖 heavy solver run：

```bash
python -m pytest \
  tests/test_import_public_api.py \
  tests/test_config_json_io.py \
  tests/test_config_typed_blocks.py \
  tests/test_config_validate.py \
  tests/test_solver_method_blocks.py \
  tests/test_result_history.py \
  tests/test_result_sampled_data.py \
  tests/test_result_report.py
```

通过后可以说明：

- public import surface 没坏；
- `SimulationConfig` 的 JSON 语义没坏；
- `Result` 的 field/history/report contract 没坏。

## 2. Tier 2: Solve Pipeline Checks

这些测试保护 `solve(...)` 的 normalization、output extraction 和 fallback 逻辑：

```bash
python -m pytest \
  tests/test_pipeline_normalize.py \
  tests/test_pipeline_clean_json.py \
  tests/test_pipeline_runtime_options.py \
  tests/test_pipeline_helpers.py \
  tests/test_pipeline_extract_outputs.py \
  tests/test_pipeline_sampled_fallback.py
```

通过后可以说明：

- `cfg` 输入形式没有被 cleanup 改坏；
- JSON mode / array-backed mode 的关键 Python path 仍然可测；
- sampled fallback 和 `Result` handoff 仍然一致。

## 3. Tier 3: Backend Smoke

如果 compiled backend 可用，跑：

```bash
python -m pytest tests/test_backend_smoke.py
```

如果 backend 不可用，这个测试应该 graceful skip，而不是把整个 Python API
测试变成失败。

## 4. Tier 4: Public Examples

最小 public example：

```bash
python examples/01_forward_solve.py
```

如果安装了 `meshio`：

```bash
python examples/02_result_fields.py
```

如果安装了 PyTorch：

```bash
python examples/03_shape_gradient.py
python examples/04_scalar_E_gradient.py
```

这些 examples 的作用是 tutorial/smoke，不是完整 benchmark。输出写到：

```text
examples/runs/
```

这个目录是 generated output，不应该作为 API cleanup commit 的必要内容。

## 5. Tier 5: Paper Reproduction

paper-facing scripts 在：

```text
experiment/paper_experiment/
```

它们可以包含：

- longer optimization loops；
- reporting helpers；
- environment overrides；
- Compute Canada paths；
- mesh snapshots；
- result summaries。

这些脚本不是新用户第一入口。Reviewer 如果只想看 library API，应该先看
`examples/` 和 `docs/*_CONTRACT.md`。

## 6. Minimal Reviewer Command Set

给 reviewer 的短版命令可以是：

```bash
python -m pytest \
  tests/test_import_public_api.py \
  tests/test_config_json_io.py \
  tests/test_result_sampled_data.py \
  tests/test_pipeline_normalize.py
```

如果 backend 可用，再跑：

```bash
python -m pytest tests/test_backend_smoke.py
python examples/01_forward_solve.py
```

如果其中某一步不能跑，需要记录：

- command；
- failure reason；
- backend/dependency 是否缺失；
- 是否是 expected skip；
- 是否影响 public API contract。

## 7. Artifact Packaging Notes

提交或打包前检查：

```bash
git status --short
```

不应该把这些 generated outputs 当成 API cleanup 内容：

```text
examples/runs/
experiment/**/runs/
experiment/**/slurm_logs/
experiment/**/training_data/
experiment/**/zip_parts/
```

如果要给 ACM artifact 或 reviewer archive，应该保留：

- source code；
- docs；
- tests；
- small checked-in example meshes；
- minimal configs；
- exact commands；
- dependency notes；
- optional larger data 的下载或生成说明。
