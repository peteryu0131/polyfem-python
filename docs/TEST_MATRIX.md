# Test Matrix

这个文件回答一个 practical 问题：每次 cleanup 一个 slice 后，到底该跑哪些测试。

原则：

- 小改动跑对应 subset，不需要每次都跑完整 heavy suite；
- public API / config / result 的 contract tests 优先；
- backend 或 PyTorch 不可用时，要区分 expected skip 和真实失败；
- 如果改动跨多个层，跑所有相关 subset 的并集。

## Always Run For API Cleanup

任何会影响 public import、docs examples、facade、guided namespace 的改动，都先跑：

```bash
python -m pytest tests/test_import_public_api.py
```

如果改了 Python 文件，还要跑语法检查：

```bash
python -m py_compile path/to/changed_file.py
```

提交前至少跑：

```bash
git diff --check
```

## Cleanup Slice Matrix

| Slice | 常见文件 | 必跑测试 |
| --- | --- | --- |
| public facade / `__all__` | `polyfempy/api/__init__.py`, `polyfempy/api/guided.py` | `tests/test_import_public_api.py` |
| guided sections | `guided.py`, `guided_builders.py`, `guided_types.py`, `_guided_config.py`, `_guided_array_mesh.py` | `tests/test_import_public_api.py`, `tests/test_guided_config_builder.py`, `tests/test_geometry_transformations.py`, `tests/test_config_typed_blocks.py` |
| config JSON semantics | `config.py` | `tests/test_config_json_io.py`, `tests/test_config_typed_blocks.py`, `tests/test_config_validate.py`, `tests/test_solver_method_blocks.py` |
| shared solve contract | `_solve_contract.py`, `_solve_settings.py` | `tests/test_pipeline_normalize.py`, `tests/test_pipeline_clean_json.py`, `tests/test_pipeline_sampled_fallback.py`, `tests/test_differentiable_solve_settings.py` |
| solve input normalization | `solve.py`, `_solve_pipeline.py` | `tests/test_pipeline_normalize.py`, `tests/test_pipeline_clean_json.py`, `tests/test_pipeline_runtime_options.py` |
| output extraction / fallback | `_solve_pipeline.py`, `result.py` | `tests/test_pipeline_extract_outputs.py`, `tests/test_pipeline_sampled_fallback.py`, `tests/test_result_sampled_data.py` |
| `Result` field contract | `result.py`, `report.py` | `tests/test_result_history.py`, `tests/test_result_meshio_roundtrip.py`, `tests/test_result_report.py`, `tests/test_result_sampled_data.py` |
| mesh I/O | `io.py`, `result.py` | `tests/test_result_meshio_roundtrip.py` |
| runtime helpers | `runtime.py`, `_runtime.py` | `tests/test_runtime_windows_shim.py`, `tests/test_pipeline_runtime_options.py` |
| predefined problems | `problems.py`, `config.py` | `tests/test_api_problems.py` |
| differentiable public imports | `polyfempy/differentiable/__init__.py` | import smoke plus focused differentiable tests if available |
| optimization result contract | `differentiable/optimization/`, `differentiable/shape/`, `differentiable/material/` | `tests/test_optimization_run_result.py`, `tests/test_parameterized_shape_problem.py`, `tests/test_import_public_api.py` |
| examples only | `examples/*.py`, `examples/README.md` | `tests/test_import_public_api.py`; run touched example if backend/deps are available |
| docs only | `docs/*.md`, `README.md` | `git diff --check`; no pytest required unless code examples/imports changed |

## Recommended Bundles

### Public API Bundle

```bash
python -m pytest \
  tests/test_import_public_api.py \
  tests/test_config_typed_blocks.py \
  tests/test_geometry_transformations.py
```

Use after changing facade, guided namespace, or examples imports.

### Config Bundle

```bash
python -m pytest \
  tests/test_config_json_io.py \
  tests/test_config_typed_blocks.py \
  tests/test_config_validate.py \
  tests/test_solver_method_blocks.py \
  tests/test_pipeline_normalize.py \
  tests/test_pipeline_runtime_options.py
```

Use before/after splitting `config.py`.

### Result Bundle

```bash
python -m pytest \
  tests/test_result_history.py \
  tests/test_result_meshio_roundtrip.py \
  tests/test_result_report.py \
  tests/test_result_sampled_data.py \
  tests/test_pipeline_extract_outputs.py \
  tests/test_pipeline_sampled_fallback.py
```

Use after changing `Result`, `report.py`, or solve output extraction.

### Solve Pipeline Bundle

```bash
python -m pytest \
  tests/test_pipeline_normalize.py \
  tests/test_pipeline_clean_json.py \
  tests/test_pipeline_runtime_options.py \
  tests/test_pipeline_helpers.py \
  tests/test_pipeline_extract_outputs.py \
  tests/test_pipeline_sampled_fallback.py
```

Use after changing `solve.py` or `_solve_pipeline.py`.

### Shared Solve Contract Bundle

```bash
python -m pytest \
  tests/test_pipeline_normalize.py \
  tests/test_pipeline_clean_json.py \
  tests/test_pipeline_runtime_options.py \
  tests/test_pipeline_sampled_fallback.py \
  tests/test_differentiable_solve_settings.py
```

Use after changing `polyfempy/api/_solve_contract.py`,
`polyfempy/differentiable/_solve_settings.py`, or `docs/SOLVE_CONTRACT.md`.

### TOMS Reviewer Smoke

```bash
python -m pytest \
  tests/test_import_public_api.py \
  tests/test_config_json_io.py \
  tests/test_result_sampled_data.py \
  tests/test_pipeline_normalize.py \
  tests/test_differentiable_solve_settings.py
```

This is the short API-contract smoke check. It is not a substitute for full
tests, but it is useful before showing the repo to an advisor/reviewer.

## When To Run Full Tests

Run:

```bash
python -m pytest tests
```

when:

- a cleanup touches both config and solve pipeline；
- a change moves public imports；
- a Result change affects fallback or mesh I/O；
- a differentiable change touches shared helpers；
- before a release/artifact snapshot；
- before pushing if the branch has accumulated multiple cleanup slices。

## Example Smoke Rules

Examples are smoke/tutorial checks, not the primary contract tests.

Run touched examples when dependencies are available:

```bash
python examples/01_forward_solve.py
python examples/02_result_fields.py
```

For PyTorch examples:

```bash
python examples/03_shape_gradient.py
python examples/04_scalar_E_gradient.py
python examples/05_parameterized_vertex_map.py
python examples/06_dataset_one_case.py
```

If an example cannot run because backend/PyTorch/meshio is missing, record that
dependency reason explicitly.

## Failure Triage

When a subset fails, classify it before changing code:

| Failure type | Meaning | Action |
| --- | --- | --- |
| import error | facade/package path broke | fix before continuing |
| assertion failure in contract tests | public semantics changed | decide if intended, then update code/docs/tests together |
| backend unavailable skip | environment issue | acceptable if documented |
| meshio/PyTorch missing | optional dependency issue | acceptable for optional examples |
| generated output mismatch | may be example/reporting issue | inspect before changing API |
