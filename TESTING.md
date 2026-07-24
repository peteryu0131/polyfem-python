# PolyFEM-Python 测试和安装流程

这份文件说明一个新用户从 GitHub clone 到安装成功后，应该如何验证
`polyfem-python`。核心目标是分清三件事：

1. Python/generator 是否正常。
2. C++ backend 是否能编译和运行。
3. generated API example 是否真的能调用 backend，并和对应的
   `polyfem-data` JSON 运行行为一致。

## 1. Fresh Clone

推荐新用户直接 clone submodules：

```powershell
git clone --recurse-submodules https://github.com/polyfem/polyfem-python.git
cd polyfem-python
```

如果已经 clone 了 repo，但是没有 submodules：

```powershell
git submodule update --init --recursive
```

这个 repo 现在依赖这些 submodule 路径：

```text
polyfem/          C++ backend source and canonical JSON specs
python-from-jse/  generic JSON-spec-to-Python generator
polyfem-data/     meshes, source JSON configs, expected tests
examples/         generated API examples
```

## 2. Backend-free 自动测试

这一步不编译 C++ backend，速度快，适合每次 push/PR 自动跑。

```powershell
python -m pip install -U pip numpy pytest
python tools\generate_polyfem_api.py
python -m pytest tests -q
```

GitHub Actions 对应文件：

```text
.github/workflows/test.yml
```

它会自动做：

```text
checkout submodules
install numpy/pytest
python tools/generate_polyfem_api.py
python -m pytest tests -q
```

这层测试证明：

```text
generator 可以从 polyfem/json-specs 生成 packaged API
polyfempy/generated_api 可以 import
classic generated examples 的 as_dict() 和 polyfem-data 源 JSON 一致
runtime 的 Python contract 没有回到 legacy u/p/vertices/cells 输出
```

这层测试不会证明 backend 可以算。

## 3. 本地 source build

真实 backend 需要编译 C++ extension。Windows 本地建议使用你的 conda 环境：

```powershell
conda activate polyfem
git submodule update --init --recursive
python -m pip install -U pip setuptools wheel cmake nanobind pytest numpy
python tools\generate_polyfem_api.py
$env:N_THREADS = "4"
python -m pip install -e . --no-build-isolation -vv
```

安装后先检查 backend 是否加载：

```powershell
python -m polyfempy backend-info --require
```

成功时应该看到：

```text
backend_available=True
backend_error=None
```

## 4. Backend smoke test

安装 backend 后，先跑最小真实计算测试：

```powershell
python -m pytest tests\test_backend_smoke.py -q -rs
```

这个测试实际调用：

```text
python -m polyfempy solve polyfem-data/units/neohookean.json ...
```

然后读取 backend 输出的：

```text
sim.json
```

再把 `sim.json` 顶层的：

```text
err_h1
err_h1_semi
err_l2
err_linf
err_linf_grad
err_lp
```

和 `polyfem-data/units/neohookean.json` 里的 `tests` 对比。

这层测试证明：

```text
C++ backend 能加载
CLI 能调用 backend
backend 能真实完成一个小 case
backend 输出的 err_* 和 polyfem-data/units 的 tests 对齐
```

## 5. GitHub backend workflow

真实 backend 编译比较慢，所以现在单独放在手动 workflow：

```text
.github/workflows/backend.yml
```

GitHub 上进入 Actions，选择 `Backend`，手动点 `Run workflow`。

默认会跑：

```text
checkout submodules
install build/test dependencies
python tools/generate_polyfem_api.py
python -m pip install -e . --no-build-isolation -vv
python -m polyfempy backend-info --require
python -m pytest tests/test_backend_smoke.py -q -rs
```

这条 workflow 现在先不要放进普通 PR 自动 gate，因为 C++ build 慢，
也更容易受 GitHub runner 环境影响。

## 6. Generated example backend diagnostic

`tests/test_generated_api_example.py` 现在只证明：

```text
generated example 的 polyfem_config.as_dict()
等于 polyfem-data/contact/examples/... 的源 JSON 展开结果
```

它不运行 backend。

如果要证明某个 generated example 真的能算，需要手动跑：

```powershell
python tools\check_generated_example_backend.py `
  --example examples\classic_example\2D\contact_2d_golf_ball_deformable_wall_generated_api.py `
  --source-json polyfem-data\contact\examples\2D\golf-ball-doformable-wall.json
```

这个工具会做两次 backend run：

```text
1. import generated example
2. call example.config_for_workspace(...)
3. call polyfempy.runtime.solve(cfg=...)
4. run source JSON through python -m polyfempy solve ...
5. compare generated output sim.json vs source JSON output sim.json
6. also report generated output vs source JSON tests
```

默认情况下，工具只要求：

```text
generated output == source JSON output
```

如果要强制检查 source JSON 的 `tests` 也必须匹配，加：

```powershell
--require-tests-match
```

目前不要把这个 generated example diagnostic 放进默认 CI gate。原因是：

```text
contact examples 比 units/neohookean 慢很多
当前新 backend 跑 contact example 的结果和 polyfem-data/contact 的 tests 不一致
但是 generated example 和原始 source JSON 的运行结果是一致的
```

这说明当前需要先和老师确认：

```text
polyfem-data/contact/examples/... 的 tests 是否需要按新 VarForm backend 更新
```

## 7. 哪些测试暂时保留

现在不要删除这些看起来像 cleanup guard 的测试：

```text
tests/test_pipeline_outputs.py
tests/test_pipeline_extract_outputs.py
tests/test_result_container.py
tests/test_varform_binding_contract.py
```

它们保护的是新 runtime 边界：

```text
只支持 VarForm backend solve(sol)
Result 只保留 raw sol 和 meta
不再恢复 legacy u/p/vertices/cells/output helper API
```

等 backend contract 稳定后，可以再减少这些 negative tests。

## 8. 推荐的长期自动化结构

当前建议保持三层：

```text
test.yml
  每次 push/PR 自动跑
  不 build backend

backend.yml
  手动跑
  build backend
  跑 backend smoke

generated example backend diagnostic
  手动跑
  比较 generated example output 和 source JSON output
  等 polyfem-data/contact tests 和新 backend 对齐后，再考虑进入 backend.yml
```
