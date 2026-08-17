# PolyFEM-Python Release Candidate Checklist

更新时间：2026-08-17

这份 checklist 的目标是避免一种最常见的 release 事故：本机能跑，但别人 fresh clone 后拿不到 submodule commit、拿不到 generated examples，或者不知道哪些 backend failure 是允许忽略的。

## 0. 当前 release 口径

老师要求运行的 contact example 范围来自：

```text
polyfem/tests/contact_2d.txt
polyfem/tests/contact_3d.txt
```

只计算没有 `#` 注释的 `contact/examples/...` 行。当前 active set 是：

```text
2D active cases: 25
3D active cases: 42
Total active:    67
```

当前 release gate 目标：

```text
PASS: 66
IGNORED: 1
FAIL: 0
Unexpected pass: 0
Unexpected fail: 0
```

唯一 expected ignored case 写在：

```text
tools/generated_contact_expected_failures.json
```

当前只有：

```text
contact/examples/3D/rigid/proxy/screw.json
```

原因是当前 Windows Python backend build 读取 `screw-coarse-to-screw.hdf5` 时出现 `h5pp` type-size mismatch。老师已经说这个可以先 ignore。

## 1. 先检查 parent repo 和 submodules

在 parent repo 根目录运行：

```powershell
git status --short
git submodule status --recursive
git submodule foreach --recursive git status --short
```

当前最需要关注的 submodules：

```text
python-from-jse
examples
```

`polyfem` 和 `polyfem-data` 如果是 clean，就不要动它们。

## 2. 先提交并推送 python-from-jse

进入 submodule：

```powershell
cd python-from-jse
git status --short
python -m pytest tests\test_generated_api.py tests\test_model_builder.py -q
git add generator\JsonToTreeClass.py generator\model_builder.py tests\test_generated_api.py tests\test_model_builder.py
git commit -m "Support model-style contact example generation"
git push origin jingyao
cd ..
```

这一步必须先于 parent repo commit。否则 parent repo 后面如果指向这个 submodule commit，别人 clone 时可能拿不到。

## 3. 再提交并推送 examples

进入 submodule：

```powershell
cd examples
git status --short
git add classic_example
git commit -m "Add generated PolyFEM contact examples"
git push origin jingyao
cd ..
```

这里包含：

```text
classic_example/2D/_contact_2d_common.py
classic_example/3D/_contact_3d_common.py
classic_example/_contact_source_loader.py
classic_example/2D/contact_*_generated_api.py
classic_example/3D/contact_*_generated_api.py
```

## 4. 回到 parent repo 更新 submodule pointers

在 parent repo 根目录运行：

```powershell
git status --short
git submodule status --recursive
git add python-from-jse examples
```

如果这次 release 也包含 parent repo 的 checker/workflow/docs 改动，一起 add：

```powershell
git add tools\run_generated_contact_backend_checks.py
git add tools\check_generated_example_backend.py
git add tools\generated_contact_expected_failures.json
git add .github\workflows\generated-contact-backend.yml
git add TESTING.md
git add doc
git add tests\test_cli.py tests\test_generated_example_backend_tool.py tests\test_release_docs.py
```

然后：

```powershell
python -m pytest tests\test_cli.py tests\test_generated_example_backend_tool.py tests\test_release_docs.py python-from-jse\tests\test_model_builder.py -q
git commit -m "Add generated contact release validation workflow"
git push origin jingyao
```

## 5. 本地 full generated contact backend sweep

backend build 完成后运行：

```powershell
conda activate polyfem
python tools\run_generated_contact_backend_checks.py `
  --output-root build\generated-contact-backend-check-release-candidate `
  --require-tests-match
```

检查：

```text
build/generated-contact-backend-check-release-candidate/summary.txt
build/generated-contact-backend-check-release-candidate/summary.json
build/generated-contact-backend-check-release-candidate/logs
```

必须满足：

```text
PASS: 66
IGNORED: 1
FAIL: 0
Unexpected pass: 0
Unexpected fail: 0
```

如果 `FAIL > 0`，说明有新的真实失败。  
如果 `Unexpected pass > 0`，说明 ignore list 里的 case 已经通过，需要从 `tools/generated_contact_expected_failures.json` 删除。

## 6. GitHub manual full sweep

GitHub Actions 里手动运行：

```text
Generated Contact Backend
```

对应 workflow：

```text
.github/workflows/generated-contact-backend.yml
```

它会自动：

```text
checkout submodules
generate packaged API
build editable backend
run backend-info --require
run active generated contact backend sweep
upload summary.txt / summary.json / logs
```

这个 workflow 是 heavy gate，不是普通 PR gate。

## 7. Fresh clone check

换一个临时目录做 fresh clone：

```powershell
git clone --recurse-submodules https://github.com/peteryu0131/polyfem-python.git polyfem-python-release-check
cd polyfem-python-release-check
git checkout jingyao
git submodule update --init --recursive
git submodule status --recursive
```

至少检查 backend-free workflow：

```powershell
python -m pip install -U pip numpy pytest
python tools\generate_polyfem_api.py
python -m pytest tests -q
```

如果 fresh clone 这里失败，优先检查 submodule commit 是否已经 push 到远端。

## 8. Package artifact check

先安装 build 工具：

```powershell
python -m pip install -U build
```

然后构建：

```powershell
python -m build
```

检查 wheel/sdist 至少包含这些 Python 运行必需内容：

```text
polyfempy/generated_api/
polyfempy/api/
polyfempy/runtime/
polyfempy/cli.py
```

如果 examples 要作为 package 内容发布，还需要额外确认 packaging 配置是否包含 `examples/` 和需要的 data files。这个策略还需要单独决定。

## 9. 最终可汇报状态

可以对老师这样说：

```text
The generated Python API and model-style examples reproduce the active PolyFEM contact examples. The active generated-contact backend gate is expected to report 66 PASS, 1 teacher-approved IGNORED HDF5 proxy-screw case, and 0 unexpected failures. The remaining release checks are submodule push order, fresh-clone validation, and package artifact inspection.
```
