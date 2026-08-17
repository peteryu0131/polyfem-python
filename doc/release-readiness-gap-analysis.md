# PolyFEM-Python Release Readiness Gap Analysis

更新时间：2026-08-17

## 2026-08-17 已完成的前两步

已经把老师允许先忽略的 backend case 从“口头/文档说明”改成了机器可读配置：

```text
tools/generated_contact_expected_failures.json
```

当前默认只包含：

```text
contact/examples/3D/rigid/proxy/screw.json
```

原因是该 case 在当前 Windows Python backend build 中读取 `screw-coarse-to-screw.hdf5` 时出现 `h5pp` type-size mismatch。老师已经说这个可以先 ignore。

批量工具现在会输出四种状态：

```text
PASS             正常通过
IGNORED          失败了，但在 expected-failures 配置中，属于老师批准的暂时忽略项
FAIL             没有被批准忽略的真实失败
UNEXPECTED_PASS  原来在 ignore list 里，但现在通过了，需要提醒我们移除 ignore
```

发布前更适合看这个口径：

```text
Total active: 67
PASS: 66
IGNORED: 1
FAIL: 0
Unexpected pass: 0
Unexpected fail: 0
```

这样 CI / release gate 不需要靠人工解释 `66 / 67`，而是直接知道剩下的 1 个是批准过的 temporary ignore。

同时已经新增 full generated contact backend manual workflow：

```text
.github/workflows/generated-contact-backend.yml
```

它会在手动触发时：

```text
checkout submodules
generate packaged API
build editable backend
run backend-info --require
run active generated contact backend sweep
upload summary.txt / summary.json / logs
```

这个 workflow 是 manual heavy gate，不建议放进普通 PR gate。

## 当前结论

老师要求的 active contact examples 目前已经基本通过。按当前口径：

- `polyfem/tests/contact_2d.txt` 和 `polyfem/tests/contact_3d.txt` 里没有 `#` 注释的 active cases 是当前测试范围。
- generated examples 已经是 `model = polyfem.model()` 风格。
- 有效结果是 `66 / 67` PASS。
- 剩下的 `contact/examples/3D/rigid/proxy/screw.json` 是 HDF5 / Windows ABI 类型宽度问题，老师说可以先 ignore。
- 如果按老师允许 ignore 的口径，active gate 是 `66 / 66` PASS。

现在主要问题不是 API 能不能生成，也不是 examples 能不能跑，而是上线前工程流程还不够规范：workflow、submodule push、release packaging、测试分层、文档同步都需要整理。

## 当前不满意的地方

### 1. Workflow 还不完整

现在已有：

- `.github/workflows/test.yml`
  - backend-free
  - Linux / macOS / Windows
  - Python 3.10 / 3.11 / 3.12
  - regenerate generated API
  - run `python -m pytest tests -q`

- `.github/workflows/backend.yml`
  - manual trigger only
  - Ubuntu + Python 3.11
  - build editable backend
  - run `python -m polyfempy backend-info --require`
  - run `tests/test_backend_smoke.py`
  - optional one generated example backend diagnostic

仍然不足：

- `backend.yml` 只跑 backend smoke 和一个 optional example diagnostic；full contact sweep 已经拆到单独的 manual workflow。
- 还没有 release / wheel / PyPI workflow。
- release gate 的文字 checklist 还需要整理。
- 目前 README 还没有完全改成用户/维护者可读的正式文档。

### 2. Submodule 更新流程还不够安全

当前 repo 有四个 submodule：

```text
polyfem/
examples/
polyfem-data/
python-from-jse/
```

当前本地状态显示：

```text
 m examples
 m python-from-jse
```

这说明至少 `examples` 和 `python-from-jse` 里面有新 commit 或未提交改动。上线前不能只在 parent repo 里更新 submodule pointer。

正确顺序应该是：

1. 进入 `python-from-jse/`，commit 并 push generator/model-builder 改动。
2. 进入 `examples/`，commit 并 push generated classic examples。
3. 确认 `polyfem/` 和 `polyfem-data/` 是否只是 pinned commit；如果有本地改动，也必须分别 commit/push。
4. 回到 `polyfem-python` parent repo，commit submodule pointer 更新。

风险：

- 如果 parent repo 指向一个没有 push 到 GitHub 的 submodule commit，别人 clone 会失败。
- 如果只更新 `polyfem/` 到最新，但没有同步 `python-from-jse`、`generator-config/` 和 generated API，API 可能不支持新 schema。
- 如果 `examples/` 更新但 parent pointer 没更新，别人不会拿到最新 examples。

### 3. Generated API 的 source-of-truth 需要写清楚

当前事实：

- `polyfem.model()` 是 generated API 入口。
- 生成位置是 `polyfempy/generated_api/generated_api.py`。
- 生成逻辑来自 `python-from-jse/generator/JsonToTreeClass.py`。
- model builder 行为来自 `python-from-jse/generator/model_builder.py`。
- PolyFEM-specific config 在 `generator-config/`。
- classic examples 由 `tools/generate_classic_contact_examples.py` 从 `polyfem-data/contact/examples/...` 生成。

不足：

- README 讲了大方向，但还没有把 maintainer workflow 写成严格步骤。
- 需要明确哪些文件可以手改，哪些不能手改。
- 需要明确 `generated_class.py` / `generated_api.py` 是 generated artifacts，不应该把长期逻辑写进去。
- 需要明确 schema 变动时的同步顺序。

建议补充一个 maintainer section：

```text
polyfem schema update
  -> update polyfem submodule
  -> update/copy linked polysolve specs if needed
  -> update generator-config if API naming/relationships changed
  -> regenerate packaged API
  -> regenerate classic examples
  -> run backend-free tests
  -> run backend smoke
  -> run active generated contact sweep
  -> push submodules first
  -> commit parent submodule pointers
```

### 4. Full contact sweep 现在还是偏手动

当前 full sweep 工具已经存在：

```powershell
python tools\run_generated_contact_backend_checks.py --output-root build\generated-contact-backend-check-YYYYMMDD --require-tests-match
```

当前状态：

- 已经有专门 GitHub workflow：`.github/workflows/generated-contact-backend.yml`。
- 已经有 expected-failure config：`tools/generated_contact_expected_failures.json`。
- Windows path too long 已经修复；旧 summary 里仍然可能显示 `63/67`，但 release candidate 应该重新生成新的 summary。

现在 workflow summary 可以直接显示：

```text
PASS: 66
IGNORED: 1
FAIL: 0
```

这比在文档里解释 `66/67` 更适合上线前 CI。

### 5. Release packaging 还没有闭环

当前 `pyproject.toml` 已经有基础 package metadata：

```toml
name = "polyfempy"
version = "0.8"
requires-python = ">=3.10"
```

不足：

- 没有 wheel build workflow。
- 没有 PyPI / TestPyPI publish workflow。
- 没有 cibuildwheel 配置。
- 没有 release tag 规则。
- 没有说明 release 前是否必须重新生成 API。
- 没有说明 generated API 是否必须被包含进 sdist/wheel。
- 没有说明 submodule content 在 sdist/wheel 里的策略。

需要确认：

- 最终用户是 `pip install polyfempy` 直接拿 wheel，还是主要从 source build？
- wheel 是否要包含 C++ backend binary？
- examples 是否随 package 发布，还是只作为 GitHub examples submodule？
- `polyfem-data` 是否随 package 发布，还是用户运行 examples 时从 repo/submodule 使用？

### 6. 文档还需要从开发状态改成用户/维护者状态

当前 README 明确说：

```text
This README is intentionally minimal while the Python API documentation is being rewritten.
```

这对开发阶段可以，但上线给大家用时不够。

需要补：

- Fresh clone
- Submodule init
- Install from source
- Install from wheel
- Run a minimal solve
- Run generated model-style example
- Regenerate API
- Maintainer update workflow
- Testing workflow
- Known ignored backend case
- Troubleshooting

特别是 Windows：

- 需要说明 conda 环境。
- 需要说明 `--no-build-isolation`。
- 需要说明 CMake / compiler / nanobind。
- 需要说明 backend DLL load failure 怎么检查：

```powershell
python -m polyfempy backend-info
python -m polyfempy backend-info --require
```

## 需要添加的东西

### A. Full Contact Backend Workflow

已新增：

```text
.github/workflows/generated-contact-backend.yml
```

用途：

- 手动触发。
- build backend。
- run active contact generated examples。
- upload `summary.txt` / `summary.json` / `logs`.
- 支持 expected ignored cases。

建议先不要放进 every PR gate，因为太慢。

### B. Expected Failure / Ignore 配置

已新增：

```text
tools/generated_contact_expected_failures.json
```

内容类似：

```json
{
  "ignored": [
    {
      "source": "contact/examples/3D/rigid/proxy/screw.json",
      "reason": "HDF5/h5pp type-size mismatch on current Windows Python backend build; teacher approved ignoring for now."
    }
  ]
}
```

然后 batch tool 输出：

```text
PASS: 66
IGNORED: 1
FAIL: 0
```

### C. Submodule Release Checklist

建议新增到 README 或单独文档：

```text
doc/submodule-release-checklist.md
```

内容包括：

1. `git -C python-from-jse status`
2. `git -C python-from-jse push`
3. `git -C examples status`
4. `git -C examples push`
5. `git submodule status --recursive`
6. parent repo commit submodule pointers
7. fresh clone test

### D. Release Workflow

建议新增：

```text
.github/workflows/release.yml
```

初版可以先做：

- build sdist
- build wheel for source/platform under current runner
- upload artifact
- do not publish automatically

等稳定后再加：

- TestPyPI
- PyPI
- GitHub release assets
- cibuildwheel matrix

### E. Testing.md 更新

`TESTING.md` 需要更新到当前状态：

- active contact list 已经有效 `66/67`。
- `rigid/proxy/screw.json` 是唯一 ignored case。
- path-length testing bug 已修复。
- generated model-style examples 是当前主线。
- full sweep 工具已经存在，应该写成 release gate。

## 建议优化的东西

### 1. 让 full sweep 输出更像 CI summary

现在 `summary.txt` 已经能看，但建议加：

```text
Total active: 67
PASS: 66
IGNORED: 1
FAIL: 0
Unexpected fail: 0
Unexpected pass: 0
```

这样老师和 GitHub Actions 都更容易看懂。

### 2. 将 old raw summary 和 effective result 分开

旧的 `build/generated-contact-backend-check-20260813-model-full/summary.txt` 记录了 path-length 修复前的 false fail。上线前最好跑一轮新的 full sweep，用修复后的 checker 生成干净 summary。

建议命令：

```powershell
conda activate polyfem
python tools\run_generated_contact_backend_checks.py --output-root build\generated-contact-backend-check-release-candidate --require-tests-match
```

如果加入 ignore config，最终应该生成：

```text
PASS 66
IGNORED 1
FAIL 0
```

### 3. 保持 generated files 可重新生成

不要把长期逻辑写进：

```text
polyfempy/generated_api/generated_class.py
polyfempy/generated_api/generated_api.py
```

稳定逻辑应该放在：

```text
python-from-jse/generator/model_builder.py
generator-config/
polyfempy/runtime/
tools/generate_classic_contact_examples.py
```

### 4. Fresh Clone Test

上线前必须做一次 fresh clone test，不能只在当前 dirty workspace 里测试。

建议流程：

```powershell
git clone --recurse-submodules <repo-url> polyfem-python-fresh
cd polyfem-python-fresh
python tools\generate_polyfem_api.py --check
python -m pytest tests -q
```

如果要测 backend：

```powershell
conda activate polyfem
python -m pip install -e . --no-build-isolation -vv
python -m polyfempy backend-info --require
python -m pytest tests\test_backend_smoke.py -q -rs
```

## 推荐优先级

### P0：必须做

- Push `python-from-jse` submodule 改动。
- Push `examples` submodule 改动。
- Parent repo commit submodule pointers。
- 更新 `TESTING.md` 到当前真实状态。
- 加 expected ignore 机制，正式处理 `rigid/proxy/screw.json`。
- 用修复后的 checker 重跑 full active sweep，生成干净 summary。

### P1：上线前应该做

- 新增 release candidate checklist。
- README 从 WIP 状态改成用户可读。
- fresh clone 测试。
- 明确 wheel/source build 发布策略。

### P2：后续优化

- release.yml / TestPyPI / PyPI。
- cibuildwheel。
- 多平台 backend build。
- 自动上传 full sweep summary/logs。
- 统一 examples 文档和 gallery。

## 当前可以对老师汇报的一句话

```text
The generated Python API and model-style examples now reproduce the active PolyFEM contact examples. The active test list is effectively passing after ignoring the teacher-approved HDF5 proxy screw case. Expected-ignore handling and the manual full generated-contact backend workflow are now in place. The remaining work is release engineering: submodule push order, fresh-clone validation, release checklist, README cleanup, and packaging/release documentation.
```
