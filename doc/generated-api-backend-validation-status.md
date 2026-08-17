# Generated API Backend Validation Status

## 2026-08-17 Tooling Update

批量 backend checker 现在已经支持 machine-readable expected failures：

```text
tools/generated_contact_expected_failures.json
```

当前只有一个 expected ignored case：

```text
contact/examples/3D/rigid/proxy/screw.json
```

它的失败原因不是 generated API JSON shape，也不是 `Multiple rules matched`，而是当前 Windows Python backend build 读取 `screw-coarse-to-screw.hdf5` 时出现 `h5pp` type-size mismatch。老师已经说这个 case 可以先 ignore。

新的 full sweep summary 口径是：

```text
PASS             generated example 和 source JSON backend comparison 通过
IGNORED          老师批准暂时忽略的失败
FAIL             没有批准忽略的真实失败
UNEXPECTED_PASS  ignore list 里的 case 现在通过了，需要移除 ignore
```

所以发布前应该看：

```text
PASS: 66
IGNORED: 1
FAIL: 0
Unexpected pass: 0
Unexpected fail: 0
```

这比只说 `66 / 67` 更准确，因为它明确区分了“API/测试真的失败”和“老师批准的系统差异 temporary ignore”。

## 会议结论

当前 Python generated API 没有已知的 payload mismatch。老师要求运行的 active contact examples 现在都已经是 `model = polyfem.model()` 风格。

按 2026-08-13 最新 model-style full sweep：raw summary 曾经是 `63 / 67` PASS，其中 3 个 FAIL 是 Windows output path 太长导致的 false fail；现在已经修复测试工具的 output workspace 命名，这 3 个 case 用同样长的 batch root 复跑都 PASS。所以有效结果仍然是 `66 / 67` PASS。唯一有效失败还是 `contact/examples/3D/rigid/proxy/screw.json`，老师已说明可以先 ignore，因为它更像系统 / HDF5 / ABI 差异。

已经修复并验证的 generated/API/runtime/testing 问题有五个：

- `solver.nonlinear.x_delta_tol` / `solver.nonlinear.grad_norm_tol` 被 Python runtime bridge 错误改名的问题。
- `dolphin-funnel` 两个例子里，single material object 被变成 one-item list，导致 generated run 和 source JSON run 结果不同的问题。
- Python runtime 以前用 `strict_validation=False` 调 C++ `Solver.set_settings(...)`，导致 `mesh_array` 和 `collision_mesh` 报 `Multiple rules matched`。现在已改成 `strict_validation=True`，和 PolyFEM C++ tests/command path 对齐。
- `contact.collision_mesh.linear_map` 以前没有按 source JSON 所在目录 resolve，导致 `rigid/proxy/screw.json` 找错 HDF5 文件。现在已和 `mesh` 路径一样处理。
- backend comparison tool 以前把很长的 generated example 文件名接到 output workspace 里，在 Windows 上会触发 path too long。现在 generated/source output workspace 改成短目录 `generated/run` 和 `source-json/run`，相关 false fail 已复跑 PASS。

当前唯一剩余失败：

```text
contact/examples/3D/rigid/proxy/screw.json
```

它现在不再是 `Multiple rules matched`，也不再是找不到 `linear_map` 文件；失败发生在读取 `screw-coarse-to-screw.hdf5` 时：

```text
h5pp: Type size mismatch ... c++ type [class std::array<long,2>]: 4 bytes
```

这个更像 HDF5 文件 / h5pp / Windows C++ ABI 类型宽度问题，不是 generated API JSON shape 问题。

## 当前结果

测试范围来自老师指定的官方 contact test list：

- `polyfem/tests/contact_2d.txt`
- `polyfem/tests/contact_3d.txt`

统计口径：

- 只算没有 `#` 注释的 `contact/examples/...`。
- 跳过 `#` 注释掉的例子。
- 跳过不在这两个 list 里的 polyfem-data 例子。
- 按展开后的 effective JSON 统计，也就是先处理 `common` 和 `patch`，再读 `tests.time_steps`。

当前 active set：

| Set | Count |
| --- | ---: |
| 2D active cases | 25 |
| 3D active cases | 42 |
| Total | 67 |

model-style 覆盖：

| Scope | Result |
| --- | --- |
| 67 active generated examples | 全部包含 `model = polyfem.model()` |
| `contact/examples/3D/static/two-cubes.json` | 不在 active gate；仍是单独的 generated schema 覆盖问题，见下面说明 |

生成来源：

| Item | Source |
| --- | --- |
| `polyfem.model()` | 由 `python-from-jse/generator/JsonToTreeClass.py` 生成到 `polyfempy/generated_api/generated_api.py` |
| model builder 行为 | 来自 `python-from-jse/generator/model_builder.py`，不是手写在每个 example 里 |
| classic examples | 由 `tools/generate_classic_contact_examples.py` 从 `polyfem-data/contact/examples/...` 生成 |

按维度统计的 pass/fail：

| Set | PASS | FAIL |
| --- | ---: | ---: |
| 2D active cases | 25 | 0 |
| 3D active cases | 41 | 1 |
| Total | 66 | 1 |

2026-08-13 最新 model-style rerun 的 raw/effective 区分：

| Result type | PASS | FAIL | Meaning |
| --- | ---: | ---: | --- |
| Old raw full sweep summary | 63 | 4 | `build/generated-contact-backend-check-20260813-model-full/summary.txt` 记录了工具修复前的路径过长 false fail |
| Path-length fix verification | 3 | 0 | 三个 high-school-physics slopetest case 用修复后的 checker 和长 batch root 复跑全部 PASS |
| Effective backend result | 66 | 1 | 修复 path-length false fail 后的真实结果 |
| Effective result after teacher-approved ignore | 66 | 0 | 再忽略 `rigid/proxy/screw.json` 后，active gate 没有剩余失败 |

路径过长 false-fail 已修复并验证的 cases：

| Case | Fixed-tool rerun output |
| --- | --- |
| `contact/examples/2D/friction/high-school-physics-slopetest-mu=0.49.json` | `build/generated-contact-backend-check-pathlength-fixed/runs/2d_friction_high_school_physics_slopetest_mu_0_49` |
| `contact/examples/2D/friction/high-school-physics-slopetest-mu=0.50.json` | `build/generated-contact-backend-check-pathlength-fixed/runs/2d_friction_high_school_physics_slopetest_mu_0_50` |
| `contact/examples/3D/friction/high-school-physics-slopetest-mu=0.49.json` | `build/generated-contact-backend-check-pathlength-fixed/runs/3d_friction_high_school_physics_slopetest_mu_0_49` |

time step 设置：

| Run length | Count | Meaning |
| --- | ---: | --- |
| `1 step` | 34 | JSON 写了 `tests.time_steps = 1`，或动态例子没有写 `tests.time_steps` 时默认跑 1 步 |
| `all` | 33 | JSON 写了 `tests.time_steps = "all"`，按完整 time steps 跑 |
| `static` | 0 | active list 里没有真正 static 的 case |

结果口径：

| Category | Count | Note |
| --- | ---: | --- |
| PASS | 66 | full sweep 后 generated example vs source JSON backend comparison 通过 |
| FAIL | 1 | `contact/examples/3D/rigid/proxy/screw.json`，HDF5 type mismatch |
| `Multiple rules matched` | 0 | strict validation 对齐后不再出现 |

严格按 backend pass/fail 算，当前是 `66 / 67 = 98.51%`。

结果文件：

- 最新 model-style full sweep raw result，记录了 path-length 修复前的 3 个 false fail：
  - `build/generated-contact-backend-check-20260813-model-full/summary.txt`
  - `build/generated-contact-backend-check-20260813-model-full/summary.json`
- path-length fix verification，使用修复后的 checker 和同样长的 batch root 复跑：
  - `build/generated-contact-backend-check-pathlength-fixed/runs/2d_friction_high_school_physics_slopetest_mu_0_49`
  - `build/generated-contact-backend-check-pathlength-fixed/runs/2d_friction_high_school_physics_slopetest_mu_0_50`
  - `build/generated-contact-backend-check-pathlength-fixed/runs/3d_friction_high_school_physics_slopetest_mu_0_49`
- 上一轮 historical full sweep:
  - `build/generated-contact-backend-check-20260812-full/summary.txt`
  - `build/generated-contact-backend-check-20260812-full/summary.json`
- 每个 case 的完整 stdout/stderr 在对应 output root 的 `logs/` 或单 case stdout 里。

## 当前唯一失败

| Case | Error | Current interpretation |
| --- | --- | --- |
| `contact/examples/3D/rigid/proxy/screw.json` | `h5pp: Type size mismatch ... std::array<long,2>: 4 bytes` | 已经越过 JSON validation 和 path resolution；失败发生在 backend 读取 `screw-coarse-to-screw.hdf5` 的 HDF5 attribute 时 |

大白话解释：

- 之前 `pile/cubes.json`、`pile/octocat-bowl.json`、`rigid/proxy/screw.json` 的 `Multiple rules matched` 是 Python runtime 用错 validation mode：我们用了 `strict_validation=False`，而老师的 C++ tests/command path 用 strict validation。
- 改成 `strict_validation=True` 后，`pile/cubes.json` 和 `pile/octocat-bowl.json` 已经 PASS。
- `rigid/proxy/screw.json` 现在的问题已经不是 JSON checker，而是 HDF5 文件读取时的 C++ type size mismatch。

## 已修复的问题

### 1. Nonlinear Solver 字段名

涉及文件：

- `polyfempy/runtime/_solve_contract.py`
- `polyfem-data/contact/examples/3D/rigid/screw.json`
- `examples/classic_example/3D/contact_3d_rigid_screw_generated_api.py`

问题：

- 当前 polysolve schema 使用 `x_delta_tol` 和 `grad_norm_tol`。
- 旧的 Python runtime bridge 会把它们错误改成 `x_delta` 和 `grad_norm`。
- backend 不认识改名后的字段，所以这是 Python runtime bridge 的问题。

修复：

- 删除这两个 root-level nonlinear solver field rename。
- 不手改 `polyfempy/generated_api/generated_class.py`。
- generated files 仍然可以重新生成；稳定逻辑放在 handwritten runtime bridge。

验证结果：

- `3D/rigid/screw.json` 对应 generated example 可以进 backend。
- generated output 和 source JSON output 的 `err_*` 指标在 tolerance 内一致。

### 2. Dolphin Material Shape

涉及文件：

- `polyfempy/runtime/_solve_contract.py`
- `tests/test_generated_payload.py`
- `polyfem-data/contact/examples/3D/stress-tests/dolphin-funnel.json`
- `polyfem-data/contact/examples/3D/stress-tests/dolphin-funnel-Linf.json`

问题：

- source JSON 是 single material object。
- generated runtime payload 旧逻辑会把它变成 one-item list。
- 这两个 shape backend 都可能接受，但求解结果不同，所以这是 payload semantics 问题。

修复：

- generated payload 里，如果 `materials` 是单个、没有 `id` 的 material list，就恢复成 source JSON 的 object shape。
- 如果 material list 里有 `id`，继续保持 list，不破坏 multi-material case。
- JSON direct path 不再强行把 material object promote 成 list。

验证命令：

```powershell
conda activate polyfem

python tools\check_generated_example_backend.py --example examples\classic_example\3D\contact_3d_stress_tests_dolphin_funnel_generated_api.py --source-json polyfem-data\contact\examples\3D\stress-tests\dolphin-funnel.json --output-root build\dolphin-funnel-backend-check-20260809 --generated-source-tolerance 1e-5 --require-tests-match

python tools\check_generated_example_backend.py --example examples\classic_example\3D\contact_3d_stress_tests_dolphin_funnel_linf_generated_api.py --source-json polyfem-data\contact\examples\3D\stress-tests\dolphin-funnel-Linf.json --output-root build\dolphin-funnel-linf-backend-check-20260809 --generated-source-tolerance 1e-5 --require-tests-match
```

验证结果：

```text
dolphin-funnel:      PASS, generated vs source err_* diff = 0
dolphin-funnel-Linf: PASS, generated vs source err_* diff = 0
```

### 3. Backend Check Output Path Too Long

涉及文件：

- `tools/check_generated_example_backend.py`
- `tests/test_generated_example_backend_tool.py`

问题：

- single-case backend checker 以前把 generated example 的完整文件名作为 output workspace 的最后一级目录。
- 对短 example 没问题，但 high-school-physics slopetest 这类名字很长；再叠加 Windows repo path、batch output root、case name 后，会触发 `WinError 206`。
- 这个失败发生在 `workspace.mkdir(...)`，simulation 还没开始，所以不是 generated API、source JSON、backend solver、或者 test expected result 的问题。

修复：

- generated run 的 workspace 从 `generated/<example_stem>` 改成 `generated/run`。
- source JSON run 的 workspace 从 `source-json/<source_stem>` 改成 `source-json/run`。
- 每个 case 本来就在 batch tool 的独立 case directory 下运行，所以不会互相覆盖。

验证：

```powershell
python -m pytest tests\test_generated_example_backend_tool.py python-from-jse\tests\test_model_builder.py -q
```

结果：

```text
37 passed
```

并且用修复后的 checker 复跑之前 path too long 的三个 case，全部 PASS：

```text
contact/examples/2D/friction/high-school-physics-slopetest-mu=0.49.json
contact/examples/2D/friction/high-school-physics-slopetest-mu=0.50.json
contact/examples/3D/friction/high-school-physics-slopetest-mu=0.49.json
```

## 老师明确可以忽略的范围

老师说 active list 里 `#` 开头的可以 ignore。所以这些不作为当前 generated backend validation gate：

- `#contact/examples/2D/friction/arch.json`
- `#contact/examples/2D/static/friction-slope.json`
- `#contact/examples/3D/static/two-cubes.json`
- `#contact/examples/3D/higher-order/golf-ball-P1.json`
- `#contact/examples/3D/higher-order/golf-ball-P2.json`
- `#contact/examples/3D/golf-ball.json`
- `#contact/examples/3D/friction/armadillo-roller.json`
- `#contact/examples/3D/stress-tests/trash-compactor-shapes.json`
- `#contact/examples/3D/stress-tests/squeeze-out.json`
- `#contact/examples/3D/higher-order/ball-bounce/P4-dt=0.01.json`

也可以先不作为当前 gate 的例子：

- `polyfem-data/contact/examples/2D/codimensional/pin-cushion-ball.json`
- `polyfem-data/contact/examples/3D/codimensional/pin-cushion-ball.json`
- `polyfem-data/contact/examples/3D/higher-order/ball-bounce/P1.json`
- `polyfem-data/contact/examples/3D/higher-order/microstructure.json`

`pin-cushion-ball.json` 虽然有 generated example，但它不在当前 active `contact_2d.txt` / `contact_3d.txt` 里，所以这轮不用当作失败汇报。

`two-cubes.json` 里的 `solver.nonlinear.use_grad_norm` 也不在当前 active gate，因为 `3D/static/two-cubes.json` 在 `contact_3d.txt` 里被 `#` 注释掉了。这个仍然可以记录为 source JSON / polysolve schema 对齐问题：当前 generated schema 里有 `x_delta_tol`、`grad_norm_tol`、`line_search.use_grad_norm_tol`，但没有 root-level `solver.nonlinear.use_grad_norm`。

## Backend Validator 边界

这里的 validator 不是 `python-from-jse` generator 写的，也不是 generated API class 写的。

当前调用链是：

```text
generated Python API
  -> Python dict / JSON payload
  -> polyfempy.runtime.solve(cfg=...)
  -> C++ binding Solver.set_settings(...)
  -> polyfem::State::init(...)
  -> jse.verify_json(args_in, rules)
```

相关位置：

- `src/state/state.cpp`: Python `Solver.set_settings(...)` 把 Python 传入的 JSON 字符串 parse 成 C++ json，然后调用 `state_.init(...)`。
- `polyfem/src/polyfem/State.cpp`: `State::init(...)` 里调用 `jse.verify_json(args_in, rules)`。
- `polyfem/CMakeLists.txt`: build PolyFEM 时 embed `polyfem/json-specs/input-spec.json` 和 linked polysolve specs。

这次 `Multiple rules matched` 的结论已经变清楚：不是 generated API 生成的 JSON 错，而是 Python runtime 之前用 `strict_validation=False` 调了 `Solver.set_settings(...)`。老师的 C++ contact tests/command path 使用的是 strict validation。Python runtime 改成 `strict_validation=True` 后，当前 active list 里已经没有 `Multiple rules matched` 失败。

## Test Workflow

### 修改 Python runtime / generated API 逻辑后

先跑 Python tests：

```powershell
conda activate polyfem
python -m pytest tests -q
```

已验证结果：

```text
178 passed, 3 skipped
```

相关 focused tests：

```powershell
conda activate polyfem
python -m pytest tests\test_generated_payload.py tests\test_pipeline_normalize.py tests\test_generated_api_example.py tests\test_generated_example_backend_tool.py -q
```

已验证结果：

```text
83 passed
```

不要用 repo root 的 `python -m pytest -q` 作为这个工作的主要信号，因为它会额外 collect `python-from-jse/tests`，当前会遇到 generator repo 的 import path 问题。这不是这次 generated API runtime fix 引入的问题。

### 修一个具体 backend case 后

用 single-case backend comparison tool：

```powershell
conda activate polyfem
python tools\check_generated_example_backend.py --example <generated-example.py> --source-json <polyfem-data-json> --output-root build\<case-name>-backend-check --generated-source-tolerance 1e-5 --require-tests-match
```

这个工具做的事：

- 展开 source JSON 的 `common`。
- 应用 source JSON 的 `patch`。
- 按 `tests.time_steps` 决定跑 1 步还是 all。
- 删除 `tests` / `default_params`，生成 backend payload。
- 对 generated example 和 source JSON payload 都跑 backend。
- 比较两边 `sim.json` 里的 `err_*`。
- 默认删除 `.vtu` / `.vtm` / `.pvd` 大文件，只保留需要对比的输出；如果要保留可加 `--keep-visual-output`。

### 什么时候跑 full active sweep

平时修单个问题，先跑对应 single-case check。

准备汇报、合并、或者改了 shared runtime bridge 时，需要对 `polyfem/tests/contact_2d.txt` 和 `polyfem/tests/contact_3d.txt` 的 active cases 做 full sweep。现在已经有批量工具，不需要手写 test list：

```powershell
conda activate polyfem
python tools\run_generated_contact_backend_checks.py --output-root build\generated-contact-backend-check-20260812-full --require-tests-match
```

这个批量工具做的事：

- 读取 `polyfem/tests/contact_2d.txt` 和 `polyfem/tests/contact_3d.txt`。
- 只运行没有 `#` 注释、并且是 `contact/examples/...` 的 active examples。
- 自动找到对应的 generated Python example。
- 对每个 case 调 `tools/check_generated_example_backend.py`。
- 每个 case 的 stdout/stderr 写入 `logs/*.log`。
- 总结写入 `summary.txt` 和 `summary.json`。

当前这次结果已经保存到：

- `build/generated-contact-backend-check-20260812-full/summary.txt`
- `build/generated-contact-backend-check-20260812-full/summary.json`

## 需要问老师的问题

可以用这段简单英文问：

```text
I reran the active contact tests from polyfem/tests/contact_2d.txt and contact_3d.txt through the Python generated API.
The current result is 66 / 67 pass.

The only remaining failure is:
contact/examples/3D/rigid/proxy/screw.json

It no longer fails in JSON validation.
It also finds the linear_map file now.
The failure happens when h5pp reads screw-coarse-to-screw.hdf5:

h5pp: Type size mismatch ... c++ type [class std::array<long,2>]: 4 bytes

Is this HDF5 file expected to work on Windows through the Python backend build?
Or is this a platform / h5pp type-size issue rather than a generated API issue?
```

重点不是问“generator 怎么修”，而是问：

- `screw.json` 这个 HDF5 proxy example 是否应该在你的 Windows Python backend build 上能跑。
- 如果老师能在 C++ command line 跑通，而 Python backend 仍然报这个 HDF5 type mismatch，要确认是不是平台或 build ABI 差异。
- 当前不需要再把 `Multiple rules matched` 当作主要问题问老师，因为 active list 里这个问题已经消失。

## Active Test List

下面是当前 active list 里真正参与统计的 67 个例子。line 是它们在 `polyfem/tests/contact_2d.txt` 或 `polyfem/tests/contact_3d.txt` 里的行号。

### 2D Active Cases

| Line | Source JSON | Run length | Reason |
| ---: | --- | --- | --- |
| 1 | `contact/examples/2D/large-ratios/large-stiffness-ratio.json` | 1 step | default: no `tests.time_steps` |
| 2 | `contact/examples/2D/large-ratios/circle-mat.json` | 1 step | default: no `tests.time_steps` |
| 3 | `contact/examples/2D/large-ratios/large-mass-ratio.json` | all | `tests.time_steps = "all"` |
| 4 | `contact/examples/2D/initial_angular_velocity.json` | all | `tests.time_steps = "all"` |
| 5 | `contact/examples/2D/golf-ball-doformable-wall.json` | 1 step | default: no `tests.time_steps` |
| 6 | `contact/examples/2D/golf-ball.json` | 1 step | default: no `tests.time_steps` |
| 7 | `contact/examples/2D/golf-ball-Linf.json` | 1 step | default: no `tests.time_steps` |
| 8 | `contact/examples/2D/codimensional/disk-codim-points.json` | 1 step | default: no `tests.time_steps` |
| 9 | `contact/examples/2D/friction/rotating-slope.json` | 1 step | explicit `tests.time_steps = 1` |
| 10 | `contact/examples/2D/friction/high-school-physics-slopetest-mu=0.49.json` | 1 step | explicit `tests.time_steps = 1` |
| 11 | `contact/examples/2D/friction/circle-rollers.json` | 1 step | default: no `tests.time_steps` |
| 12 | `contact/examples/2D/friction/moving-ground.json` | 1 step | explicit `tests.time_steps = 1` |
| 13 | `contact/examples/2D/friction/high-school-physics-slopetest-mu=0.50.json` | all | `tests.time_steps = "all"` |
| 15 | `contact/examples/2D/friction/card-house.json` | 1 step | default: no `tests.time_steps` |
| 16 | `contact/examples/2D/unit-tests/erleben/sliding-spike.json` | all | `tests.time_steps = "all"` |
| 17 | `contact/examples/2D/unit-tests/erleben/spikes.json` | all | `tests.time_steps = "all"` |
| 18 | `contact/examples/2D/unit-tests/erleben/internal-edges.json` | all | `tests.time_steps = "all"` |
| 19 | `contact/examples/2D/unit-tests/erleben/spike-in-crack.json` | all | `tests.time_steps = "all"` |
| 20 | `contact/examples/2D/unit-tests/erleben/cliff-edges.json` | all | `tests.time_steps = "all"` |
| 21 | `contact/examples/2D/unit-tests/vertex-edge.json` | all | `tests.time_steps = "all"` |
| 22 | `contact/examples/2D/unit-tests/5-squares.json` | 1 step | default: no `tests.time_steps` |
| 23 | `contact/examples/2D/unit-tests/vertex-vertex.json` | all | `tests.time_steps = "all"` |
| 24 | `contact/examples/2D/unit-tests/edge-vertex.json` | all | `tests.time_steps = "all"` |
| 25 | `contact/examples/2D/unit-tests/triangle-corner.json` | all | `tests.time_steps = "all"` |
| 26 | `contact/examples/2D/unit-tests/edge-edge.json` | all | `tests.time_steps = "all"` |

### 3D Active Cases

| Line | Source JSON | Run length | Reason |
| ---: | --- | --- | --- |
| 2 | `contact/examples/3D/large-ratios/sphere-mat.json` | 1 step | default: no `tests.time_steps` |
| 3 | `contact/examples/3D/large-ratios/large-stiffness-ratio.json` | 1 step | default: no `tests.time_steps` |
| 4 | `contact/examples/3D/large-ratios/large-mass-ratio.json` | all | `tests.time_steps = "all"` |
| 8 | `contact/examples/3D/codimensional/mat-knives.json` | 1 step | default: no `tests.time_steps` |
| 9 | `contact/examples/3D/friction/ball-rollers.json` | 1 step | default: no `tests.time_steps` |
| 11 | `contact/examples/3D/friction/high-school-physics-slopetest-mu=0.49.json` | 1 step | explicit `tests.time_steps = 1` |
| 12 | `contact/examples/3D/friction/high-school-physics-slopetest-mu=0.50.json` | all | `tests.time_steps = "all"` |
| 13 | `contact/examples/3D/friction/arch.json` | 1 step | default: no `tests.time_steps` |
| 14 | `contact/examples/3D/friction/stick-slip.json` | 1 step | default: no `tests.time_steps` |
| 15 | `contact/examples/3D/friction/card-house.json` | 1 step | default: no `tests.time_steps` |
| 16 | `contact/examples/3D/rigid/screw.json` | 1 step | explicit `tests.time_steps = 1` |
| 17 | `contact/examples/3D/unit-tests/erleben/sliding-wedge.json` | all | `tests.time_steps = "all"` |
| 18 | `contact/examples/3D/unit-tests/erleben/spike-in-hole.json` | all | `tests.time_steps = "all"` |
| 19 | `contact/examples/3D/unit-tests/erleben/sliding-spike.json` | all | `tests.time_steps = "all"` |
| 20 | `contact/examples/3D/unit-tests/erleben/spikes.json` | all | `tests.time_steps = "all"` |
| 21 | `contact/examples/3D/unit-tests/erleben/spike-and-wedge.json` | all | `tests.time_steps = "all"` |
| 22 | `contact/examples/3D/unit-tests/erleben/internal-edges.json` | all | `tests.time_steps = "all"` |
| 23 | `contact/examples/3D/unit-tests/erleben/wedge-in-crack.json` | all | `tests.time_steps = "all"` |
| 24 | `contact/examples/3D/unit-tests/erleben/spike-in-crack.json` | all | `tests.time_steps = "all"` |
| 25 | `contact/examples/3D/unit-tests/erleben/wedges.json` | all | `tests.time_steps = "all"` |
| 26 | `contact/examples/3D/unit-tests/erleben/cliff-edges.json` | 1 step | explicit `tests.time_steps = 1` |
| 27 | `contact/examples/3D/unit-tests/erleben/spike-in-hole-rigid.json` | all | `tests.time_steps = "all"` |
| 28 | `contact/examples/3D/unit-tests/erleben/wedge-in-crack-rigid.json` | all | `tests.time_steps = "all"` |
| 29 | `contact/examples/3D/unit-tests/erleben/spike-in-crack-rigid.json` | all | `tests.time_steps = "all"` |
| 30 | `contact/examples/3D/unit-tests/vertex-face.json` | all | `tests.time_steps = "all"` |
| 31 | `contact/examples/3D/unit-tests/5-cubes.json` | 1 step | default: no `tests.time_steps` |
| 32 | `contact/examples/3D/unit-tests/vertex-vertex.json` | all | `tests.time_steps = "all"` |
| 33 | `contact/examples/3D/unit-tests/edge-edge.json` | all | `tests.time_steps = "all"` |
| 34 | `contact/examples/3D/unit-tests/face-vertex.json` | all | `tests.time_steps = "all"` |
| 35 | `contact/examples/3D/unit-tests/edge-edge-parallel.json` | all | `tests.time_steps = "all"` |
| 36 | `contact/examples/3D/unit-tests/tet-corner.json` | 1 step | explicit `tests.time_steps = 1` |
| 37 | `contact/examples/3D/unit-tests/5-cubes-fast.json` | all | `tests.time_steps = "all"` |
| 38 | `contact/examples/3D/stress-tests/trash-compactor-octocat.json` | 1 step | default: no `tests.time_steps` |
| 39 | `contact/examples/3D/stress-tests/trash-compactor-octocat-Linf.json` | 1 step | default: no `tests.time_steps` |
| 41 | `contact/examples/3D/stress-tests/rod-twist.json` | 1 step | default: no `tests.time_steps` |
| 43 | `contact/examples/3D/stress-tests/dolphin-funnel.json` | 1 step | explicit `tests.time_steps = 1` |
| 44 | `contact/examples/3D/stress-tests/mat-twist.json` | 1 step | default: no `tests.time_steps` |
| 45 | `contact/examples/3D/stress-tests/dolphin-funnel-Linf.json` | 1 step | explicit `tests.time_steps = 1` |
| 46 | `contact/examples/3D/stress-tests/mat-twist-Linf.json` | 1 step | default: no `tests.time_steps` |
| 48 | `contact/examples/3D/rigid/proxy/screw.json` | 1 step | explicit `tests.time_steps = 1` |
| 49 | `contact/examples/3D/pile/cubes.json` | 1 step | explicit `tests.time_steps = 1` |
| 50 | `contact/examples/3D/pile/octocat-bowl.json` | 1 step | explicit `tests.time_steps = 1` |
