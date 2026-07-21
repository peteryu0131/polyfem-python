# PolyFEM-Python Repo 和 Build 架构说明

这份文档的目的，是用简单的话说明未来几个 repo 应该怎么分工，以及
`polyfem-python` 怎么通过一个清楚的 build 流程把这些 repo 接起来。

前半部分可以直接给老师讲，重点是 repo 边界和 build 流程。最后的
Maintainer Appendix 是给以后维护者看的，里面会写更具体的命令和路径。

## 结论

当前方向是对的：

- `polyfem-python` 负责 Python binding 和 Python package。
- `python-from-jse` 负责从 JSON spec 生成 Python config API。
- `polyfem-data` 负责 meshes、source JSON examples、测试数据。
- `examples` 负责用户可以直接运行和学习的示例。

未来这些目录不应该都作为普通文件直接 commit 到 `polyfem-python`。
更清楚的做法是：`polyfem-python` 作为主 repo，通过 Git submodule 或
明确的 bootstrap 脚本，把其他 repo 拉到固定位置。

## 当前本地结构

现在 checkout 大概是这样：

```text
polyfem-python/
  polyfempy/
    runtime/
    generated_api/
    differentiable/
  src/
  generator-config/
  tools/
  cmake/
  python-from-jse/
  polyfem-data/
  examples/
```

这里的重点是：

- `polyfempy/runtime/` 是手写 solve/runtime 层。
- `polyfempy/generated_api/` 是用户写 config 的 generated API 输出。
- `polyfempy/api/` 和 `polyfempy/generated/` 已经删除，不再作为兼容入口。
- `src/` 是 C++ binding 源码。
- `generator-config/` 是 PolyFEM Python API 的配置。
- `python-from-jse/` 是通用 generator，不是 PolyFEM Python 专属代码。
- `polyfem-data/` 是数据，不应该混在 Python package 源码里。
- `examples/` 是用户示例，可以独立维护。

## Python Package 内部分层

`polyfempy/` 里面现在分成三层：

```text
polyfempy/
  runtime/        手写 solve/runtime 层
  generated_api/  generator 输出，用户写 config 的 API
  differentiable/ 旧 experimental/reference code
```

推荐用户 import：

```python
from polyfempy.runtime import solve
from polyfempy.generated_api import generated_api as polyfem
```

旧入口不再保留：

```python
from polyfempy.api import solve
from polyfempy.generated import generated_api
```

## 每个 Repo 的职责

### `polyfem-python`

这是 Python binding 的主 repo。

它应该包含：

- `polyfempy/`
- `src/`
- `setup.py`
- `pyproject.toml`
- `CMakeLists.txt`
- `cmake/`
- `generator-config/`
- `tools/generate_polyfem_api.py`
- tests that belong to the Python binding

它不应该长期普通 commit：

- 完整的 `polyfem-data/`
- 完整的 external examples repo
- `python-from-jse/generated/`
- `polyfempy/generated_api/`，除非 release packaging 明确决定要 commit generated output

`polyfempy/generated_api/` 是打包时需要的 Python API，但它是生成结果，
不是手写源码。是否 commit 这个目录，需要单独决定。

### `python-from-jse`

这是通用 JSON-spec-to-Python generator。

它应该包含：

- generator core code
- `json-specs/`
- generic examples
- validators
- standalone tests
- `tools/generate_with_overrides.py`

`json-specs/` 放在这里是合理的。原因是 PolyFEM 主 repo 自己的 CMake 也把
JSON specs 当作 build 输入，例如它会从 `json-specs/input-spec.json`、
`json-specs/opt-input-spec.json`、`json-specs/objective-spec.json` 生成 embedded
spec headers。

所以 JSON spec 不是 Python binding 私有配置。它是 generator 支持的输入格式。

### `generator-config`

这个目录应该留在 `polyfem-python`。

原因是这里的文件不是通用 schema，而是 PolyFEM Python API 的暴露策略：

- 哪些名字要 alias
- 哪些 builder relationship 要支持
- `polyfem.model` 这种 Python 用户 API 怎么组织

换句话说：

```text
json-specs/       = 后端真实配置结构
generator-config/ = Python API 怎么变得更好用
```

### `polyfem-data`

这个 repo 应该负责数据：

- meshes
- source JSON examples
- expected outputs
- large test assets

它不适合普通 commit 到 `polyfem-python`，因为数据体积大，而且变化节奏和
Python binding 不一样。

### `examples`

这个 repo 应该负责用户示例：

- classic examples
- generated API examples
- tutorial scripts
- notebook examples, if needed

`polyfem-python` 可以保留很小的 smoke examples，但完整示例集最好独立。

## Submodule、CPM、FetchContent 的区别

这三个东西解决的问题不一样。

### Git submodule

Submodule 适合管理“人需要看、需要改、需要固定版本”的 repo。

适合用 submodule 的目录：

- `python-from-jse/`
- `polyfem-data/`
- `examples/`

优点：

- clone 后目录结构清楚。
- 每个 repo 有自己的历史。
- `polyfem-python` 可以固定每个 repo 的 commit。
- 老师或维护者能明确看到依赖版本。

缺点：

- clone 时需要：

```bash
git clone --recurse-submodules <polyfem-python-url>
```

或者 clone 后运行：

```bash
git submodule update --init --recursive
```

### CPM / FetchContent

CPM 和 FetchContent 适合 CMake 在 build 时拉 C++ dependency。

当前 `polyfem-python` 已经在 CMake 里通过 CPM 拉 PolyFEM C++ 本体：

```cmake
CPMAddPackage("gh:polyfem/polyfem#e8bd3d3")
```

当前 `polyfem-data` 则通过 FetchContent 逻辑下载到数据目录。

这对 C++ dependency 是合理的，但对 `python-from-jse` 这种 Python generator
不一定最清楚。因为 generator 是开发者会直接看和改的工具，用 submodule 更容易讲清楚。

## 推荐的未来结构

推荐结构：

```text
polyfem-python/
  polyfempy/
  src/
  cmake/
  generator-config/
  tools/
    generate_polyfem_api.py
    bootstrap_repos.py
  python-from-jse/   # submodule
  polyfem-data/      # submodule
  examples/          # submodule, optional
```

这个结构里：

- Python binding repo 是主入口。
- `python-from-jse` 是工具依赖。
- `polyfem-data` 是数据依赖。
- `examples` 是示例依赖。
- C++ PolyFEM 本体仍然可以由 CMake/CPM 固定版本拉取。

## 推荐的一键准备流程

建议不要让 `setup.py` 静默 clone 其他 repo。

原因：

- package build 应该可重复。
- 网络下载失败时，错误应该清楚。
- `setup.py` 里做 git clone 会让 pip build 变得难调试。

更好的方式是加一个明确的 bootstrap 脚本，例如：

```bash
python tools/bootstrap_repos.py
python tools/generate_polyfem_api.py --check
python -m pip install -e . --no-build-isolation
```

`bootstrap_repos.py` 可以做这些事：

1. 检查 `.gitmodules` 是否存在。
2. 运行或提示用户运行：

```bash
git submodule update --init --recursive
```

3. 检查关键路径是否存在：

```text
python-from-jse/tools/generate_with_overrides.py
python-from-jse/json-specs/input-spec.json
generator-config/api_aliases.json
generator-config/id_relationships.json
polyfem-data/
```

4. 如果路径缺失，给出明确错误，而不是让 CMake 或 Python import 在后面失败。

## Build 流程建议

推荐分成三个阶段。

### 1. 拉 repo

```bash
git clone --recurse-submodules <polyfem-python-url>
cd polyfem-python
```

如果已经 clone：

```bash
git submodule update --init --recursive
```

### 2. 生成 Python API

```bash
python tools/generate_polyfem_api.py --check
```

这个命令负责：

- 从 `python-from-jse/json-specs/input-spec.json` 读 schema
- 合并 `generator-config/` 里的 PolyFEM Python API config
- 输出到 `polyfempy/generated_api/`
- 运行 backend-free checks

### 3. 编译 Python binding

```bash
python -m pip install -e . --no-build-isolation
```

这个命令通过 `setup.py` 调 CMake，最后编译出：

```text
polyfempy/polyfempy*.pyd   # Windows
polyfempy/polyfempy*.so    # Linux/macOS
```

注意：当前 `setup.py` 只负责 build extension。它不应该静默 clone
`python-from-jse`、`polyfem-data` 或 `examples`，也不应该偷偷生成
`polyfempy/generated_api/`。这些动作应该放在明确的 bootstrap/generation
步骤里。

## Packaging 需要单独决定的点

`polyfempy/generated_api/` 是 generated artifact。

这里先不拍板。我们把选择和 tradeoff 写清楚，再和老师确认。

### 选择 A：不 commit generated output

这种方式更干净。

要求：

- build package 前必须运行 `tools/generate_polyfem_api.py`
- CI 和 release workflow 要明确生成它
- sdist/wheel 构建脚本要保证 generated files 已经存在

### 选择 B：commit generated output

这种方式更容易发 package。

代价：

- generated 文件会出现在 git diff 里
- generator 变化后必须记得重新生成并一起提交

当前 repo 更像选择 A，因为 `.gitignore` 里忽略了 `polyfempy/generated_api/`。
如果未来要发 wheel，需要确认 release workflow 会先生成它。

### 现在的建议

短期先按选择 A 继续走：不 commit generated output，但 CI/release 必须显式
生成它。等 packaging 策略确定后，再决定要不要切到选择 B。

## 当前需要注意的问题

### 1. CMake 默认 data path 还不一致

当前 `CMakeLists.txt` 默认数据路径是：

```cmake
${CMAKE_CURRENT_SOURCE_DIR}/data/
```

但当前 repo 方向已经是：

```text
polyfem-data/
```

所以后续应该考虑把默认路径改成 `polyfem-data/`，或者要求 build 时显式传：

```bash
-DINPUT_POLYFEMPY_DATA_ROOT=polyfem-data
```

### 2. 现在还没有 `.gitmodules`

当前 checkout 有 `python-from-jse/`、`polyfem-data/`、`examples/`，但根目录没有
`.gitmodules`。

这说明它们现在还不是正式 submodule 结构。后续如果要让老师或其他人 clone 后
自动得到同样结构，需要加 submodule。

### 3. `setup.py` 不应该负责 clone repo

`setup.py` 应该负责 build extension。

它可以做 preflight check，例如：

- `python-from-jse` 不存在就报错
- `generator-config` 不存在就报错
- generated API 不存在就提示先运行 `tools/generate_polyfem_api.py`

但不建议它自己去 git clone。

### 4. C++ extension 名字还可以以后再清理

当前 compiled extension 仍然叫：

```text
polyfempy.polyfempy
```

所以 build 后文件名还是：

```text
polyfempy/polyfempy*.pyd
polyfempy/polyfempy*.so
```

这个名字有点重复。未来可以考虑改成：

```text
polyfempy._core
```

但这不是 repo split 的第一步。当前优先级更高的是把 repo 边界、generated
路径、CI/build 流程讲清楚。

## 给老师讲的时候可以这样说

我们把项目拆成几个 repo，不是为了复杂，而是为了每个 repo 的责任更清楚：

- Python binding 只维护 Python package 和 C++ binding。
- Generator 只维护 JSON spec 到 Python API 的生成逻辑。
- Data repo 只维护 mesh 和测试数据。
- Examples repo 只维护用户示例。

`polyfem-python` 作为主 repo，通过 submodule 固定其他 repo 的版本。
这样一个人 clone 主 repo 后，可以用 `--recurse-submodules` 拉到完整开发环境。

build 时分两步：

1. 用 generator 生成 `polyfempy/generated_api/`。
2. 用 CMake 编译 `polyfempy.polyfempy` extension。

Python package 内部也分成两层：

- `polyfempy.generated_api` 让用户写 config。
- `polyfempy.runtime` 负责 solve、payload preparation、backend dispatch 和 result。

这样既保留了 repo 边界，也让开发者可以从一个主入口完成 build。

## 建议的下一步

1. 保留现在的 repo 边界文档。
2. 和老师确认 `polyfempy/generated_api/` 是否进入 git。
3. 增加 `.gitmodules`，把 `python-from-jse`、`polyfem-data`、`examples` 变成正式 submodule。
4. 增加 `tools/bootstrap_repos.py` 或 PowerShell 版本。
5. 修改 CMake 默认 data path，避免继续默认指向旧的 `data/`。
6. 更新 GitHub Actions，让 CI 使用 submodules 并运行 generated API checks。
7. 后续再考虑把 compiled extension 从 `polyfempy.polyfempy` 改成 `polyfempy._core`。

## Maintainer Appendix

这一段给以后维护者看，不一定要给老师逐条讲。

### 当前推荐 import

```python
from polyfempy.runtime import solve
from polyfempy.generated_api import generated_api as polyfem
```

### 当前 generation 命令

从 `polyfem-python` repo root 运行：

```powershell
python tools\generate_polyfem_api.py
```

带 backend-free checks：

```powershell
python tools\generate_polyfem_api.py --check
```

这个 wrapper 当前等价于：

```powershell
python python-from-jse\tools\generate_with_overrides.py `
  --schema-file python-from-jse\json-specs\input-spec.json `
  --output-file polyfempy\generated_api\generated_class.py `
  --api-output-file polyfempy\generated_api\generated_api.py `
  --manifest-dir polyfempy\generated_api `
  --relationships generator-config\id_relationships.json `
  --api-aliases generator-config\api_aliases.json `
  --model-entry polyfem.model
```

### CI 应该做什么

未来 GitHub Actions 可以按这个顺序：

```text
checkout repo with submodules
install Python test dependencies
python tools/generate_polyfem_api.py --check
python -m pytest tests
```

如果 `polyfempy/generated_api/` 不 commit，那么 CI 和 release workflow 必须
先生成它，再 build sdist/wheel。

### Fresh Clone 后的预期

如果选择不 commit generated output，那么 fresh clone 后可能没有：

```text
polyfempy/generated_api/generated_class.py
polyfempy/generated_api/generated_api.py
polyfempy/generated_api/*_manifest.json
```

这是正常的。维护者需要先运行：

```powershell
python tools\generate_polyfem_api.py
```

再运行测试或 package build。

参考：

- PolyFEM 主仓库 CMake 使用 `json-specs` 生成 embedded spec：
  https://github.com/polyfem/polyfem/blob/main/CMakeLists.txt
- 本 repo 的生成入口：
  `tools/generate_polyfem_api.py`
- 本 repo 的 CMake build 入口：
  `CMakeLists.txt`
- 本 repo 的 Python build 入口：
  `setup.py`
