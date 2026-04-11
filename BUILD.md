# 如何构建 polyfempy（与 build 相关的文件说明）

本文说明：**从源码用 `pip` 编出带 C++ 扩展的 `polyfempy`**、常用命令、以及**和构建相关的仓库文件**各自做什么。

---

## 1. 构建在干什么（一句话）

`pip install -e .` 会调用 `**setuptools`** → 执行 `**setup.py**` → 在临时目录里跑 `**cmake**`（根目录 `**CMakeLists.txt**`）→ 编译出 `**polyfempy.polyfempy**` 扩展（Windows 为 `**polyfempy/polyfempy.pyd**`，Linux 一般为 `**polyfempy/polyfempy*.so**`），并与 `**polyfempy/**` 下的纯 Python 代码组成可 `import` 的包。

---

## 2. 环境准备

### 2.1 通用

- **CMake**：根 `**CMakeLists.txt`** 要求 **≥ 3.25**（见文件内 `REQUIRED_CMAKE_VERSION`）。
- **C++17** 编译器：Windows 上通常为 **Visual Studio Build Tools / MSVC**；Linux 上为 **GCC 或 Clang**。
- **Python**：建议使用仓库测试过的版本（例如 **3.11**）；与当前解释器一致的头文件会参与编译（`setup.py` 里传 `-DPYTHON_EXECUTABLE` / `-DPYTHON_INCLUDE_DIR`）。
- **nanobind**：`pyproject.toml` 的 `[build-system] requires` 中包含；绑定用 **nanobind**（不再走 pybind11）。

### 2.2 Conda（推荐，尤其 Windows）

在目标环境里安装构建期常用依赖（具体包名以你环境为准），例如：`cmake`、`nanobind`、`ninja`（可选）、以及 CMake 可能通过 `**CMAKE_PREFIX_PATH`** 找到的 C++ 库。

---

## 3. 推荐安装命令

在**仓库根目录**执行；**务必用「要装到的那个环境」里的 `python -m pip`**（可先 `conda activate polyfem`，或使用 `conda run -n polyfem ...`）。

```bash
python -m pip install -e . --no-build-isolation
```

### 3.1 为什么要加 `--no-build-isolation`

- **默认**：`pip` 会用**隔离的临时环境**构建，里面往往**没有**你在 conda 里装好的 C++ 依赖，CMake 容易 **找不到** 例如 `tsl-robin-map` 等包。  
- `**--no-build-isolation`**：用**当前**这份 Python 环境来跑 CMake，**能继承 `CONDA_PREFIX` / `CMAKE_PREFIX_PATH`**，与 `pyproject.toml` 顶部注释一致。

### 3.2 可选：并行编译线程数

`setup.py` 会读环境变量 `**N_THREADS**`（否则在 Windows 上对并行度有保守上限）。例如：

```bash
set N_THREADS=4
python -m pip install -e . --no-build-isolation
```

（Linux / macOS 用 `export N_THREADS=4`。）

### 3.3 装完自检

```bash
python -c "import polyfempy as pf; print('cpp_backend', pf.cpp_backend_available()); print(pf.cpp_backend_error())"
```

期望 `**cpp_backend True**`、`cpp_backend_error` 为 `**None**`。

可微相关还需安装 PyTorch，例如：

```bash
python -m pip install "torch>=1.9.0"
```

或 `pip install -e ".[differentiable]"`（见 `pyproject.toml` 的 `[project.optional-dependencies]`）。

---

## 4. 构建类型（Release / Debug）

当前 `**setup.py**` 将 `**CMAKE_BUILD_TYPE` 固定为 `Release**`（避免 Debug/Release 运行时混用等问题）。若要 Debug，需**改 `setup.py` 里传给 CMake 的配置**并全量重编（与导师用的 `scikit-build-core` 里改 `CMAKE_BUILD_TYPE` 是同一类操作，只是入口不同）。

---

## 5. 产物与清理


| 产物 / 目录                                                                  | 说明                                           |
| ------------------------------------------------------------------------ | -------------------------------------------- |
| `**polyfempy/polyfempy.pyd`**（Win）或 `**polyfempy/polyfempy*.so**`（Linux） | C++ 扩展模块；`.gitignore` 通常忽略 `*.pyd` / `*.so`。 |
| `**build/**`（或 setuptools 使用的 `build/temp.*`）                            | CMake 配置与编译中间文件；可删后重装做干净构建。                  |
| `***.egg-info/**`                                                        | 包元数据；可编辑安装会更新。                               |


干净重装（概念上）：删掉 `**build/**` 与错误安装痕迹后，再执行一次 `**pip install -e . --no-build-isolation**`。

---

## 6. 与 build 直接相关的文件（按调用顺序）

### 6.1 Python / 打包入口


| 文件                   | 作用                                                                                                                                                                                                                                       |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `**pyproject.toml**` | 声明 `**[build-system]**`（`setuptools` + `wheel` + `nanobind` + `cmake`）、`**[project]**` 元数据与 `**optional-dependencies**`（如 `differentiable` → `torch`）。                                                                                   |
| `**setup.py**`       | **真正触发 CMake**：定义 `**CMakeExtension('polyfempy.polyfempy')`**、自定义 `**CMakeBuild**`；拼 `**cmake_args**`（Python 路径、`USE_NANOBIND`、`nanobind_DIR`、`CMAKE_BUILD_TYPE=Release`、Windows 输出目录等）；在 `build/temp*` 里 `**cmake` + `cmake --build**`。 |


### 6.2 CMake 根与三方


| 文件                                                                                                            | 作用                                                                                                                                                                                             |
| ------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `**CMakeLists.txt**`（仓库根）                                                                                     | CMake 工程入口：`project(polyfempy)`、C++17、`**nanobind_add_module(polyfempy NB_SHARED)**`、`**add_subdirectory(src)**`、链接 `**polyfem::polyfem**`、设置扩展名后缀（`.pyd` / `Python_EXTENSION_MODULE_SUFFIX`）。 |
| `**cmake/recipes/polyfem.cmake**`                                                                             | 通过 **CPM** 拉取 **上游 PolyFEM**（固定 **commit/tag**）；决定与哪个 C++ 库版本链接。                                                                                                                               |
| `**cmake/recipes/polyfem_data.cmake`**                                                                        | PolyFEM 相关数据/资源依赖（若启用）。                                                                                                                                                                        |
| `**cmake/recipes/polyfem_cpm_cache.cmake**`                                                                   | **CPM** 缓存路径等，加速重复拉依赖。                                                                                                                                                                         |
| `**cmake/recipes/CPM.cmake`**                                                                                 | **CPM** 本体的 CMake 脚本。                                                                                                                                                                          |
| `**cmake/recipes/nanobind.cmake`**                                                                            | 与 **nanobind** 集成相关的 CMake 片段（若被 `include` 使用）。                                                                                                                                                |
| `**cmake/recipes/robin-map`**                                                                                 | 根 `CMakeLists.txt` 内 `**CPMAddPackage(robin-map ...)**`：锁定 **robin-map** 版本，满足 nanobind / 间接依赖要求。                                                                                              |
| `**cmake/Warnings.cmake`**、`cmake/CXXFeatures.cmake`、`cmake/UseColors.cmake`、`cmake/PrependCurrentPath.cmake` | 编译警告、C++ 特性、彩色输出、路径等辅助。                                                                                                                                                                        |


### 6.3 C++ 源码与分目录 CMake


| 路径                                                  | 作用                                                                                                          |
| --------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `**src/**`                                          | nanobind 绑定与封装：`**src/state/state.cpp**`（`Solver` / `State`）、`**src/differentiable/**`、`**src/solver/**` 等。 |
| `**src/CMakeLists.txt**` 及各子目录 `**CMakeLists.txt**` | 把 `**SOURCES**` 编进 `**polyfempy**` 目标。                                                                      |


### 6.4 纯 Python包（不参与编译，但随安装一起部署）


| 路径                                                | 作用                                                           |
| ------------------------------------------------- | ------------------------------------------------------------ |
| `**polyfempy/**`（除 `**polyfempy.pyd` / `.so**` 外） | `**api/**`、`differentiable/`、`__init__.py` 等；与扩展同包名，安装后一起可用。 |


### 6.5 与 build 间接相关


| 文件               | 作用                                                                            |
| ---------------- | ----------------------------------------------------------------------------- |
| `**.gitignore**` | 忽略 `**build/**`、`***.pyd**`、`**_deps/**`、`**CPM_modules/**` 等，避免把编译产物提交进 Git。 |
| `**README.md**`  | 若仓库根有安装说明，可与本文互补。                                                             |


---

## 7. Linux 上与 Windows 的差异（概念）

- 命令相同：`**python -m pip install -e . --no-build-isolation**`（用对 Python）。  
- `**setup.py**` 里对 **MSVC `/m:N`** 与 **Unix `-jN`** 的分支不同；**CMake generator** 由本机默认决定。  
- 扩展后缀：由 `**Python_EXTENSION_MODULE_SUFFIX`** 或回退逻辑决定（`**.so**` 等）。

---

## 8. 常见问题（指向）


| 现象                                                     | 建议                                                                                                      |
| ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------- |
| CMake 找不到 conda 里的包                                    | 使用 `**--no-build-isolation**`，并确认 `**conda run -n <env>` / `activate**` 后再 `pip`。                       |
| `import polyfempy` 成功但 `cpp_backend_available` 为 False | 用了**错误的 Python**（如 base 的 3.13）；或 `**.pyd` 依赖的 DLL** 不在 `PATH`（Windows）。                                |
| 想对齐导师的 **Debug** 构建                                    | 改 `**setup.py`** 中 `**CMAKE_BUILD_TYPE**` 相关逻辑后**全量重编**；与 `**scikit-build-core`** 的 `cmake.args` 是不同入口。 |


更完整的问题记录（可微接触 SIGSEGV、JSON 与 Windows 差异等）见 `**experiment/new_experiment_02/ISSUES_AND_ENVIRONMENT.md**`。

---

*若仓库的 CMake 最低版本或依赖有更新，以根目录 `**CMakeLists.txt*`* 与 `**pyproject.toml**` 为准，并同步修订本节。*