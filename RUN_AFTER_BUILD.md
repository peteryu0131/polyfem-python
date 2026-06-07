# PolyFEM-Python 运行备忘

这份文档记录在 Compute Canada / Alliance 机器上，源码 build 成功之后如何重新进入环境、验证后端、运行 examples 和 tests。

## 每次开新 terminal 先做这个

先 load module，再 activate venv：

```bash
module load StdEnv/2023 python/3.11.5 cmake
source "$HOME/polyfem_env311/bin/activate"
hash -r

cd "$HOME/scratch/polyfem-python"
```

确认当前 Python 是 venv 里的 Python：

```bash
which python
python --version
```

期望 `which python` 类似：

```text
/home/peteryu/polyfem_env311/bin/python
```

不要直接用系统/module 的裸 `python` 跑这个项目。这个项目需要 Python >= 3.10，并且要和 build 时安装了 `nanobind` 的环境一致。

## 验证安装是否正常

```bash
python -c "import polyfempy as pf; print(pf.cpp_backend_available()); print(pf.cpp_backend_error())"
```

成功时应该看到：

```text
True
None
```

如果第一行是 `False`，说明 Python 包能 import，但 C++ 后端没有正常加载，需要重新 build 或检查错误信息。

## 运行基础 examples

这些 examples 使用仓库自带 mesh，通常不需要 Gmsh：

```bash
python examples/01_forward_solve.py
python examples/02_result_fields.py
```

输出会写到：

```text
examples/runs/
```

如果安装了 PyTorch，也可以跑 differentiable examples：

```bash
python examples/03_shape_gradient.py
python examples/05_parameterized_vertex_map.py
```

## 运行 tests

先跑最小 smoke check：

```bash
python -m pytest tests/test_import_public_api.py
python -m pytest tests/test_backend_smoke.py
```

需要跑完整测试时：

```bash
python -m pytest tests
```

## 需要 mesh / shapely / gmsh 时

如果运行的脚本需要 `shapely`、`geos` 或 mesh 生成工具，开新 terminal 时这样进环境：

```bash
module load StdEnv/2023 python/3.11.5 cmake geos
source "$HOME/polyfem_env311/bin/activate"
hash -r
cd "$HOME/scratch/polyfem-python"
```

如果脚本还需要官方 Gmsh SDK，并且你已经配置了 `$HOME/env_polyfem_gmsh.sh`：

```bash
source "$HOME/env_polyfem_gmsh.sh"
```

然后检查：

```bash
python -c "import gmsh, shapely; print('mesh imports OK')"
```

## 重新 build

如果改了 C++ binding、CMake、`setup.py`，或者后端加载失败，重新 build：

```bash
module load StdEnv/2023 python/3.11.5 cmake
source "$HOME/polyfem_env311/bin/activate"
hash -r
cd "$HOME/scratch/polyfem-python"

export CPM_SOURCE_CACHE="$HOME/scratch/.cache/CPM"
mkdir -p "$CPM_SOURCE_CACHE"

N_THREADS=4 python -m pip install -e . --no-build-isolation
```

第一次 build 会比较久。`FetchContent_Populate(...) is deprecated` 这类 CMake warning 可以忽略，只要最后没有 `CMake Error` 或 `ERROR: Failed building editable`。

## 查看 build 是否还在跑

另开一个 terminal，直接运行：

```bash
pgrep -af 'pip|cmake|make|gmake|ninja|g\+\+|cc1plus'
```

如果看到 `pip install -e .`、`cmake --build`、`gmake`、`g++` 或 `cc1plus`，说明还在 build。

## 常见问题

`which python` 不是 `/home/peteryu/polyfem_env311/bin/python`：

重新执行：

```bash
module load StdEnv/2023 python/3.11.5 cmake
source "$HOME/polyfem_env311/bin/activate"
hash -r
```

`Package 'polyfempy' requires a different Python`：

说明用了 Python 3.9 或更老的环境。使用 `$HOME/polyfem_env311`。

`nanobind not found via pip`：

确认在 venv 里：

```bash
python -m pip show nanobind
```

如果没有：

```bash
python -m pip install nanobind
```

`CPM ... cmake.lock creation failed`：

把 CPM cache 放到 scratch：

```bash
export CPM_SOURCE_CACHE="$HOME/scratch/.cache/CPM"
mkdir -p "$CPM_SOURCE_CACHE"
```
