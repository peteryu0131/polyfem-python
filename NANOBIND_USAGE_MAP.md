# 哪里用了 nanobind（构建层 + C++ 绑定层）

本仓库的核心结论：

- **nanobind 只在“构建层”和“C++ 绑定层”使用**，Python 代码本身不会 `import nanobind`。
- Python 侧应该始终 **`import polyfempy as pf`**；底层是 nanobind 还是 pybind11，由 **编译时开关 `USE_NANOBIND`** 决定。

---

## 1) 构建层：CMake 如何使用 nanobind

### `cmake/recipes/nanobind.cmake`

- 作用：定位/引入 nanobind，并保证依赖（例如 `robin-map`）版本满足 nanobind 要求。
- 关键动作：
  - `python -m nanobind --cmake_dir`
  - `find_package(nanobind CONFIG REQUIRED)`

> 这一步决定了“能不能用 nanobind 编译扩展模块”，但不影响 Python 的导入路径（仍是 `polyfempy`）。

### `CMakeLists.txt` / `src/CMakeLists.txt`

- 作用：决定是否开启 `USE_NANOBIND`，以及用 nanobind 的方式生成扩展模块（例如 `nanobind_add_module(...)` 或类似逻辑）。
- 你可以把它理解成：**这里选择编译后端（nanobind vs pybind11）**。

---

## 2) C++ 绑定层：哪里直接包含/使用 nanobind API

### 关键开关：`USE_NANOBIND`

所有“是否使用 nanobind”的分支都由 `USE_NANOBIND` 控制。

### 核心适配头：`src/binding_wrapper.hpp`

这是整个仓库 C++ 绑定层与 nanobind 的“总闸门”：

- `#ifdef USE_NANOBIND`
  - `#include <nanobind/nanobind.h>` 等
  - `namespace py = nanobind; namespace nb = nanobind;`
  - `#define PY_MODULE(name, m) NB_MODULE(name, m)`
- `#else`
  - `#include <pybind11/...>`
  - `#define PY_MODULE(...) PYBIND11_MODULE(...)`

也就是说：**凡是 `#include "binding_wrapper.hpp"` 的绑定文件，都会随 `USE_NANOBIND` 切换后端**。

### 直接受 `USE_NANOBIND` 影响的绑定实现文件（示例）

这些文件本身不一定直接写 nanobind API，但它们通过 `binding_wrapper.hpp` 统一使用 `py::`（= nanobind 或 pybind11）：

- `src/binding.cpp`（模块入口 `PY_MODULE(polyfempy, m)`）
- `src/state/state.cpp`（`pf.Solver` 的大部分绑定方法，包括 `solve()` / `solve_adjoint()`）
- `src/differentiable/objective.cpp`, `src/differentiable/utils.cpp`（含 `#ifndef USE_NANOBIND` 分支以避免 `pybind11_json`）

---

## 3) Python 层“怎么知道自己现在是不是 nanobind 编译出来的？”

因为 Python 统一导入 `polyfempy`，所以你需要通过模块里的版本信息判断：

- C++ 绑定里定义了 `pf.version()`：
  - 如果 `USE_NANOBIND`：返回 `"polyfempy nanobind backend"`
  - 否则：返回 `"polyfempy pybind11 backend"`

建议运行：

```python
import polyfempy as pf
print(pf.version())
```

---

## 4) 一句话总结

- **用了 nanobind 的地方**：CMake 引入 nanobind + C++ 绑定通过 `binding_wrapper.hpp`（`USE_NANOBIND`）切换到 nanobind。
- **没用 nanobind 的地方**：Python 代码（只 import `polyfempy`，不 import `nanobind`）。

