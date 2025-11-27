# PolyFEM Python API 架构说明

本文档详细说明 `polyfempy/api/` 文件夹下的所有文件，包括设计理念、函数意义和示例部分。

## 目录结构

```
polyfempy/api/
├── __init__.py              # 模块入口，导出主要 API
├── solve.py                 # 主求解函数（统一入口）
├── config.py                # 配置类（SimulationConfig）
├── result.py                # 结果容器（Result）
├── errors.py                # 错误处理（统一错误模型）
├── backend_base.py          # 后端 SPI 定义（接口契约）
├── backend_dummy.py         # Dummy 后端实现（测试用）
├── backend_nanobind.py      # Nanobind 后端适配器（C++ 连接）
├── batch.py                 # 批量处理（batch_solve）
├── tensor.py                # 张量转换（多后端支持）
└── examples/                # API 示例代码
    ├── run_dummy_elasticity.py
    ├── run_elasticity.py
    ├── load_from_json.py
    ├── parameter_sweep.py
    ├── batch_processing.py
    ├── with_callbacks.py
    └── README.md
```

---

## 1. `__init__.py` - 模块入口

### 设计理念

**单一入口原则**：只导出用户需要的主要 API，隐藏实现细节。

### 导出内容

```python
from .solve import solve
from .config import SimulationConfig
from .result import Result
from .selection import Selection
from .batch import batch_solve

__all__ = ["solve", "SimulationConfig", "Result", "Selection", "batch_solve"]
```

### 函数意义

- **`solve`**: 主求解函数，用户的主要入口
- **`SimulationConfig`**: 配置类，用于创建仿真配置
- **`Result`**: 结果容器，包含解和元数据
- **`Selection`**: 几何选择工具，用于通过几何形状选择边界条件
- **`batch_solve`**: 批量求解函数，支持错误隔离

### 示例

- 示例文件：`polyfempy/api/examples/`
- 示例内容：参见 [Examples README](../polyfempy/api/examples/README.md)

---

## 核心概念

在深入各模块之前，先了解几个贯穿整个架构的核心概念：

### 多后端数组支持（NumPy/Torch/JAX）

**设计原因**：
- 用户可能使用不同的数组库（NumPy、PyTorch、JAX）
- 支持 PyTorch 和 JAX 的自动求导功能
- 提供统一的 API，隐藏后端差异

**实现机制**：
- `tensor.py` 提供多后端转换功能
- 自动检测数组类型（`detect_backend()`）
- 转换为 NumPy（`as_numpy()`）供 C++ 后端使用
- 结果转换回原始后端（`to_backend()`）
- 零拷贝优化（当可能时）

#### 与 nanobind 兼容的专项工作

- `tensor.py` 强制所有数组在进入 C++ 前变成 **CPU、C-contiguous 的 NumPy**，满足 nanobind 的零拷贝要求
- `_ensure_i32()` 与 `Result` 统一单元连接的 dtype 为 `int32`，与 nanobind/Eigen 的接口约束一致
- `backend_nanobind.py` 提供薄适配层，当用户编译 `polyfem_nb` 时可直接调用 `solve_cpp`
- `solve.py` 直接探测 nanobind 构建的 `polyfempy`，依赖其零拷贝机制与 `Solver/State` API
- `polyfempy/api/examples/` 以 NumPy 数据为主，方便验证 nanobind 数据路径

**详细说明**：参见 [11. `tensor.py` - 张量转换](#11-tensorpy---张量转换)

### int32 类型要求

**原因**：
- C++ 后端要求单元连接数组必须是 `int32` 类型
- 内存效率：`int32`（4 字节）比 `int64`（8 字节）节省内存
- 数值范围足够：`int32` 可表示约 21 亿个顶点，满足大多数场景

**实现**：
- `_ensure_i32()` 函数确保类型正确
- 在 `solve()` 和 `Result` 中自动转换

### 版本兼容机制

**原因**：
- 支持不同版本的 polyfempy C++ 绑定
- 平滑升级路径
- 向后兼容

**实现**：
- `_first_attr()` 查找可用方法名
- 支持多个方法名变体（如 `set_mesh`、`set_mesh_data`、`load_mesh_from_points`）
- 优雅降级

---

## 2. `solve.py` - 主求解函数

### 设计理念

**统一接口 + 版本兼容**：
- 提供统一的 API 接口，隐藏底层实现细节
- 支持不同版本的 polyfempy C++ 绑定（通过反射机制）
- 自动处理多后端数组（参见[核心概念 - 多后端数组支持](#多后端数组支持numptorchjax)）
- 版本兼容：通过 `_first_attr()` 查找可用方法名

### 核心函数

#### `solve(vertices, cells, cfg, sidesets_func=None, dtype=None)`

**功能**：
- 统一的求解入口点
- 自动归一化用户数组（NumPy/Torch/JAX → NumPy）
- 构建后端 Settings/Problem
- 适配不同版本的 polyfempy
- 应用边界条件
- 运行求解器
- 返回 Result 对象

**参数**：
- `vertices`: 顶点坐标，shape (N, dim)，支持 NumPy/Torch/JAX（参见[核心概念 - 多后端数组支持](#多后端数组支持numptorchjax)）。如果为 None 且 cfg 包含 geometry，将从文件加载网格
- `cells`: 单元连接，shape (M, k)，支持 NumPy/Torch/JAX，自动转换为 int32（参见[核心概念 - int32 类型要求](#int32-类型要求)）。如果为 None 且 cfg 包含 geometry，将从文件加载网格
- `cfg`: SimulationConfig 实例、dict 或 str（文件路径）。支持完整的 PolyFEM JSON 配置
- `sidesets_func`: 可选的侧边集构建函数
- `dtype`: 可选的 NumPy 数据类型

**返回**：
- `Result`: 结果对象，包含解和元数据

**工作流程**：

```
1. 处理配置（SimulationConfig/dict/str → SimulationConfig）
   - 如果是文件路径，使用 from_json_file() 加载
   - 如果是 dict，使用 from_json_dict() 转换
2. 检查 JSON 模式（是否有完整 JSON 配置和 geometry）
   - 如果有 geometry 且 vertices/cells 为 None，使用 JSON 模式
3. 归一化输入数组（如果提供，NumPy/Torch/JAX → NumPy）
4. 导入 polyfempy（C++ 绑定）
5. 构造求解器（支持不同版本）
6. 应用设置：
   - JSON 模式：直接使用完整 JSON 配置，从文件加载网格
   - 普通模式：构建 Settings，设置网格数组
7. 构建基函数和组装（JSON 模式需要）
8. 设置侧边集（可选）
9. 应用边界条件（普通模式，JSON 模式中 BC 已在配置中）
10. 运行求解
11. 获取解并返回 Result
```

### 辅助函数

#### `_first_attr(obj, *names)`

**功能**：查找对象上第一个存在的属性名

**用途**：版本兼容，支持不同版本的 API

**示例**：
```python
# 查找 settings 或 set_settings
name = _first_attr(solver, "settings", "set_settings")
if name:
    getattr(solver, name)(settings)
```

#### `_ensure_i32(cells)`

**功能**：确保单元数组是 int32 类型

**用途**：类型规范化，确保与 C++ 后端兼容（参见[核心概念 - int32 类型要求](#int32-类型要求)）

### 版本兼容机制

通过 `_first_attr()` 支持不同版本的 API（参见[核心概念 - 版本兼容机制](#版本兼容机制)）：

```python
# 支持不同的方法名
for name in ("set_mesh", "set_mesh_data", "load_mesh_from_points"):
    if hasattr(solver, name):
        fn = getattr(solver, name)
        try:
            fn(V_np, C_np)
            break
        except TypeError:
            try:
                fn(points=V_np, cells=C_np)
                break
            except Exception:
                pass
```

### 示例

- 示例文件：`polyfempy/api/examples/`
- 示例内容：参见 [Examples README](../polyfempy/api/examples/README.md)

---

## 3. `config.py` - 配置类

> **详细配置指南**：参见 [配置指南](config-guide.md) - 包含参数输入、验证、数据流转和扩展的完整说明。

### 设计理念

**人性化配置 → 规范形式 → 后端设置**：
- 提供直观的配置字段
- 自动规范化别名（PDE 名称、材料参数）
- 转换为后端 Settings/Problem 对象
- 支持 JSON 序列化

### 核心类

#### `SimulationConfig`

**功能**：
- 存储仿真配置
- 规范化别名（PDE、材料参数）
- 转换为后端 Settings
- 支持 JSON 序列化
- 验证配置有效性

**属性**：
- `pde`: PDE 名称（自动规范化）
- `discr_order`: 离散化阶数
- `materials`: 材料参数（自动规范化键名）
- `boundary_conditions`: 边界条件
- `extras`: 高级选项
- `selection`: 可选的几何选择对象（`Selection` 类型），用于通过几何形状选择边界条件
- `problem_type`: 可选的预定义问题类型（如 'Gravity', 'Franke', 'TorsionElastic' 等）
- `problem_params`: 预定义问题的参数字典（如 `{'force': 0.1}` 用于 Gravity 问题）

**主要方法**：

##### `canonicalized() -> SimulationConfig`

**功能**：返回规范化后的配置副本

**用途**：规范化别名，确保一致性

**示例**：
```python
cfg = SimulationConfig(pde="linear_elasticity")
cfg_canon = cfg.canonicalized()
# cfg_canon.pde == "LinearElasticity"
```

##### `to_settings() -> pf.Settings`

**功能**：转换为后端 Settings 对象

**用途**：构建后端配置对象

**策略**：
- PDE: 'Poisson' → `pf.GenericScalar()`
- PDE: 'LinearElasticity' → `pf.GenericTensor()`
- 材料: 设置 E/nu
- 高级选项: 通过 `set_advanced_option` 传递
- **预定义问题**：如果指定了 `problem_type`，优先使用预定义问题（从 `pf` 或 `polyfempy.legacy.Problems` 加载）
- **Selection 处理**：如果配置包含 `selection`，将其转换为字典并存储在 `settings._selection` 中，供后续使用
- **版本兼容**：自动适配不同版本的 polyfempy API（支持 `set_problem`、`set_pde` 等多种方式）
- **优雅降级**：如果 C++ 后端不可用，使用 `_DummySettings` 占位符保持 API 可用性

##### `to_json_str() -> str`

**功能**：序列化为 JSON 字符串

**用途**：配置持久化

##### `to_dict() -> dict`

**功能**：将 SimulationConfig 转换为字典表示

**用途**：获取配置的字典形式，供后端使用或序列化

**行为**：
- 如果配置包含完整 JSON（`extras["_full_json_config"]`），直接返回该完整配置
- 否则，从字段构造字典，包括规范化后的字段和可选字段（extras, problem_type, problem_params, selection）
- **参数提升**：从 `extras` 中提取常用参数（如 `max_iters`、`random_seed`）并提升到顶层，同时进行验证和类型转换

**示例**：
```python
cfg = SimulationConfig.linear_elasticity(2100, 0.3)
config_dict = cfg.to_dict()
# 返回包含 pde, discr_order, materials, boundary_conditions 等的字典

# 参数提升示例
cfg = SimulationConfig(extras={"max_iters": "10", "random_seed": "42"})
d = cfg.to_dict()
# d["max_iters"] = 10 (整数，从 extras 提升到顶层)
# d["random_seed"] = 42 (整数，从 extras 提升到顶层)
```

**详细说明**：参见 [配置指南](config-guide.md) 中的"数据流转过程"和"参数验证机制"章节

##### `from_json_str(s: str) -> SimulationConfig`

**功能**：从 JSON 字符串反序列化

**用途**：配置加载

##### `from_json_dict(d: dict) -> SimulationConfig`

**功能**：从完整的 PolyFEM JSON 字典创建配置

**用途**：加载完整的 PolyFEM JSON 配置（支持所有参数）

**支持的参数**：
- `geometry`: 网格文件、变换、选择
- `materials`: 所有材料类型和参数
- `boundary_conditions`: 所有边界条件类型
- `time`: 瞬态设置（t0, tend, dt, integrator）
- `contact`: 接触设置（enabled, dhat, mu, epsv 等）
- `solver`: 线性/非线性求解器设置
- `output`: Paraview、JSON 输出设置
- `space`: 离散化阶数（支持列表格式，如 `[{"id": 2, "order": 2}]`）
- `common`: JSON 引用（自动合并）

**实现策略**：
- 完整 JSON 保存在 `extras["_full_json_config"]` 中
- 已知字段（pde, discr_order, materials）也被提取以便使用
- 支持 `common.json` 的深层嵌套合并
- **PDE 自动推断**：如果未指定 PDE 且材料类型为 NeoHookean 或 SaintVenant，自动推断为 `NonLinearElasticity`
- **材料格式支持**：支持字典格式和数组格式的材料定义（数组格式时取第一个材料）
- **discr_order 格式支持**：支持标量、列表和嵌套字典格式（如 `space.discr_order`）

**示例**：
```python
import json
with open("config.json") as f:
    config_dict = json.load(f)
cfg = SimulationConfig.from_json_dict(config_dict)
# 完整配置可通过 cfg.to_dict() 或 cfg.extras["_full_json_config"] 访问
```

##### `from_json_file(filepath: str) -> SimulationConfig`

**功能**：从 JSON 文件加载配置

**用途**：直接从文件加载 PolyFEM JSON 配置

**实现**：
- 自动处理 `common.json` 引用
- 自动合并引用的 common.json 文件
- 支持深层嵌套合并

**示例**：
```python
cfg = SimulationConfig.from_json_file("data/contact/examples/2D/unit-tests/5-squares.json")
```

##### `validate() -> None`

**功能**：验证配置有效性

**检查项**：
- `discr_order` 必须是正整数
- `materials['E']` 和 `materials['nu']` 必须是数字

##### `linear_elasticity(E, nu, order=1) -> SimulationConfig`

**功能**：创建线性弹性配置的便捷方法

**用途**：快速创建常用配置

**示例**：
```python
cfg = SimulationConfig.linear_elasticity(E=2100, nu=0.3, order=2)
```

##### `poisson(order=1) -> SimulationConfig`

**功能**：创建 Poisson 配置的便捷方法

**用途**：快速创建常用配置

**示例**：
```python
cfg = SimulationConfig.poisson(order=1)
```

##### `gravity(force=0.1, E=None, nu=None, order=1) -> SimulationConfig`

**功能**：创建重力问题配置的便捷方法

**用途**：快速创建重力问题配置

**参数**：
- `force`: 重力大小，默认为 0.1
- `E`: 杨氏模量（可选）
- `nu`: 泊松比（可选）
- `order`: 离散化阶数，默认为 1

**示例**：
```python
cfg = SimulationConfig.gravity(force=0.1, E=1e6, nu=0.3)
```

##### `franke(order=1) -> SimulationConfig`

**功能**：创建 Franke 问题配置的便捷方法（标量问题，有精确解）

**用途**：快速创建 Franke 问题配置

**示例**：
```python
cfg = SimulationConfig.franke(order=2)
```

##### `torsion(axis_coordinate=2, n_turns=0.5, fixed_boundary=5, turning_boundary=6, E=None, nu=None, order=1) -> SimulationConfig`

**功能**：创建扭转问题配置的便捷方法（3D）

**用途**：快速创建 3D 扭转问题配置

**参数**：
- `axis_coordinate`: 轴方向（1=x, 2=y, 3=z），默认为 2（y 轴）
- `n_turns`: 转数，默认为 0.5
- `fixed_boundary`: 固定边界的面集 ID，默认为 5
- `turning_boundary`: 转动边界的面集 ID，默认为 6
- `E`: 杨氏模量（可选）
- `nu`: 泊松比（可选）
- `order`: 离散化阶数，默认为 1

**示例**：
```python
cfg = SimulationConfig.torsion(axis_coordinate=2, n_turns=0.5)
```

##### `flow(inflow=1, outflow=3, inflow_amount=0.25, outflow_amount=0.25, direction=0, obstacle=None, order=1) -> SimulationConfig`

**功能**：创建流动问题配置的便捷方法（流入/流出）

**用途**：快速创建流体流动问题配置

**参数**：
- `inflow`: 流入面集 ID，默认为 1
- `outflow`: 流出面集 ID，默认为 3
- `inflow_amount`: 流入量，默认为 0.25
- `outflow_amount`: 流出量，默认为 0.25
- `direction`: 流动方向，默认为 0
- `obstacle`: 障碍物面集 ID 列表，默认为 [7]
- `order`: 离散化阶数，默认为 1

**示例**：
```python
cfg = SimulationConfig.flow(inflow=1, outflow=3)
```

##### `driven_cavity(order=1) -> SimulationConfig`

**功能**：创建驱动腔问题配置的便捷方法

**用途**：快速创建经典驱动腔问题配置

**示例**：
```python
cfg = SimulationConfig.driven_cavity(order=2)
```

##### `flow_with_obstacle(U=1.5, time_dependent=True, order=1) -> SimulationConfig`

**功能**：创建带障碍物的流动问题配置的便捷方法

**用途**：快速创建带障碍物的流体流动问题配置

**参数**：
- `U`: 流动速度，默认为 1.5
- `time_dependent`: 是否为瞬态问题，默认为 True
- `order`: 离散化阶数，默认为 1

**示例**：
```python
cfg = SimulationConfig.flow_with_obstacle(U=1.5)
```

**便捷工厂方法总结**：

这些便捷方法分为两类：

1. **基础 PDE 便捷方法**：
   - `linear_elasticity()`: 线性弹性问题
   - `poisson()`: Poisson 问题

2. **预定义问题工厂方法**：
   - `gravity()`: 重力问题
   - `franke()`: Franke 问题（标量，有精确解）
   - `torsion()`: 扭转问题（3D）
   - `flow()`: 流动问题（流入/流出）
   - `driven_cavity()`: 驱动腔问题
   - `flow_with_obstacle()`: 带障碍物的流动问题

这些方法都是类方法（`@classmethod`），可以直接通过 `SimulationConfig.method_name()` 调用，无需先创建实例。它们内部会设置相应的 `pde`、`problem_type` 和 `problem_params` 参数。

---

## 4. `selection.py` - 几何选择工具

### 设计理念

**几何选择而非 ID 选择**：允许用户通过几何形状（球体、盒子、平面）选择边界条件，而不需要知道网格文件中的具体 sideset ID。这在网格文件没有正确的 sideset 标记时特别有用。

### 核心类

#### `Selection`

**功能**：
- 通过几何形状选择边界条件
- 支持 body 和 sideset 选择
- 支持多种几何形状（球体、盒子、平面）

**属性**：
- `body_ids`: 体选择列表
- `boundary_sidesets`: 边界面集选择列表

**主要方法**：

##### Body 选择方法

- `select_body_with_sphere(id, center, radius)`: 使用球体选择体
- `select_body_with_box(id, box_min, box_max)`: 使用轴对齐盒子选择体
- `select_body_with_axis_plane(id, axis, position)`: 使用轴对齐平面选择体
- `select_body_with_plane(id, normal, offset)`: 使用通用平面选择体

##### Sideset 选择方法

- `select_sideset_with_sphere(id, center, radius)`: 使用球体选择边界面集
- `select_sideset_with_box(id, box_min, box_max)`: 使用轴对齐盒子选择边界面集
- `select_sideset_with_axis_plane(id, axis, position)`: 使用轴对齐平面选择边界面集
- `select_sideset_with_plane(id, normal, offset)`: 使用通用平面选择边界面集

##### 转换方法

- `to_dict() -> dict`: 转换为字典表示
- `to_json_str() -> str`: 转换为 JSON 字符串

**示例**：
```python
from polyfempy.api import SimulationConfig, Selection

# 创建 Selection 对象
selection = Selection()
selection.select_sideset_with_sphere(id=1, center=[0, 0, 0], radius=1.0)
selection.select_sideset_with_box(id=2, box_min=[0, 0, 0], box_max=[1, 1, 1])

# 在配置中使用 Selection
cfg = SimulationConfig(
    selection=selection,
    boundary_conditions={
        "dirichlet_boundary": [{"id": 1, "value": [0, 0]}],
    }
)
```

**用途**：
- 当网格文件没有正确的 sideset 标记时
- 需要根据几何位置动态选择边界条件时
- 简化边界条件的设置过程

### 示例

- 示例文件：`polyfempy/api/examples/`
- 示例内容：参见 [Examples README](../polyfempy/api/examples/README.md)

---

## 5. `result.py` - 结果容器

### 设计理念

**统一容器 + 多后端支持**：
- 内部存储为 NumPy 数组（C-contiguous）
- 支持转换回原始后端（参见[核心概念 - 多后端数组支持](#多后端数组支持numptorchjax)）
- 提供字段管理功能
- 支持 VTK 导出

### 核心类

#### `Result`

**功能**：
- 存储网格和点场数据
- 支持多后端数组转换
- 提供字段管理功能
- 支持 VTK 导出

**属性**：
- `backend`: 原始后端名称（'numpy'|'torch'|'jax'）
- `vertices`: 顶点坐标，shape (N, dim)
- `cells`: 单元连接，shape (M, k)，自动转换为 int32（参见[核心概念 - int32 类型要求](#int32-类型要求)）
- `fields`: 字段字典，例如 {'u': (N, dim)}
- `meta`: 元数据字典

**主要方法**：

##### `field(name) -> np.ndarray`

**功能**：获取字段值

**用途**：访问字段数据

##### `set_field(name, value) -> self`

**功能**：设置字段值

**用途**：添加或更新字段

##### `remove_field(name) -> self`

**功能**：删除字段

**用途**：移除不需要的字段

##### `as_numpy() -> self`

**功能**：规范化到 NumPy（幂等操作）

**用途**：确保所有数组都是 NumPy/C-contiguous

##### `to_backend(include_mesh=False) -> self`

**功能**：转换回原始后端

**用途**：返回用户原始后端格式的数组

##### `magnitude(name, out_name=None, eps=0.0) -> self`

**功能**：计算向量场的模

**用途**：创建标量场（例如位移模）

##### `to_vtk(path) -> None`

**功能**：导出到 VTK 格式

**用途**：可视化结果

**策略**：
- 如果安装了 `meshio`，使用 VTK 格式
- 否则，保存为 NPZ 格式

##### `summary() -> dict`

**功能**：返回结果摘要

**用途**：调试和验证

##### `field_names() -> list[str]`

**功能**：返回所有字段名

**用途**：列出可用字段

### 内部方法

##### `_make_contiguous_inplace() -> None`

**功能**：确保内部数组是 C-contiguous

**用途**：性能优化和兼容性

##### `_point_fields() -> dict`

**功能**：收集可以作为点数据的字段

**用途**：VTK 导出

##### `_guess_cell_type(cells, vertices) -> str`

**功能**：猜测网格单元类型

**用途**：VTK 导出

### 示例

- 示例文件：`polyfempy/api/examples/`
- 示例内容：参见 [Examples README](../polyfempy/api/examples/README.md)

---

## 6. `errors.py` - 错误处理

### 设计理念

**统一错误模型 + 清晰前缀**：
- 所有错误都有明确的前缀（INPUT:/CALLBACK:/BACKEND:）
- 错误类型与错误原因对应
- 提供清晰的错误消息

### 核心函数

#### `raise_input_error(msg: str) -> None`

**功能**：抛出输入错误

**用途**：输入验证失败

**错误类型**：`ValueError`

**前缀**：`INPUT:`

**示例**：
```python
raise_input_error("vertices must be float64 C-contiguous")
# ValueError: INPUT: vertices must be float64 C-contiguous
```

#### `raise_callback_type_error(msg: str) -> None`

**功能**：抛出回调错误

**用途**：回调返回值类型错误

**错误类型**：`TypeError`

**前缀**：`CALLBACK:`

**示例**：
```python
raise_callback_type_error("body_force must return ndarray")
# TypeError: CALLBACK: body_force must return ndarray
```

#### `raise_backend_error(msg: str) -> None`

**功能**：抛出后端错误

**用途**：后端内部失败

**错误类型**：`RuntimeError`

**前缀**：`BACKEND:`

**示例**：
```python
raise_backend_error("solver failed to converge")
# RuntimeError: BACKEND: solver failed to converge
```

### 示例

- 示例文件：`polyfempy/api/examples/`
- 示例内容：参见 [Examples README](../polyfempy/api/examples/README.md)

---

## 7. `backend_base.py` - 后端 SPI 定义

### 设计理念

**接口契约 + 文档说明**：
- 定义后端接口契约（SPI）
- 提供详细的文档说明
- 确保所有后端实现一致性

### 核心函数

#### `solve_impl(V, C, settings, callbacks) -> dict`

**功能**：后端 SPI 接口（文档函数）

**用途**：定义后端接口契约

**输入契约**：
- `V`: np.ndarray, shape (N, dim), dtype float64, C-contiguous
- `C`: np.ndarray, shape (M, k), dtype int32, C-contiguous
- `settings`: dict from SimulationConfig.to_dict()
- `callbacks`: dict[str, callable] or None

**输出契约**：
- 必须返回 dict，包含以下键：
  - `u`: np.ndarray, shape (N, dim), dtype float64, C-contiguous
  - `strain`: np.ndarray or None
  - `stress`: np.ndarray or None
  - `meta`: dict with required keys (backend, iters, residual, seed)

**Meta Schema**：
- `backend`: str ("dummy" | "nanobind")
- `iters`: int (迭代次数)
- `residual`: float (最终残差，必须是有限值)
- `seed`: int or None (随机种子)

### 后端职责

1. **输入验证**：验证输入是否符合契约（虽然调用方已经验证）
2. **回调处理**：按顺序调用回调（before_solve → after_iter×K → after_solve）
3. **确定性输出**：如果提供了 random_seed，产生确定性输出
4. **返回结果**：返回符合契约的 dict

### 示例

- 示例文件：`polyfempy/api/examples/`
- 示例内容：参见 [Examples README](../polyfempy/api/examples/README.md)

---

## 8. `backend_dummy.py` - Dummy 后端实现

### 设计理念

**严格验证 + 确定性输出 + 回调测试**：
- 严格验证输入（dtype、shape、contiguity）
- 产生确定性伪随机输出
- 正确处理回调生命周期
- 统一错误模型

### 核心函数

#### `solve_impl(V, C, settings, callbacks) -> dict`

**功能**：Dummy 后端实现

**实现策略**：
1. **输入验证**：严格验证 V 和 C 的 dtype、shape、contiguity
2. **解析设置**：从 settings 获取 max_iters、random_seed
3. **回调处理**：按顺序调用回调（before_solve → after_iter×K → after_solve）
4. **生成输出**：使用确定性随机数生成器生成伪随机位移场
5. **返回结果**：返回符合 SPI 契约的 dict

**确定性输出**：
```python
rng = np.random.RandomState(random_seed)
u = rng.normal(loc=0.0, scale=1e-3, size=(N, dim)).astype(np.float64)
```

**残差模型**：
```python
residual = 1e-3 / (i + 1)  # 线性递减
```

### 示例

- 示例文件：`polyfempy/api/examples/`
- 示例内容：参见 [Examples README](../polyfempy/api/examples/README.md)

---

## 9. `backend_nanobind.py` - Nanobind 后端适配器

### 设计理念

**适配器模式 + 优雅降级**：
- 尝试导入 C++ 后端
- 如果不可用，提供清晰的错误消息
- 转发调用到 C++ 后端

### 核心函数

#### `solve_impl(V, C, settings, callbacks) -> dict`

**功能**：Nanobind 后端适配器

**实现策略**：
1. **检查可用性**：尝试导入 `polyfem_nb`
2. **错误处理**：如果不可用，抛出 `NotImplementedError`
3. **转发调用**：如果可用，转发到 `solve_cpp()`

**错误消息**：
```python
raise NotImplementedError(
    "nanobind backend not connected. "
    "Please build the C++ module (polyfem_nb) first."
)
```

### 示例

- 示例文件：`polyfempy/api/examples/`
- 示例内容：参见 [Examples README](../polyfempy/api/examples/README.md)

---

## 10. `batch.py` - 批量处理

### 设计理念

**顺序处理 + 错误隔离 + 顺序保持**：
- 顺序处理所有任务
- 错误隔离：一个任务失败不影响其他任务
- 顺序保持：结果顺序与输入顺序一致

### 核心函数

#### `batch_solve(jobs) -> list[Result]`

**功能**：批量求解

**参数**：
- `jobs`: 任务列表，每个任务是：
  - 3-tuple: `(V, C, cfg)`
  - 4-tuple: `(V, C, cfg, kwargs_dict)`

**返回**：
- `list[Result]`: 结果列表，顺序与输入一致

**实现策略**：
```python
out = []
for job in jobs:
    if len(job) == 3:
        V, C, cfg = job
        kwargs = {}
    else:
        V, C, cfg, kwargs = job
        kwargs = kwargs or {}
    res = solve(V, C, cfg, **kwargs)
    out.append(res)
return out
```

**错误处理**：
- 如果任务失败，结果位置包含 Exception 对象
- 其他任务继续执行
- 顺序保持：结果顺序与输入顺序一致

### 示例

- 示例文件：`polyfempy/api/examples/batch_processing.py` - 批量处理示例
- 示例内容：参见 [Examples README](../polyfempy/api/examples/README.md)

---

## 11. `tensor.py` - 张量转换

### 设计理念

**多后端支持 + 零拷贝优化**（参见[核心概念 - 多后端数组支持](#多后端数组支持numptorchjax)）：
- 支持 NumPy/Torch/JAX 数组
- 自动检测后端
- 零拷贝转换（当可能时）
- 强制 C-contiguous 布局

### 为什么需要 tensor.py？即使 nanobind 支持 zero copy

**核心原因**：nanobind 的 zero copy **只适用于 NumPy 数组**，但用户可能使用 Torch 或 JAX 数组。

**nanobind 的限制**：
- ✅ 支持：NumPy 数组 → C++（zero copy）
- ❌ 不支持：Torch 张量 → C++
- ❌ 不支持：JAX 数组 → C++
- ❌ 不支持：GPU 张量

**tensor.py 的作用**：
1. **多后端支持**：将 Torch/JAX 数组转换为 NumPy 数组
2. **设备处理**：将 GPU 张量移动到 CPU
3. **内存布局**：确保 C-contiguous 布局
4. **返回转换**：将结果转换回用户原始后端

**数据流**：
```
用户 Torch 张量
    ↓ tensor.py: 转换为 NumPy（零拷贝，CPU + contiguous）
NumPy 数组 (C-contiguous)
    ↓ nanobind: zero copy NumPy → C++
C++ Eigen 矩阵
    ↓ 计算
C++ 结果
    ↓ nanobind: zero copy C++ → NumPy
NumPy 数组
    ↓ tensor.py: 转换回 Torch（零拷贝）
用户 Torch 张量
```

**详细说明**：相关内容已在本文档的"为什么需要 tensor.py？"和"零拷贝优化"章节中详细说明。

### 核心函数

#### `detect_backend(x) -> str`

**功能**：检测数组后端

**用途**：自动识别数组类型

**返回**：'numpy' | 'torch' | 'jax'

**实现策略**：
```python
def detect_backend(x):
    if _is_torch_tensor(x):
        return "torch"
    if _is_jax_array(x):
        return "jax"
    return "numpy"
```

#### `as_numpy(x, dtype=None) -> (np.ndarray, str)`

**功能**：转换为 NumPy 数组

**用途**：归一化多后端数组

**返回**：
- `arr_np`: NumPy 数组（C-contiguous）
- `backend`: 原始后端名称

**实现策略**：
- Torch: 零拷贝转换（CPU + contiguous）
- JAX: 通过 `np.asarray` 转换
- NumPy: 直接返回（确保 C-contiguous）

#### `to_backend(arr, backend) -> array`

**功能**：转换回指定后端

**用途**：返回用户原始后端格式

**实现策略**：
- Torch: 使用 `torch.from_numpy`（零拷贝）
- JAX: 使用 `jnp.asarray`
- NumPy: 直接返回

#### `from_numpy(arr, backend) -> array`

**功能**：从 NumPy 转换到指定后端

**用途**：后端转换

### 零拷贝优化

**重要说明**：零拷贝发生在两个不同层面：

1. **PyTorch 的零拷贝**（Torch ↔ NumPy）：
   - 这是 PyTorch 本身的功能，不是 nanobind 的
   - `tensor.numpy()`：当 tensor 在 CPU 且内存连续时，返回的 NumPy 数组与 tensor **共享内存**
   - `torch.from_numpy()`：从 NumPy 数组创建 tensor，**共享内存**（要求 C-contiguous）

2. **nanobind 的零拷贝**（NumPy ↔ C++）：
   - nanobind 支持 NumPy 数组与 C++ 之间的零拷贝
   - 但不支持 Torch 张量直接到 C++

**完整的数据流**：
```
用户 Torch 张量
    ↓ PyTorch 零拷贝：t.numpy()（CPU + contiguous）
NumPy 数组 (共享内存)
    ↓ nanobind 零拷贝：NumPy → C++
C++ Eigen 矩阵
    ↓ 计算
C++ 结果
    ↓ nanobind 零拷贝：C++ → NumPy
NumPy 数组
    ↓ PyTorch 零拷贝：torch.from_numpy()
用户 Torch 张量
```

**Torch CPU 零拷贝实现**：
```python
def _torch_to_numpy_zero_copy(t):
    t = t.detach()
    if t.device.type != "cpu":
        t = t.cpu()
    if not t.is_contiguous():
        t = t.contiguous()
    return t.numpy()  # PyTorch 提供的零拷贝（CPU + contiguous）
```

### 示例

- 示例文件：`polyfempy/api/examples/`
- 示例内容：参见 [Examples README](../polyfempy/api/examples/README.md)

---

## 示例架构

### 示例文件结构

```
polyfempy/api/examples/
├── run_dummy_elasticity.py      # 基础 Dummy 后端示例
├── run_elasticity.py            # 最小 2D 线性弹性示例
├── load_from_json.py            # 从 JSON 文件加载配置
├── parameter_sweep.py           # 参数扫描（参数敏感性研究）
├── batch_processing.py          # 批量处理（错误隔离）
├── with_callbacks.py            # 使用 callbacks 监控求解
└── README.md                    # 示例说明文档
```

### 示例分类

#### 1. 基础示例

**目的**：展示基本的 API 使用方法

**示例文件**：
- `run_dummy_elasticity.py` - 最简单的使用示例
- `run_elasticity.py` - 线性弹性问题示例

**演示内容**：
- 创建网格（顶点和单元）
- 配置仿真参数
- 运行求解器
- 查看结果

#### 2. 配置示例

**目的**：展示不同的配置方式

**示例文件**：
- `load_from_json.py` - 从 JSON 文件加载完整配置

**演示内容**：
- 从 JSON 文件加载配置
- 使用包含 geometry 的 JSON（自动加载网格）
- 处理完整的 PolyFEM JSON 配置

#### 3. 高级用法示例

**目的**：展示高级功能和最佳实践

**示例文件**：
- `parameter_sweep.py` - 参数扫描和敏感性分析
- `batch_processing.py` - 批量处理和错误隔离
- `with_callbacks.py` - 使用 callbacks 监控进度

**演示内容**：
- 运行多个仿真（不同参数）
- 批量处理多个配置
- 错误隔离（一个失败不影响其他）
- 监控求解进度
- 记录收敛历史

### 运行示例

所有示例都可以通过以下方式运行：

```bash
# 运行单个示例
python -m polyfempy.api.examples.run_dummy_elasticity
python -m polyfempy.api.examples.parameter_sweep
python -m polyfempy.api.examples.batch_processing
python -m polyfempy.api.examples.with_callbacks
python -m polyfempy.api.examples.load_from_json
```

详细说明请参见：[Examples README](../polyfempy/api/examples/README.md)

---

## 设计模式

### 1. 适配器模式（Adapter Pattern）

**应用**：`backend_nanobind.py`

**目的**：适配 C++ 后端到 Python API

**实现**：
```python
def solve_impl(V, C, settings, callbacks):
    return solve_cpp(V, C, settings, callbacks)
```

### 2. 策略模式（Strategy Pattern）

**应用**：后端切换（dummy vs nanobind）

**目的**：运行时选择后端实现

**实现**：
```python
if backend == "dummy":
    return backend_dummy.solve_impl(V, C, settings, callbacks)
elif backend == "nanobind":
    return backend_nanobind.solve_impl(V, C, settings, callbacks)
```

### 3. 工厂模式（Factory Pattern）

**应用**：`SimulationConfig.to_settings()`

**目的**：根据配置创建后端 Settings 对象

**实现**：
```python
if c.pde == "Poisson":
    problem = pf.GenericScalar()
else:
    problem = pf.GenericTensor()
```

### 4. 外观模式（Facade Pattern）

**应用**：`solve()` 函数

**目的**：隐藏底层复杂性，提供简单接口

**实现**：
```python
def solve(vertices, cells, cfg, ...):
    # 归一化输入
    # 构建设置
    # 创建求解器
    # 运行求解
    # 返回结果
```

### 5. 单例模式（Singleton Pattern）

**应用**：错误处理函数

**目的**：提供统一的错误处理接口

**实现**：
```python
def raise_input_error(msg):
    raise ValueError("INPUT: " + msg)
```

---

## 数据流

### 求解流程

```
用户代码
    ↓
solve(vertices, cells, cfg)
    ↓
1. 归一化输入（tensor.py）
    - as_numpy(vertices) → V_np, backend
    - as_numpy(cells) → C_np, _
    ↓
2. 构建设置（config.py）
    - cfg.to_settings() → settings
    ↓
3. 选择后端（backend_base.py）
    - backend == "dummy" → backend_dummy.solve_impl()
    - backend == "nanobind" → backend_nanobind.solve_impl()
    ↓
4. 后端实现（backend_dummy.py / backend_nanobind.py）
    - 输入验证
    - 回调处理
    - 运行求解
    - 返回结果 dict
    ↓
5. 构建结果（result.py）
    - Result(backend, vertices, cells, fields, meta)
    ↓
6. 返回结果
    - result.to_backend() → 转换回原始后端
    ↓
用户代码
```

### 配置流程

**方式 1：程序化配置**
```
用户配置
    ↓
SimulationConfig(pde="linear_elasticity", ...)
    ↓
cfg.canonicalized()
    - 规范化 PDE 名称
    - 规范化材料参数
    ↓
cfg.to_settings()
    - 创建 pf.Settings
    - 设置 PDE 类型
    - 设置材料参数
    - 设置高级选项
    ↓
pf.Settings 对象
```

**方式 2：JSON 文件配置**
```
JSON 文件（可能包含 common.json 引用）
    ↓
SimulationConfig.from_json_file(filepath)
    - 加载 JSON 文件
    - 检测并合并 common.json（深层递归合并）
    - 提取已知字段（pde, discr_order, materials）
    - 完整 JSON 保存在 extras["_full_json_config"]
    ↓
SimulationConfig 实例（包含完整 JSON）
    ↓
solve() 检测 JSON 模式
    - 如果有 geometry 且无 vertices/cells，使用 JSON 模式
    - 直接使用完整 JSON 配置
    - 从文件加载网格
    ↓
求解器配置完成
```

### 结果流程

```
后端输出 dict
    ↓
Result(backend, vertices, cells, fields, meta)
    ↓
result.as_numpy()
    - 确保所有数组是 NumPy/C-contiguous
    ↓
result.to_backend()
    - 转换字段回原始后端
    - 可选：转换网格回原始后端
    ↓
用户结果
```

---

## 总结

### 核心设计理念

1. **统一接口**：提供简单的 API，隐藏复杂性
2. **多后端支持**：支持 NumPy/Torch/JAX 数组
3. **版本兼容**：支持不同版本的 C++ 绑定
4. **错误隔离**：批量处理中错误不互相影响
5. **确定性输出**：相同输入产生相同输出
6. **清晰错误**：所有错误都有明确前缀和消息

### 文件职责

- **`solve.py`**: 主求解函数，统一入口
- **`config.py`**: 配置类，规范化配置
- **`result.py`**: 结果容器，多后端支持
- **`errors.py`**: 错误处理，统一错误模型
- **`backend_base.py`**: 后端 SPI 定义，接口契约
- **`backend_dummy.py`**: Dummy 后端实现，测试用
- **`backend_nanobind.py`**: Nanobind 后端适配器，C++ 连接
- **`batch.py`**: 批量处理，错误隔离
- **`tensor.py`**: 张量转换，多后端支持

### 示例说明

所有示例都位于 `polyfempy/api/examples/` 目录，每个示例都是可运行的独立脚本，展示了不同的 API 使用场景。详细说明请参见 [Examples README](../polyfempy/api/examples/README.md)。

