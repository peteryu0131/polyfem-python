# SimulationConfig 配置指南

本文档全面说明 `SimulationConfig` 的使用方法，包括参数输入、验证、数据流转和扩展。

---

## 目录

1. [快速开始](#快速开始)
2. [数据流转过程](#数据流转过程)
3. [参数输入方式](#参数输入方式)
4. [Class-Based 配置详解](#class-based-配置详解)
5. [参数验证机制](#参数验证机制)
6. [JSON 参数支持](#json-参数支持)
7. [实际应用示例](#实际应用示例)
8. [添加新参数](#添加新参数)

**相关文档**：
- [Config Classes 使用指南](config-classes.md) - 详细的 Class-Based 配置说明
- [Configuration Guide (English)](config-guide-en.md) - English version of this guide

---

## 快速开始

### 基本使用

```python
from polyfempy.api import SimulationConfig, solve
import numpy as np

# 创建简单配置
cfg = SimulationConfig(
    pde="linear_elasticity",
    materials={"E": 1e6, "nu": 0.3},
    extras={"max_iters": 10, "random_seed": 42}
)

# 使用配置
V = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
C = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
result = solve(V, C, cfg)
```

### 从 JSON 文件加载

```python
# 从 JSON 文件加载完整配置
cfg = SimulationConfig.from_json_file("config.json")
result = solve(vertices=None, cells=None, cfg=cfg)
```

---

## 数据流转过程

### 为什么有这么多字典？

在 API 中有几个不同的字典，它们在不同阶段使用：

1. **`SimulationConfig.extras`** - 用户输入的额外参数
2. **`SimulationConfig.to_dict()` 返回的字典** - 传递给后端的配置
3. **`settings` 字典（传递给 `solve_impl()`）** - 后端实际使用的配置

### 数据流转过程

#### 阶段 1：用户创建配置

```python
# 用户创建 SimulationConfig
cfg = SimulationConfig(
    pde="linear_elasticity",
    materials={"E": 1e6, "nu": 0.3},
    extras={"max_iters": 10, "random_seed": 42}  # ← 放在 extras 中
)
```

**为什么放在 `extras`？**
- `SimulationConfig` 是一个 dataclass，只有固定的字段：`pde`, `discr_order`, `materials`, `boundary_conditions`, `extras`, `selection`, `problem_type`, `problem_params`
- `max_iters` 和 `random_seed` 不是 `SimulationConfig` 的直接字段
- 所以需要放在 `extras` 这个"杂项"字典中

#### 阶段 2：转换为字典（`to_dict()`）

```python
# 调用 to_dict() 转换为字典
settings_dict = cfg.to_dict()
# 结果：
# {
#     "pde": "linear_elasticity",
#     "discr_order": 1,
#     "materials": {"E": 1e6, "nu": 0.3},
#     "boundary_conditions": {},
#     "extras": {"max_iters": 10, "random_seed": 42},
#     "max_iters": 10,        # ← 从 extras 提升到顶层
#     "random_seed": 42       # ← 从 extras 提升到顶层
# }
```

**为什么要提升到顶层？**
- 后端（`backend_dummy.py`, `backend_nanobind.py`）期望 `max_iters` 和 `random_seed` 在字典的顶层
- 后端代码：`max_iters = settings.get("max_iters", 10)` - 直接从顶层读取
- 如果只在 `extras` 中，后端需要写 `settings["extras"]["max_iters"]`，这样不方便

#### 阶段 3：传递给后端

```python
# solve() 函数调用后端
from .backend_dummy import solve_impl

settings = cfg.to_dict()  # 获取字典
result = solve_impl(V, C, settings, callbacks=None)
```

**后端如何使用：**

```python
# backend_dummy.py
def solve_impl(V, C, settings, callbacks):
    # 直接从顶层读取
    max_iters = settings.get("max_iters", 10)      # ← 从顶层读取
    random_seed = settings.get("random_seed", 42)  # ← 从顶层读取
    
    # 使用这些参数运行求解
    for i in range(max_iters):
        # ... 求解逻辑 ...
```

### 完整的数据流示例

```python
# 1. 用户创建配置
cfg = SimulationConfig(
    pde="linear_elasticity",
    materials={"E": 1e6, "nu": 0.3},
    extras={"max_iters": 10, "random_seed": 42}
)

# 2. 转换为字典（内部自动调用）
settings = cfg.to_dict()
# settings = {
#     "pde": "linear_elasticity",
#     "discr_order": 1,
#     "materials": {"E": 1e6, "nu": 0.3},
#     "boundary_conditions": {},
#     "extras": {"max_iters": 10, "random_seed": 42},  # 保留原始 extras
#     "max_iters": 10,        # 提升到顶层，方便后端读取
#     "random_seed": 42       # 提升到顶层，方便后端读取
# }

# 3. solve() 函数调用后端
result = solve(V, C, cfg)  # 内部会调用 cfg.to_dict() 获取 settings

# 4. 后端使用 settings
# backend_dummy.py 中：
# max_iters = settings.get("max_iters", 10)  # 从顶层读取，得到 10
```

### 设计原因

**设计原因 1：类型安全 vs 灵活性**
- **`SimulationConfig` 是类型安全的**：有固定的字段，IDE 可以自动补全
- **`extras` 提供灵活性**：可以存储任意额外参数，不需要修改类定义
- **`to_dict()` 提供转换**：将类型安全的对象转换为灵活的字典

**设计原因 2：后端兼容性**
- **后端需要简单的字典**：后端代码期望简单的 `dict`，不需要复杂的对象
- **提升常用参数**：`max_iters` 和 `random_seed` 是常用参数，提升到顶层方便使用
- **保留原始数据**：`extras` 仍然保留，以防需要访问其他额外参数

**设计原因 3：支持完整 JSON 配置**
```python
# 如果从完整 JSON 加载
cfg = SimulationConfig.from_json_file("config.json")
# 完整 JSON 存储在 extras["_full_json_config"]
# to_dict() 会直接返回完整 JSON，而不是构造的字典
```

---

## 参数输入方式

### 方式 1：使用 Class（推荐 - IDE 自动补全支持）

为了提供更好的 IDE 自动补全支持和类型检查，`SimulationConfig` 的主要参数支持使用 class 形式。

**基本示例：**
```python
from polyfempy.api.config import SimulationConfig, Material, BoundaryConditions, GravityParams

# 使用 Material class
material = Material(E=2100, nu=0.3, rho=1.0)
cfg = SimulationConfig(materials=material)

# 使用 BoundaryConditions class
bc = BoundaryConditions()
bc.add_dirichlet(id=4, value=[0.0, 0.0])
cfg = SimulationConfig(boundary_conditions=bc)

# 使用 ProblemParams classes
gravity_params = GravityParams(force=0.1)
cfg = SimulationConfig(problem_type="Gravity", problem_params=gravity_params)
```

**优势**：
- ✅ IDE 自动补全：输入 `material.` 时，IDE 会提示所有可用属性
- ✅ 类型检查：IDE 可以验证参数类型
- ✅ 错误预防：拼写错误会被 IDE 发现
- ✅ 代码清晰：`material.E` 比 `materials["E"]` 更清晰

**详细说明**：请参考 [Config Classes 使用指南](config-classes.md)，了解所有可用的 Classes、便捷方法和完整示例。

### 方式 2：使用 Dict（向后兼容）

```python
# 旧方式仍然支持（但 IDE 无法自动补全）
cfg = SimulationConfig(
    pde="linear_elasticity",
    discr_order=2,
    materials={"E": 1e6, "nu": 0.3},  # ⚠️ IDE 无法提示键名
    boundary_conditions={
        "dirichlet_boundary": [{"id": 4, "value": [0.0, 0.0]}],  # ⚠️ IDE 无法提示
        "rhs": [1.0, 0.0]
    },
    problem_params={"force": 0.1}  # ⚠️ IDE 无法提示键名
)
```

**注意**：虽然 dict 方式仍然支持，但建议使用 class 方式以获得更好的开发体验。

### 方式 3：通过 extras 输入额外参数

```python
cfg = SimulationConfig(
    pde="linear_elasticity",
    materials={"E": 1e6, "nu": 0.3},
    extras={
        # 常用参数（会被验证和提升到顶层）
        "max_iters": 10,
        "random_seed": 42,
        
        # 其他 JSON 参数（保留在 extras 中）
        "solver": {
            "linear": {"max_iter": 1000, "tolerance": 1e-6},
            "nonlinear": {"max_iter": 50}
        },
        "time": {
            "t0": 0.0,
            "tend": 1.0,
            "dt": 0.01
        }
    }
)
```

### 方式 4：从 JSON 文件加载（推荐用于完整配置）

```python
# 从 JSON 文件加载完整配置
cfg = SimulationConfig.from_json_file("config.json")
# 所有参数都保存在 extras["_full_json_config"] 中
```

### 方式 5：从字典加载

```python
config_dict = {
    "pde": "LinearElasticity",
    "materials": [{"type": "LinearElasticity", "E": 1e6, "nu": 0.3}],
    "solver": {...},
    "time": {...}
}
cfg = SimulationConfig.from_json_dict(config_dict)
```

---

## Class-Based 配置详解

为了提供更好的 IDE 自动补全支持和类型检查，`SimulationConfig` 的主要参数支持使用 class 形式（如 `Material`、`BoundaryConditions`、`GravityParams` 等）。

**详细文档**：请参考 [Config Classes 使用指南](config-classes.md)，了解：
- 为什么使用 Class 而不是 Dict
- 所有可用的 Classes（Material、BoundaryConditions、ProblemParams 等）
- 便捷方法和 Convenience Factories
- 完整的使用示例

---

## 参数验证机制

### 验证层级

#### 1. SimulationConfig 层面的验证

**会被检查的参数：**
- `discr_order`：必须是正整数（通过 `validate()` 方法）
- `materials['E']` 和 `materials['nu']`：必须是数字（通过 `validate()` 方法）

**示例：**
```python
# ✅ 会被检查
cfg = SimulationConfig(discr_order=-1)  # ValueError: discr_order must be positive
cfg = SimulationConfig(materials={"E": "abc"})  # ValueError: materials['E'] must be a number
```

#### 2. to_dict() 层面的验证

**当前行为：**
- `to_dict()` **会验证和转换** `extras` 中的常用参数
- `max_iters`：必须是正整数，字符串 `"10"` 会自动转换为整数 `10`
- `random_seed`：必须是整数或 None，字符串 `"42"` 会自动转换为整数 `42`
- 无效值会立即抛出 `ValueError`，提供清晰的错误信息

**示例：**
```python
# ✅ 字符串会自动转换为整数
cfg = SimulationConfig(extras={"max_iters": "10"})
d = cfg.to_dict()
print(d["max_iters"])  # 10 (整数)
print(type(d["max_iters"]))  # <class 'int'>

# ❌ 无效字符串会立即报错
cfg = SimulationConfig(extras={"max_iters": "abc"})
d = cfg.to_dict()  # ValueError: extras['max_iters'] must be a positive integer, got 'abc'

# ❌ 负数会立即报错
cfg = SimulationConfig(extras={"max_iters": -5})
d = cfg.to_dict()  # ValueError: extras['max_iters'] must be a positive integer, got -5
```

### 验证规则

1. **`max_iters`**：
   - 必须是正整数
   - 字符串 `"10"` 会自动转换为整数 `10`
   - 负数或无效字符串会抛出 `ValueError`

2. **`random_seed`**：
   - 必须是整数或 `None`
   - 字符串 `"42"` 会自动转换为整数 `42`
   - 无效字符串会抛出 `ValueError`

3. **其他 JSON 参数**（`solver`, `time`, `output`, `contact`, `geometry` 等）：
   - 可以通过 `extras` 输入
   - 不会被验证（类型检查）
   - 保留在 `extras` 中，不会提升到顶层
   - 如果使用完整 JSON 模式（从文件加载），所有参数都会保留
   - 需要后端（C++ 绑定）支持才能使用

### 未知参数的处理

**设计行为：**
- 未知参数不会被验证
- 保留在 `extras` 中，但不会提升到顶层
- 不会报错（允许自定义参数）

**示例：**
```python
# 用户输入了拼写错误
cfg = SimulationConfig(extras={"max_iter": 10})  # 应该是 max_iters

# 不会报错，但参数被忽略（使用默认值）
settings = cfg.to_dict()
# settings["max_iters"] 不存在，使用默认值 10
```

**为什么这样设计？**
- 允许用户添加自定义参数到 `extras`，这些参数可能被其他工具使用
- 只验证已知的关键参数（`max_iters`, `random_seed`）

---

## JSON 参数支持

### PolyFEM JSON 配置的完整参数列表

根据 `from_json_dict()` 的文档，PolyFEM JSON 配置支持以下所有参数：

#### 1. 基础参数
- `pde`: PDE 类型（"Poisson", "LinearElasticity", "NonLinearElasticity" 等）
- `discr_order`: 离散化阶数（1, 2, ...）
- `space`: 空间配置（包含 `discr_order` 等）

#### 2. 材料参数
- `materials`: 材料配置（数组或字典格式）
  - `type`: 材料类型（"LinearElasticity", "NeoHookean", "SaintVenant" 等）
  - `E`: 杨氏模量
  - `nu`: 泊松比
  - `rho`: 密度
  - `id`: 材料 ID

#### 3. 边界条件
- `boundary_conditions`: 边界条件配置
  - `dirichlet_boundary`: Dirichlet 边界条件
  - `neumann_boundary`: Neumann 边界条件
  - `pressure`: 压力边界条件
  - `rhs`: 体力和右端项

#### 4. 求解器参数
- `solver`: 求解器配置
  - `linear`: 线性求解器配置
    - `max_iter`: 最大迭代次数
    - `tolerance`: 容差
  - `nonlinear`: 非线性求解器配置
    - `max_iter`: 最大迭代次数
    - `tolerance`: 容差
  - `max_threads`: 最大线程数

#### 5. 时间参数（瞬态问题）
- `time`: 时间配置
  - `t0`: 初始时间
  - `tend`: 结束时间
  - `dt`: 时间步长
  - `time_steps`: 时间步数
  - `integrator`: 积分器类型（"ImplicitEuler", "ExplicitEuler" 等）

#### 6. 输出参数
- `output`: 输出配置
  - `directory`: 输出目录
  - `paraview`: ParaView 输出配置
  - `json`: JSON 输出配置

#### 7. 接触参数
- `contact`: 接触配置
  - `enabled`: 是否启用接触
  - `dhat`: 接触距离阈值
  - `mu`: 摩擦系数
  - `epsv`: 粘性参数

#### 8. 几何参数
- `geometry`: 几何配置
  - `mesh`: 网格文件路径
  - `transformations`: 变换
  - `selections`: 选择

### 如何输入这些参数

这些 JSON 参数可以通过以下方式输入（详细说明请参考 [参数输入方式](#参数输入方式) 章节）：

- **从 JSON 文件加载**：`SimulationConfig.from_json_file("config.json")`
- **通过 extras 输入**：`SimulationConfig(extras={"solver": {...}, "time": {...}})`
- **从字典加载**：`SimulationConfig.from_json_dict(config_dict)`

### to_dict() 的行为

1. **如果有完整 JSON 配置**（`extras["_full_json_config"]`）：
   ```python
   # 直接返回完整 JSON，包含所有参数
   return dict(self.extras["_full_json_config"])
   ```

2. **如果没有完整 JSON 配置**：
   ```python
   # 只构造基本字段
   result = {
       "pde": ...,
       "discr_order": ...,
       "materials": ...,
       "boundary_conditions": ...,
   }
   
   # 从 extras 提升常用参数到顶层
   if "max_iters" in extras:
       result["max_iters"] = extras["max_iters"]  # 验证和转换
   
   # 其他 extras 参数保留在 extras 中
   result["extras"] = dict(extras)
   ```

---

## 实际应用示例

### 与 polyfem-data 示例的兼容性

✅ **API 设计成功**：来自 `polyfem-data/contact/examples/2D` 和 `3D` 的所有 86 个 JSON 配置文件都可以使用新 API 成功加载和配置。
- **2D 示例**：28/28 成功
- **3D 示例**：58/58 成功

### 测试结果

#### 配置加载
- **86/86 成功** - 所有 JSON 文件都可以加载到 `SimulationConfig`
  - 2D: 28/28 成功
  - 3D: 58/58 成功
- 所有参数都保存在 `extras["_full_json_config"]` 中
- 未检测到缺失参数

#### 求解器配置
- **86/86 成功** - 所有配置都可以用于配置求解器
  - 2D: 28/28 成功
  - 3D: 58/58 成功
- 所有参数都正确传递给 `solver.set_settings()`

#### 问题类型统计
- **80/86** 是瞬态问题（需要时间步进）
  - 2D: 27/28 瞬态
  - 3D: 53/58 瞬态
- **58/86** 启用了接触
  - 2D: 20/28 有接触
  - 3D: 38/58 有接触
- **6/86** 是静态问题

### common.json 支持

✅ **common.json 自动合并**：API 完全支持 `common.json` 引用和合并功能。

#### 合并规则

1. **自动检测**：如果配置文件中包含 `"common": "路径"` 引用，会自动加载并合并
2. **深层合并**：嵌套字典会递归合并（如 `output.paraview` 会合并 `file_name` 和 `options`）
3. **优先级**：原始配置的值会覆盖 `common.json` 中的默认值
4. **自动移除**：合并后 `common` 键会被自动移除

#### 示例

```json
// common.json
{
  "contact": {"enabled": true, "dhat": 0.001},
  "output": {
    "paraview": {
      "options": {"material": true}
    }
  }
}

// 5-squares.json
{
  "common": "../../common.json",
  "output": {
    "paraview": {
      "file_name": "5-squares.pvd"
    }
  }
}

// 合并后
{
  "contact": {"enabled": true, "dhat": 0.001},
  "output": {
    "paraview": {
      "file_name": "5-squares.pvd",  // 来自原始配置
      "options": {"material": true}   // 来自 common.json
    }
  }
}
```

**测试结果**：86 个示例文件中，所有引用 `common.json` 的文件都能正确合并。

### 加载现有示例

所有示例都可以使用以下方式加载：

```python
from polyfempy.api import SimulationConfig, solve

# 方法 1：从文件路径加载
cfg = SimulationConfig.from_json_file("data/contact/examples/2D/unit-tests/5-squares.json")

# 方法 2：从字典加载
import json
with open("data/contact/examples/2D/unit-tests/5-squares.json") as f:
    config_dict = json.load(f)
cfg = SimulationConfig.from_json_dict(config_dict)

# 方法 3：直接求解（适用于静态问题）
result = solve(cfg="data/contact/examples/2D/unit-tests/5-squares.json")
```

### 支持的参数（已验证）

通过测试 86 个实际示例，确认 API 支持 PolyFEM JSON 格式中的所有参数：

- ✅ **geometry**：网格文件、变换、volume_selection、surface_selection
- ✅ **materials**：所有材料类型（LinearElasticity、NeoHookean 等）以及 E、nu、rho 等参数
- ✅ **boundary_conditions**：Dirichlet、Neumann、pressure、RHS 等
- ✅ **time**：瞬态设置（t0、tend、dt、integrator）
- ✅ **contact**：接触设置（enabled、dhat、mu、epsv 等）
- ✅ **solver**：线性/非线性求解器设置
- ✅ **output**：Paraview、JSON 输出设置
- ✅ **space**：离散化阶数（支持列表格式，如 `[{"id": 2, "order": 2}]`）
- ✅ **common**：JSON 引用（自动合并，支持深层嵌套合并）

### 瞬态问题处理

**重要提示**：大多数示例（80/86）是需要时间步进的瞬态问题。当前的 `solve()` 函数可以自动处理静态问题，但对于瞬态问题，需要使用底层 API：

```python
import polyfempy as pf
import json
from polyfempy.api import SimulationConfig

# 加载配置
cfg = SimulationConfig.from_json_file("data/contact/examples/2D/unit-tests/5-squares.json")
full_json = cfg.extras["_full_json_config"]

# 配置求解器
solver = pf.Solver()
solver.set_settings(json.dumps(full_json), strict_validation=False)
solver.load_mesh_from_settings()
solver.build_basis()
solver.assemble()

# 时间步进
config = solver.settings()
t0 = config["time"]["t0"]
dt = config["time"]["dt"]
sol = solver.init_timestepping(t0, dt)

# 运行时间步
for i in range(1, 5):
    for t in range(1):
        sol = solver.step_in_time(sol, t0, dt, t+1)
    t0 += dt
    solver.export_vtu(f"step_{i}.vtu", sol, np.zeros((0, 0)), t0, dt)
```

### 未来增强

为了在高级 `solve()` API 中完全支持瞬态问题，可以考虑：

1. **自动检测瞬态问题**：检查 `time` 配置
2. **时间步进循环**：自动处理 `init_timestepping()` 和 `step_in_time()`
3. **结果聚合**：为瞬态问题返回时间序列结果

---

## 添加新参数

### 当前设计

使用 `_EXTRAS_PROMOTION_RULES` 字典来配置哪些参数需要提升，以及如何验证和转换它们。

### 如何添加新参数

#### 步骤 1：定义验证器函数（如果需要）

```python
def _validate_positive_float(v):
    """Validate and convert to positive float."""
    v = float(v)
    if v <= 0:
        raise ValueError("must be positive")
    return v
```

#### 步骤 2：在 `_EXTRAS_PROMOTION_RULES` 中添加配置

在 `polyfempy/api/config.py` 中找到 `_EXTRAS_PROMOTION_RULES` 字典，添加新参数：

```python
_EXTRAS_PROMOTION_RULES = {
    "max_iters": (
        _validate_positive_int,
        "extras['max_iters'] must be a positive integer, got {value!r} (type: {type_name})"
    ),
    "random_seed": (
        _validate_int_or_none,
        "extras['random_seed'] must be an integer or None, got {value!r} (type: {type_name})"
    ),
    # 添加新参数：只需要这一行！
    "tolerance": (
        _validate_positive_float,
        "extras['tolerance'] must be a positive float, got {value!r} (type: {type_name})"
    ),
}
```

#### 步骤 3：完成

无需修改 `to_dict()` 方法，系统会自动处理。

### 常见验证器模式

#### 模式 1：正整数
```python
def _validate_positive_int(v):
    v = int(v)
    if v <= 0:
        raise ValueError("must be positive")
    return v
```

#### 模式 2：正浮点数
```python
def _validate_positive_float(v):
    v = float(v)
    if v <= 0:
        raise ValueError("must be positive")
    return v
```

#### 模式 3：可选整数（允许 None）
```python
def _validate_int_or_none(v):
    if v is None:
        return None
    return int(v)
```

#### 模式 4：枚举值（字符串）
```python
def _validate_solver_type(v):
    v = str(v)
    if v not in ["linear", "nonlinear", "mixed"]:
        raise ValueError("must be 'linear', 'nonlinear', or 'mixed'")
    return v
```

#### 模式 5：范围检查
```python
def _validate_probability(v):
    v = float(v)
    if not (0.0 <= v <= 1.0):
        raise ValueError("must be between 0 and 1")
    return v
```

### 完整示例

假设要添加 `tolerance` 参数：

```python
# 1. 定义验证器（在 _EXTRAS_PROMOTION_RULES 之前）
def _validate_positive_float(v):
    v = float(v)
    if v <= 0:
        raise ValueError("must be positive")
    return v

# 2. 在 _EXTRAS_PROMOTION_RULES 中添加
_EXTRAS_PROMOTION_RULES = {
    # ... 现有参数 ...
    "tolerance": (
        _validate_positive_float,
        "extras['tolerance'] must be a positive float, got {value!r} (type: {type_name})"
    ),
}

# 3. 使用
cfg = SimulationConfig(extras={"tolerance": "1e-6"})
d = cfg.to_dict()
# d["tolerance"] = 1e-6 (自动转换为 float)
```

### 优势

- ✅ **可扩展性**：添加新参数只需要在字典中添加一行
- ✅ **一致性**：所有参数使用相同的验证和提升机制
- ✅ **可维护性**：所有参数配置集中在一个地方
- ✅ **类型安全**：自动类型转换（字符串 → 数字）

---

## 总结

### 数据流

```
用户输入 (extras) 
  → SimulationConfig.extras 
  → to_dict() (验证、转换、提升到顶层) 
  → settings 字典 
  → 后端使用
```

### 关键点

- **`extras`**：用户输入的额外参数（灵活性）
- **`to_dict()`**：负责验证、转换和提升（兼容性）
- **`settings`**：后端使用的格式（简单性）

### 验证状态

- ✅ **已验证和转换**：`max_iters`, `random_seed`
- ⚠️ **未验证但支持**：所有其他 JSON 参数（通过完整 JSON 模式）
- ❌ **当前限制**：通过 `extras` 输入的参数（除了常用参数）不会被验证

### 使用建议

1. **简单配置**：使用基本字段 + `extras` 中的常用参数
2. **完整配置**：从 JSON 文件加载
3. **需要额外参数**：使用完整 JSON 字典

---


