# Differentiable 功能完整指南

本文档详细介绍了 PolyFEM Python API 中的可微分仿真功能，包括设计理念、使用方法、技术细节和最佳实践。

## 目录

1. [快速开始](#快速开始)
2. [概述](#概述)
3. [为什么需要 Differentiable？](#为什么需要-differentiable)
4. [技术原理](#技术原理)
5. [API 设计](#api-设计)
6. [使用指南](#使用指南)
7. [新实现 vs 旧实现](#新实现-vs-旧实现)
8. [支持的导数类型](#支持的导数类型)
9. [完整示例](#完整示例)
10. [高级用法](#高级用法)
11. [常见问题](#常见问题)
12. [实现细节](#实现细节)

---

## 快速开始

### 最简单的例子（5 行代码）

```python
from polyfempy.differentiable import solve_differentiable
import torch
import numpy as np

# Prepare mesh and configuration
V = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]], dtype=np.float64)
C = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
cfg = {
    "materials": [{"type": "LinearElasticity", "E": 1e6, "nu": 0.3}],
    "boundary_conditions": {
        "dirichlet_boundary": [{"selection": 1, "value": [0.0, 0.0]}],
        "neumann_boundary": [{"selection": 2, "value": [0.0, -1000.0]}],
    }
}

# Run differentiable simulation (only 5 lines!)
vertices = torch.tensor(V, requires_grad=True)
result = solve_differentiable(vertices, C, cfg)
loss = torch.norm(result.u)
loss.backward()
grad = vertices.grad   # Gradient is computed automatically!
```

**就这么简单！** 不需要手动写 `torch.autograd.Function`，不需要手动设置缓存，不需要手动调用 adjoint。

---

## 概述

### 什么是 Differentiable？

Differentiable（可微分）功能允许你计算仿真结果对输入参数的梯度，这对于以下场景非常重要：

- **形状优化**：优化结构形状以最小化应力或位移
- **材料参数优化**：找到最佳材料参数
- **逆问题求解**：从观测数据反推参数
- **机器学习**：将物理仿真集成到神经网络训练中

### 设计原则

1. **默认简单**：`solve()` API 保持简单，不涉及 differentiable
2. **可选增强**：提供独立的 differentiable 模块
3. **渐进式复杂度**：从简单到高级，用户按需选择
4. **向后兼容**：现有的复杂用法仍然支持

---

## 为什么需要 Differentiable？

Differentiable 功能主要适用于需要计算梯度进行优化的场景。例如：

- **形状优化**：找到最优的网格顶点，使位移或应力最小
- **材料参数优化**：找到最佳的材料参数（如弹性模量 E、泊松比 ν）
- **逆问题求解**：从观测数据反推未知参数
- **机器学习训练**：将物理仿真作为神经网络的一层

**典型使用场景**：
```python
# 目标：找到最优的 vertices，使得位移最小
def objective(vertices):
    result = solve_differentiable(vertices, C, cfg)  # 可微分仿真
    return torch.norm(result.u)  # 计算损失

# 需要计算梯度：d(objective)/d(vertices)
# 这样才能用梯度下降优化
```

### 为什么不能直接用 PyTorch 的自动微分？

**PDE 求解器太大，自动微分不适用：**

- 百万自由度：存储所有中间计算需要巨大内存
- 迭代求解：非线性求解器需要多次迭代，自动微分会记录每一步
- 效率问题：自动微分的时间复杂度通常是前向计算的 3-5 倍

### 解决方案：伴随方法（Adjoint Method）

**伴随方法只需要一次反向求解，效率高：**

1. **前向传播**：运行一次仿真，求解 PDE
2. **反向传播**：求解一次伴随方程，直接得到梯度
3. **内存效率**：只需要存储必要的中间结果，而不是所有计算

这就是为什么我们需要 `torch.autograd.Function` 来自定义反向传播！

---

## 技术原理

### 核心思想：分层设计 + 职责分离

```
用户层（简单易用）
    ↓
solve_differentiable() - 高级封装
    ↓
PolyFEMFunction - PyTorch 集成层（自定义反向传播）
    ↓
底层 C++ API (pf.Solver) - 实际计算
```

### 为什么必须使用 `torch.autograd.Function`？

`torch.autograd.Function` 允许我们：

1. **自定义前向传播**：运行 PolyFEM 仿真
2. **自定义反向传播**：使用伴随方法计算梯度（而不是自动微分）
3. **与 PyTorch 集成**：让 PolyFEM 仿真成为 PyTorch 计算图的一部分

**关键点**：
- 必须继承 `torch.autograd.Function`
- 必须实现 `forward()` 和 `backward()` 静态方法
- `backward()` 返回的梯度数量必须与 `forward()` 的输入数量匹配

### 模块架构

#### 1. `__init__.py` - 模块入口和依赖管理

**职责**：
- ✅ 依赖检查：检查 PyTorch 是否安装
- ✅ 统一导出：提供清晰的 API 接口
- ✅ 优雅降级：如果没有 PyTorch，给出清晰的错误提示

#### 2. `solve.py` - 高级封装（用户主要接口）

**职责**：
- ✅ 统一接口：提供与 `solve()` 一致的 API 风格
- ✅ 输入处理：自动处理 numpy/torch 类型转换
- ✅ 配置管理：统一处理 SimulationConfig 和 dict
- ✅ Solver 生命周期管理：创建、设置、运行、清理

#### 3. `torch_integration.py` - PyTorch 集成层

**职责**：
- ✅ 封装 PyTorch Function：实现 forward/backward
- ✅ 自动处理缓存：自动设置 CacheLevel.Derivatives
- ✅ 自动调用 adjoint：自动调用 solve_adjoint
- ✅ 自动计算导数：自动调用导数函数

#### 4. `result.py` - 结果容器

**职责**：
- ✅ 封装结果：包含解、solver、元数据
- ✅ PyTorch 支持：包含 torch.Tensor，支持 .backward()
- ✅ 类型转换：提供 to_numpy() 方法

#### 5. `helpers.py` - 辅助工具

**职责**：
- ✅ 减少样板代码：提供常用场景的封装
- ✅ 梯度验证：提供梯度检查工具

### 伴随方法（Adjoint Method）详解

#### 数学原理

对于优化问题：
```
minimize J(u(θ), θ)
subject to F(u(θ), θ) = 0  (PDE 约束)
```

其中：
- `u` 是 PDE 的解（位移场）
- `θ` 是优化参数（如顶点坐标）
- `J` 是目标函数（如总位移）
- `F` 是 PDE 残差

**直接计算梯度**：`dJ/dθ = ∂J/∂θ + (∂J/∂u) · (du/dθ)`

计算 `du/dθ` 需要求解线性系统，对于每个参数都要解一次，非常昂贵。

**伴随方法**：引入伴随变量 `λ`，满足：
```
F_u^T · λ = -∂J/∂u
```

然后梯度变为：
```
dJ/dθ = ∂J/∂θ + λ^T · F_θ
```

**优势**：只需要求解一次伴随方程，无论有多少参数！

#### 工作流程

1. **前向传播（forward）**：
   ```python
   def forward(ctx, solver, vertices, derivative_type):
       # 1. 更新网格顶点
       solver.mesh().set_vertices(vertices)
       
       # 2. 启用导数缓存（关键！）
       solver.set_cache_level(pf.CacheLevel.Derivatives)
       
       # 3. 运行仿真，求解 PDE
       solver.solve()
       
       # 4. 保存 solver 到 ctx（用于 backward）
       ctx.solver = solver
       
       # 5. 返回解
       return torch.tensor(solver.get_solutions())
   ```

2. **反向传播（backward）**：
   ```python
   def backward(ctx, grad_output):
       # grad_output = d(loss)/d(solution)
       
       # 1. 求解伴随方程
       ctx.solver.solve_adjoint(grad_output.numpy())
       
       # 2. 计算参数导数（使用伴随解）
       if ctx.derivative_type == "shape":
           grad = pf.shape_derivative(ctx.solver)
       elif ctx.derivative_type == "material":
           grad = pf.elastic_material_derivative(ctx.solver)
       
       # 3. 返回梯度
       return None, torch.tensor(grad), None
   ```

### Differentiable 与 Backend 的关系

#### 为什么 Differentiable 需要 `nanobind` backend？

**核心原因**：Differentiable 功能需要直接访问 C++ Solver 对象和底层方法，这些功能只有通过 `nanobind` backend 才能访问。

##### 1. 需要 C++ Solver 对象

Differentiable 功能不通过新的 `solve()` API（它使用 backend SPI），而是直接使用旧的 `pf.Solver()` API：

```python
# solve_differentiable() 内部实现
import polyfempy as pf
solver = pf.Solver()  # 直接创建 C++ Solver 对象
solver.set_mesh(V, C)
solver.build_basis()
solver.assemble()
solver.solve()
```

**为什么？** 因为需要：
- 在 `forward()` 和 `backward()` 之间保持 solver 状态
- 直接调用 C++ 方法（如 `solve_adjoint()`）
- 访问内部缓存和中间结果

##### 2. 需要 C++ 伴随方法

反向传播需要调用 C++ 实现的伴随方法：

```python
# backward() 中必须调用
ctx.solver.solve_adjoint(grad_output)  # C++ 方法
grad = pf.shape_derivative(ctx.solver)  # C++ 方法
```

这些方法只有 C++ 实现，必须通过 `nanobind` backend 访问。

##### 3. 需要导数计算功能

Differentiable 需要计算各种导数：
- `pf.shape_derivative()`：形状导数
- `pf.elastic_material_derivative()`：材料参数导数
- `pf.initial_velocity_derivative()`：初始速度导数

这些都需要真实的刚度矩阵、残差等中间结果，只有 `nanobind` backend 能提供。

#### Backend 检查机制

`solve_differentiable()` 会强制检查 backend：

```python
def solve_differentiable(..., backend: str = "nanobind"):
    if backend != "nanobind":
        raise ValueError(
            f"Differentiable simulations require 'nanobind' backend, got '{backend}'. "
            "Please use backend='nanobind' and ensure the C++ module is built."
        )
    
    # 检查 C++ 模块是否可用
    try:
        import polyfempy as pf
    except ImportError:
        raise ImportError(
            "PolyFEM C++ module is required for differentiable simulations. "
            "Please build the C++ module first."
        )
```

#### 架构对比

**普通 `solve()` API**：
```
用户代码
    ↓
solve() - 统一接口
    ↓
Backend SPI (backend_base.py)
    ↓
backend_nanobind.py
    ↓
返回 Result 对象
```

**Differentiable API**：
```
用户代码
    ↓
solve_differentiable() - 高级封装
    ↓
直接使用 pf.Solver() (C++ API)
    ↓
PolyFEMFunction (PyTorch 集成)
    ↓
返回 DifferentiableResult 对象
```

**关键区别**：
- `solve()` 通过 backend SPI，使用统一的接口
- `solve_differentiable()` 直接使用 C++ API，绕过 backend SPI

#### 为什么不能通过 Backend SPI？

理论上可以让 `backend_nanobind.py` 暴露 Solver 对象，但这样会：
1. **破坏封装**：Backend SPI 设计为黑盒接口，不应该暴露内部对象
2. **增加复杂度**：需要在 SPI 中增加特殊方法
3. **不必要**：Differentiable 是高级功能，直接使用 C++ API 更清晰

#### 使用建议

1. **确保 C++ 模块已编译**：
   - `solve_differentiable()` 需要 C++ 模块 `polyfem_nb`
   - 如果未编译，会给出清晰的错误提示

2. **错误处理**：
   - 如果 C++ 模块未编译，`solve_differentiable()` 会抛出 `ImportError`
   - 如果传入错误的 backend，会立即抛出 `ValueError`

#### 总结

| 方面 | `solve()` | `solve_differentiable()` |
|------|-----------|-------------------------|
| **Backend** | `nanobind` | `nanobind` |
| **API 路径** | Backend SPI | 直接 C++ API |
| **Solver 对象** | 不暴露 | 直接使用 `pf.Solver()` |
| **用途** | 普通仿真 | 梯度计算、优化 |
| **依赖** | 需要 C++ 模块 | 需要 C++ 模块 |

---

## API 设计

### 推荐方案：独立模块（Plan B）

**核心设计**：

```python
# 1. 基础 API 保持不变
from polyfempy.api import solve
result = solve(V, C, cfg)  # 简单，清晰

# 2. Differentiable 作为独立模块
from polyfempy.differentiable import solve_differentiable
result = solve_differentiable(V, C, cfg)  # 明确，专业
```

**目录结构**：

```
polyfempy/
├── api/                    # 基础 API（所有用户）
│   ├── solve.py
│   ├── config.py
│   └── ...
│
└── differentiable/          # 可选模块（高级用户）
    ├── __init__.py
    ├── solve.py             # solve_differentiable() 函数
    ├── torch_integration.py # PolyFEMFunction 封装
    ├── result.py            # DifferentiableResult 类
    ├── helpers.py           # 辅助工具
    └── examples/            # 示例代码
```

### 为什么选择独立模块？

#### 对比：集成到 solve() vs 独立模块

| 方面 | 集成到 solve() | 独立模块（推荐） |
|------|---------------|----------------|
| API 统一性 | ✅ 统一接口 | ✅ 与新 API 统一 |
| 代码量 | ❌ 函数变复杂 | ✅ 职责清晰 |
| 依赖管理 | ❌ 所有用户都需要 PyTorch | ✅ 只有需要的用户导入 |
| 向后兼容 | ❌ 可能影响现有用户 | ✅ 不影响现有用户 |
| 易于维护 | ❌ 逻辑混合 | ✅ 逻辑分离 |

**结论**：独立模块更好，因为它不影响 90% 的用户，同时为 10% 的用户提供清晰的接口。

---

## 使用指南

### 基本使用

详细的基本使用示例请参见 [快速开始](#快速开始) 部分。以下是关键步骤：

1. 准备数据（网格顶点 `V`、单元连接 `C`、配置 `cfg`）
2. 将顶点转换为 PyTorch Tensor 并设置 `requires_grad=True`
3. 调用 `solve_differentiable()` 运行可微分仿真
4. 定义损失函数并调用 `.backward()` 计算梯度
5. 从 `.grad` 属性获取梯度

---

## 新实现 vs 旧实现

### 详细对比

#### 旧实现（20+ 行，容易出错）

```python
import polyfempy as pf
import torch

# 用户必须手动写 Function 类
class Simulate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, solver, vertices):
        # 1. 更新网格
        solver.mesh().set_vertices(vertices.detach().cpu().numpy())
        
        # 2. 设置缓存（⚠️ 容易忘记！）
        solver.set_cache_level(pf.CacheLevel.Derivatives)
        
        # 3. 运行仿真
        solver.solve()
        
        # 4. 保存状态
        sol = torch.tensor(solver.get_solutions())
        ctx.solver = solver
        return sol
    
    @staticmethod
    def backward(ctx, grad_output):
        # 1. 求解伴随问题（⚠️ 容易忘记！）
        ctx.solver.solve_adjoint(grad_output.detach().cpu().numpy())
        
        # 2. 计算导数
        return None, torch.tensor(pf.shape_derivative(ctx.solver))

# 使用（需要手动配置 solver）
solver = pf.Solver()
solver.set_settings(json.dumps(cfg))
solver.set_mesh(V, C)
solver.build_basis()
solver.assemble()

vertices = torch.tensor(V, requires_grad=True)
result = Simulate.apply(solver, vertices)
loss = torch.norm(result)
loss.backward()
grad = vertices.grad
```

**问题**：
- ❌ 代码量大（20+ 行）
- ❌ 容易忘记设置缓存
- ❌ 容易忘记调用 adjoint
- ❌ 需要手动配置 solver
- ❌ 与 `solve()` API 不一致

#### 新实现（5 行，自动处理）

```python
from polyfempy.differentiable import solve_differentiable
import torch

# 使用（5 行代码！）
vertices = torch.tensor(V, requires_grad=True)
result = solve_differentiable(vertices, C, cfg)  # ✅ 自动处理一切
loss = torch.norm(result.u)
loss.backward()
grad = vertices.grad  # ✅ 自动计算梯度
```

**优势**：
- ✅ 代码量减少 75%（从 20+ 行到 5 行）
- ✅ 自动设置缓存（不会忘记）
- ✅ 自动调用 adjoint（不会忘记）
- ✅ 自动配置 solver（无需手动）
- ✅ API 与 `solve()` 完全一致
- ✅ 支持多种导数类型
- ✅ 清晰的错误提示

### 功能对比表

| 功能 | 旧实现 | 新实现 | 说明 |
|------|--------|--------|------|
| **代码量** | 20+ 行 | 5 行 | 减少 75% |
| **设置缓存** | 手动，容易忘记 | 自动 | ✅ 改进 |
| **调用 adjoint** | 手动，容易忘记 | 自动 | ✅ 改进 |
| **配置 solver** | 手动，复杂 | 自动 | ✅ 改进 |
| **形状优化** | ✅ | ✅ | 都支持，新实现更简单 |
| **材料优化** | ❌ | ✅ | ✅ 新功能 |
| **初始速度优化** | ✅ | ✅ | 都支持 |
| **API 统一性** | ❌ | ✅ | ✅ 新优势 |
| **错误处理** | 基础 | 完善 | ✅ 改进 |
| **文档和示例** | 少 | 完整 | ✅ 改进 |

### 为什么新实现更好？

1. **减少错误**：自动处理所有细节，不会忘记关键步骤
2. **提高效率**：代码量减少 75%，开发更快
3. **易于学习**：API 统一，学习成本低
4. **易于维护**：逻辑集中，易于改进
5. **功能更全**：支持更多导数类型和辅助工具

---

## 支持的导数类型

`solve_differentiable()` 支持多种导数类型，通过 `derivative_type` 参数指定：

### 1. 形状导数（最常用）

计算目标函数对网格顶点的导数：`d(loss)/d(vertices)`

```python
result = solve_differentiable(vertices, C, cfg, derivative_type="shape")
loss = torch.norm(result.u)
loss.backward()
grad = vertices.grad  # 形状导数
```

### 2. 材料参数导数

计算目标函数对材料参数的导数：`d(loss)/d(material_params)`

```python
result = solve_differentiable(V, C, cfg, derivative_type="material")
# 需要将材料参数设置为 differentiable
```

### 3. 初始速度导数

计算目标函数对初始速度的导数：`d(loss)/d(initial_velocity)`

```python
result = solve_differentiable(V, C, cfg, derivative_type="initial_velocity")
```

---

## 完整示例

### 形状优化示例

完整的形状优化示例，包括梯度计算和优化循环：

```python
from polyfempy.differentiable import solve_differentiable
import torch
import numpy as np

# 准备数据
V = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]], dtype=np.float64)
C = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

cfg = {
    "materials": [{"type": "LinearElasticity", "E": 1e6, "nu": 0.3}],
    "boundary_conditions": {
        "dirichlet_boundary": [{"selection": 1, "value": [0.0, 0.0]}],
        "neumann_boundary": [{"selection": 2, "value": [0.0, -1000.0]}],
    }
}

# 创建可微分的顶点
vertices = torch.tensor(V, requires_grad=True)

# 运行可微分仿真
result = solve_differentiable(vertices, C, cfg, derivative_type="shape")

# 定义损失函数（例如：最小化总位移）
loss = torch.norm(result.u)

# 计算梯度
loss.backward()

# 获取梯度
grad = vertices.grad
print(f"Gradient shape: {grad.shape}")
print(f"Gradient norm: {torch.norm(grad).item()}")

# 使用梯度进行优化（例如：梯度下降）
learning_rate = 0.01
vertices_new = vertices - learning_rate * grad
```

---

## 高级用法

### 1. 自定义 Function（高级用户）

如果你需要完全控制前向和反向传播，可以继承 `PolyFEMFunction`：

```python
from polyfempy.differentiable import PolyFEMFunction
import torch

class MyCustomSimulate(PolyFEMFunction):
    """自定义 Function，添加额外的处理逻辑"""
    
    @staticmethod
    def forward(ctx, solver, vertices, derivative_type="shape"):
        # 可以在这里添加自定义逻辑
        # 例如：预处理顶点、记录额外信息等
        
        # 调用父类方法
        result = PolyFEMFunction.forward(ctx, solver, vertices, derivative_type)
        
        # 可以在这里添加后处理
        # 例如：计算额外的量、保存中间结果等
        
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        # 可以自定义反向传播逻辑
        # 或者直接使用父类方法
        return PolyFEMFunction.backward(ctx, grad_output)

# 使用
result = MyCustomSimulate.apply(solver, vertices, "shape")
```

### 2. 使用辅助工具

#### 创建形状优化器

```python
from polyfempy.differentiable import create_shape_optimizer

# 创建优化器函数
optimizer = create_shape_optimizer(V, C, cfg)

# 使用
vertices = torch.tensor(V, requires_grad=True)
loss, grad = optimizer(vertices)
```

#### 验证梯度

```python
from polyfempy.differentiable import gradient_check

def loss_fn(vertices):
    result = solve_differentiable(vertices, C, cfg)
    return torch.norm(result.u)

vertices = torch.tensor(V, requires_grad=True)
result = solve_differentiable(vertices, C, cfg)
loss = loss_fn(vertices)
loss.backward()
grad = vertices.grad

# 验证梯度是否正确
is_correct, error = gradient_check(loss_fn, vertices, grad)
print(f"Gradient correct: {is_correct}, relative error: {error}")
```

### 3. 与优化库集成

```python
from polyfempy.differentiable import solve_differentiable
import torch
import torch.optim as optim

# 准备数据
vertices = torch.tensor(V, requires_grad=True)

# 创建优化器
optimizer = optim.Adam([vertices], lr=0.01)

# 优化循环
for iteration in range(100):
    optimizer.zero_grad()
    
    # 运行可微分仿真
    result = solve_differentiable(vertices, C, cfg)
    
    # 计算损失
    loss = torch.norm(result.u)
    
    # 反向传播
    loss.backward()
    
    # 更新参数
    optimizer.step()
    
    print(f"Iteration {iteration}: loss = {loss.item()}")
```

---

## 常见问题

### Q1: 为什么必须使用 `torch.autograd.Function`？

**A:** 因为我们需要自定义反向传播。PyTorch 的自动微分不适合 PDE 求解器（内存消耗太大），我们需要使用伴随方法。详细说明参见 [为什么必须使用 `torch.autograd.Function`？](#为什么必须使用-torchautogradfunction) 章节。

### Q2: 新实现和旧实现有什么区别？

**A:** 主要区别：
- **代码量**：从 20+ 行减少到 5 行（减少 75%）
- **自动化**：自动处理缓存、adjoint 等，不会忘记
- **API 统一**：与 `solve()` API 完全一致
- **功能更全**：支持更多导数类型和辅助工具

详细对比和示例代码请参见 [新实现 vs 旧实现](#新实现-vs-旧实现) 章节。

### Q3: 什么时候应该使用 differentiable？

**A:** 当你需要计算梯度时：
- 形状优化
- 材料参数优化
- 逆问题求解
- 机器学习训练

如果只需要运行仿真看结果，使用普通的 `solve()` 即可。

### Q4: 性能如何？

**A:** 
- **前向传播**：与普通 `solve()` 相同，运行一次仿真
- **反向传播**：只需要一次伴随求解，无论有多少参数，非常高效
- **内存**：只需要存储必要的中间结果，比自动微分节省很多

详细性能考虑参见 [性能考虑](#性能考虑) 部分。

### Q5: 支持哪些导数类型？

**A:** 目前支持：
- `"shape"`：形状导数（最常用），计算 d(loss)/d(vertices)
- `"material"`：材料参数导数，计算 d(loss)/d(material_params)
- `"initial_velocity"`：初始速度导数，计算 d(loss)/d(initial_velocity)

使用示例请参见 [支持的导数类型](#支持的导数类型) 部分。未来可能会添加更多类型。

### Q6: 可以同时计算多种导数吗？

**A:** 目前不支持，需要分别调用。未来可能会支持。

### Q7: 如何验证梯度是否正确？

**A:** 使用 `gradient_check()` 函数，它会用有限差分法验证梯度。使用示例请参见 [使用辅助工具 - 验证梯度](#验证梯度) 部分。

### Q8: 新实现会影响性能吗？

**A:** 不会。新实现只是封装了旧实现，性能完全相同。实际上，由于自动处理了所有细节，减少了用户错误，可能更高效。

### Q9: 为什么必须使用 `nanobind` backend？

**A:** Differentiable 功能**必须**使用 `nanobind` backend，原因如下：

1. **需要 C++ Solver 对象**：Differentiable 直接使用 `pf.Solver()` C++ API，而不是通过 backend SPI
2. **需要伴随方法**：反向传播需要调用 `solve_adjoint()` 等 C++ 方法
3. **需要导数计算**：需要真实的刚度矩阵、残差等中间结果来计算导数

如果你传入错误的 backend，`solve_differentiable()` 会立即抛出 `ValueError`。

**详细说明**：参见 [Differentiable 与 Backend 的关系](#differentiable-与-backend-的关系) 章节。

---

## 实现细节

### 模块架构

模块架构和各模块职责的详细说明请参见 [技术原理 - 模块架构](#模块架构) 章节。

**模块结构**：
```
polyfempy/differentiable/
├── __init__.py              # 依赖检查和统一导出
├── solve.py                 # solve_differentiable() - 高级封装
├── torch_integration.py     # PolyFEMFunction - PyTorch 集成层
├── result.py                # DifferentiableResult - 结果容器
├── helpers.py               # 辅助工具函数
└── examples/                # 示例代码
    ├── simple_shape_optimization.py
    └── multi_solver_shape_optimization.py
```

### 关键技术细节

#### 1. 为什么 `backward()` 返回元组？

`backward()` 必须返回与 `forward()` 输入数量相同的梯度元组：

```python
# forward 有 3 个输入：solver, vertices, derivative_type
@staticmethod
def forward(ctx, solver, vertices, derivative_type):
    ...

# backward 必须返回 3 个梯度
@staticmethod
def backward(ctx, grad_output):
    return None, grad_tensor, None
    #     ↑      ↑           ↑
    #   solver  vertices  derivative_type
```

#### 2. 为什么需要 `ctx`？

`ctx` 用于在 `forward()` 和 `backward()` 之间传递数据：

```python
def forward(ctx, ...):
    # 保存到 ctx，供 backward 使用
    ctx.solver = solver
    ctx.derivative_type = derivative_type

def backward(ctx, grad_output):
    # 从 ctx 读取
    solver = ctx.solver
    derivative_type = ctx.derivative_type
```

#### 3. 为什么需要 `set_cache_level(CacheLevel.Derivatives)`？

计算导数需要中间结果（如刚度矩阵、残差等），必须启用缓存：

```python
solver.set_cache_level(pf.CacheLevel.Derivatives)
```

如果不设置，`backward()` 时会出错。

#### 4. 为什么需要 `solve_adjoint()`？

伴随方法的核心是求解伴随方程：

```python
# backward 中必须调用
ctx.solver.solve_adjoint(grad_output)
```

这会求解伴随问题，为后续计算导数做准备。

### 性能考虑

1. **内存**：只缓存必要的中间结果，比自动微分节省很多
2. **计算**：只需要一次伴随求解，无论有多少参数
3. **并行**：可以并行运行多个独立的仿真

### 限制和未来改进

**当前限制**：
- 必须使用 `nanobind` backend，需要编译 C++ 模块。详细原因参见 [Differentiable 与 Backend 的关系](#differentiable-与-backend-的关系)
- 一次只能计算一种导数类型
- 多个 solver 的复用可以优化

**未来改进**：
- 支持同时计算多种导数
- 优化多个 solver 的复用
- 支持更多导数类型
- 更好的错误处理和诊断

---

## 总结

### 核心价值

- ✅ **易用性**：5 行代码搞定，自动处理所有细节
- ✅ **统一性**：与 `solve()` API 完全一致
- ✅ **可靠性**：不会忘记关键步骤，减少错误
- ✅ **灵活性**：支持从简单到复杂的所有场景
- ✅ **高效性**：使用伴随方法，性能优异

### 使用建议

1. **大多数用户**：直接使用 `solve_differentiable()`，5 行搞定
2. **高级用户**：可以继承 `PolyFEMFunction` 自定义
3. **性能敏感**：当前实现已经优化，性能与手动实现相同

### 设计哲学

1. **分层设计**：每层职责清晰，易于维护
2. **自动处理**：减少用户负担，避免常见错误
3. **向后兼容**：不破坏现有代码，旧代码仍可用
4. **灵活扩展**：支持多种使用方式，易于扩展

---

## 参考资料

- [API 架构文档](api-architecture.md) - 整体架构设计
- [配置指南](config-guide.md) - 配置参数说明
- [示例代码](../polyfempy/differentiable/examples/) - 可运行的示例

