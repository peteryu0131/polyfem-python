# Config Classes 使用指南

本文档详细介绍 `SimulationConfig` 中使用的各种配置类（Classes），包括 Material、BoundaryConditions 和 ProblemParams 等。

---

## 目录

1. [为什么使用 Class？](#为什么使用-class)
2. [可用的 Classes](#可用的-classes)
3. [便捷方法](#便捷方法)
4. [Convenience Factories](#convenience-factories)
5. [向后兼容性](#向后兼容性)
6. [完整示例](#完整示例)

---

## 为什么使用 Class？

使用 class 而不是 dict 的主要原因是**提供更好的 IDE 自动补全支持**。

### 问题：Dict + Typing 的局限性

```python
# ❌ 使用 dict - IDE 无法自动补全
cfg = SimulationConfig(materials={"E": 2100, "nu": 0.3})
cfg.materials["E"] = 2100  # IDE 不知道 "E" 是有效的键
cfg.materials["wrong_key"] = 100  # IDE 也不会报错
cfg.materials[""]  # 输入引号后，IDE 不会提示有哪些键
```

**问题**：
- IDE 无法知道 `materials` 字典里有哪些有效的键
- 用户输入 `cfg.materials["` 时，IDE 不会提示 `"E"` 或 `"nu"`
- 拼写错误不会被 IDE 发现
- 不同 problem_type 的参数容易混淆

### 解决方案：使用 Class

```python
# ✅ 使用 class - IDE 自动补全
from polyfempy.api.config import Material, GravityParams

material = Material(E=2100, nu=0.3)
cfg = SimulationConfig(materials=material)
material.E  # IDE 会提示这是有效的属性
material.nu  # IDE 会提示这是有效的属性

params = GravityParams(force=0.1)
cfg = SimulationConfig(problem_type="Gravity", problem_params=params)
params.force  # IDE 会提示这是有效的属性
```

**优势**：
- ✅ IDE 可以提供完整的自动补全
- ✅ 类型检查更严格
- ✅ 代码更清晰，不容易出错
- ✅ 可以添加验证逻辑
- ✅ 防止不同 problem_type 的参数混淆

---

## 可用的 Classes

### Material Classes

PolyFEM 提供了多种材料类，每种材料类都有专门的类定义，支持强 IDE 参数提示和多输入方式。

#### 通用 Material 类

基础的 `Material` 类，适用于简单的线性弹性材料：

```python
from polyfempy.api.config import Material

material = Material(
    E=2100,              # Young's modulus
    nu=0.3,              # Poisson's ratio
    rho=1.0,             # Density
    type="LinearElasticity"  # Material type
)

# IDE 会自动补全以下属性：
# material.E
# material.nu
# material.rho
# material.type
```

**注意**：虽然 `Material` 类支持 `type` 参数来指定材料类型，但**推荐使用专门的材料类**（如 `NeoHookean`、`LinearElasticity` 等），以获得更好的类型提示和参数验证。

#### 专门的材料类

以下材料类提供了更强的类型安全性和 IDE 支持：

##### NeoHookean

NeoHookean 材料，支持两种输入方式：

```python
from polyfempy.api.config import NeoHookean

# 方式 1：E-nu 输入（Young's modulus 和 Poisson's ratio）
material = NeoHookean(
    E=2100,      # Young's modulus (必需)
    nu=0.3,      # Poisson's ratio (必需)
    id=0,        # Material ID (可选，默认 0)
    rho=1,       # Density (可选，默认 1)
    phi=0,       # First angle (可选，默认 0)
    psi=0        # Second angle (可选，默认 0)
)

# 方式 2：lambda-mu 输入（Lamé 参数）
material = NeoHookean(
    lambda_=1000,  # First Lamé parameter (必需)
    mu=800,        # Shear modulus (必需)
    id=0,
    rho=1,
    phi=0,
    psi=0
)
```

##### IsochoricNeoHookean

等容 NeoHookean 材料，同样支持 E-nu 和 lambda-mu 两种输入：

```python
from polyfempy.api.config import IsochoricNeoHookean

# E-nu 输入
material = IsochoricNeoHookean(E=2100, nu=0.3)

# lambda-mu 输入
material = IsochoricNeoHookean(lambda_=1000, mu=800)
```

##### LinearElasticity

线性弹性材料，支持两种输入方式：

```python
from polyfempy.api.config import LinearElasticity

# 方式 1：E-nu 输入
material = LinearElasticity(
    E=2100,
    nu=0.3,
    id=0,
    rho=1,
    phi=0,  # E-nu 模式支持
    psi=0   # E-nu 模式支持
)

# 方式 2：lambda-mu 输入
material = LinearElasticity(
    lambda_=1000,
    mu=800,
    id=0,
    rho=1
)
```

##### HookeLinearElasticity

Hooke 线性弹性材料，支持两种输入方式：

```python
from polyfempy.api.config import HookeLinearElasticity

# 方式 1：E-nu 输入
material = HookeLinearElasticity(
    E=2100,
    nu=0.3,
    id=0,
    rho=1,
    fiber_direction=[0, 0, 0]  # Fiber direction vector
)

# 方式 2：elasticity_tensor 输入
material = HookeLinearElasticity(
    elasticity_tensor=[...],  # Full elasticity tensor
    id=0,
    rho=1,
    fiber_direction=[0, 0, 0]
)
```

##### SaintVenant

Saint-Venant 材料，支持两种输入方式：

```python
from polyfempy.api.config import SaintVenant

# 方式 1：E-nu 输入
material = SaintVenant(
    E=2100,
    nu=0.3,
    id=0,
    rho=1,
    phi=0,
    psi=0,
    fiber_direction=[0, 0, 0]
)

# 方式 2：elasticity_tensor 输入
material = SaintVenant(
    elasticity_tensor=[...],
    id=0,
    rho=1,
    phi=0,
    psi=0,
    fiber_direction=[0, 0, 0]
)
```

##### MooneyRivlin

Mooney-Rivlin 材料：

```python
from polyfempy.api.config import MooneyRivlin

material = MooneyRivlin(
    c1=0.5,      # First Mooney-Rivlin parameter (必需)
    c2=0.1,      # Second Mooney-Rivlin parameter (必需)
    k=1000,      # Bulk modulus (必需)
    id=0,        # Material ID (可选，默认 0)
    rho=1        # Density (可选，默认 1)
)
```

##### MooneyRivlin3Param

三参数 Mooney-Rivlin 材料：

```python
from polyfempy.api.config import MooneyRivlin3Param

material = MooneyRivlin3Param(
    c1=0.5,      # First Mooney-Rivlin parameter (必需)
    c2=0.1,      # Second Mooney-Rivlin parameter (必需)
    c3=0.05,     # Third Mooney-Rivlin parameter (必需)
    d1=1000,     # First volumetric parameter (必需)
    id=0,
    rho=1
)
```

##### MooneyRivlin3ParamSymbolic

符号三参数 Mooney-Rivlin 材料：

```python
from polyfempy.api.config import MooneyRivlin3ParamSymbolic

material = MooneyRivlin3ParamSymbolic(
    c1=0.5,
    c2=0.1,
    c3=0.05,
    d1=1000,
    id=0,
    rho=1
)
```

##### UnconstrainedOgden

无约束 Ogden 材料：

```python
from polyfempy.api.config import UnconstrainedOgden

material = UnconstrainedOgden(
    alphas=2.0,           # Alpha parameters (必需)
    mus=[1.0, 0.5],       # Mu parameters list (必需)
    Ds=[0.1, 0.2],        # D parameters list (必需)
    id=0,
    rho=1
)
```

##### IncompressibleOgden

不可压缩 Ogden 材料：

```python
from polyfempy.api.config import IncompressibleOgden

material = IncompressibleOgden(
    c=1.0,        # C parameters (必需，可以是 float/string/object/list)
    m=2.0,        # M parameters (必需，可以是 float/string/object/list)
    k=1000,       # Bulk modulus (必需)
    id=0,
    rho=1
)
```

##### IncompressibleLinearElasticity

不可压缩线性弹性材料：

```python
from polyfempy.api.config import IncompressibleLinearElasticity

material = IncompressibleLinearElasticity(
    E=2100,       # Young's modulus (必需)
    nu=0.3,       # Poisson's ratio (必需)
    id=0,
    rho=1
)
```

##### Stokes

Stokes 流体材料：

```python
from polyfempy.api.config import Stokes

material = Stokes(
    viscosity=0.1,  # Viscosity (必需)
    id=0,
    rho=1
)
```

##### NavierStokes

Navier-Stokes 流体材料：

```python
from polyfempy.api.config import NavierStokes

material = NavierStokes(
    viscosity=0.1,  # Viscosity (必需)
    id=0,
    rho=1
)
```

##### OperatorSplitting

算子分裂材料：

```python
from polyfempy.api.config import OperatorSplitting

material = OperatorSplitting(
    viscosity=0.1,  # Viscosity (必需)
    id=0,
    rho=1
)
```

##### Electrostatics

静电材料：

```python
from polyfempy.api.config import Electrostatics

material = Electrostatics(
    epsilon=8.85e-12,  # Permittivity (必需)
    id=0,
    rho=1
)
```

#### 使用示例

```python
from polyfempy.api.config import (
    SimulationConfig,
    NeoHookean,
    LinearElasticity,
    Stokes
)

# 使用专门的材料类（推荐）
neo_hookean = NeoHookean(E=2100, nu=0.3)
cfg1 = SimulationConfig(materials=neo_hookean)

linear_elastic = LinearElasticity(E=2100, nu=0.3)
cfg2 = SimulationConfig(materials=linear_elastic)

stokes = Stokes(viscosity=0.1)
cfg3 = SimulationConfig(materials=stokes)

# 所有材料类都支持 IDE 自动补全
# 输入 material. 后，IDE 会提示所有可用属性
```

### BoundaryConditions

边界条件容器类，支持 IDE 自动补全。

```python
from polyfempy.api.config import BoundaryConditions

bc = BoundaryConditions()
bc.add_dirichlet(id=4, value=[0.0, 0.0])  # IDE 会提示参数
bc.add_neumann(id=2, value=[0.0, -1000.0])  # IDE 会提示参数
bc.set_rhs([1.0, 0.0])  # IDE 会提示参数

# IDE 会自动补全以下属性：
# bc.dirichlet_boundary
# bc.neumann_boundary
# bc.rhs

cfg = SimulationConfig(boundary_conditions=bc)
```

### ProblemParams Classes

不同问题类型的参数类：

#### GravityParams

```python
from polyfempy.api.config import GravityParams

params = GravityParams(force=0.1)  # IDE 会提示 'force'
cfg = SimulationConfig(problem_type="Gravity", problem_params=params)
```

#### TorsionParams

```python
from polyfempy.api.config import TorsionParams

params = TorsionParams(
    axis_coordinate=2,  # IDE 会提示所有参数
    n_turns=0.5,
    fixed_boundary=5,
    turning_boundary=6
)
cfg = SimulationConfig(problem_type="TorsionElastic", problem_params=params)
```

#### FlowParams

```python
from polyfempy.api.config import FlowParams

params = FlowParams(
    inflow=1,           # IDE 会提示所有参数
    outflow=3,
    inflow_amount=0.25,  # 注意：class 中使用正确拼写（不是 inflow_amout）
    outflow_amount=0.25,
    direction=0,
    obstacle=[7]
)
cfg = SimulationConfig(problem_type="Flow", problem_params=params)
```

#### FlowWithObstacleParams

```python
from polyfempy.api.config import FlowWithObstacleParams

params = FlowWithObstacleParams(U=1.5, time_dependent=True)
cfg = SimulationConfig(problem_type="FlowWithObstacle", problem_params=params)
```

---

## 便捷方法

`SimulationConfig` 提供了便捷方法来设置参数（都支持 IDE 自动补全）：

```python
cfg = SimulationConfig()

# 设置材料参数
cfg.set_material(E=2100, nu=0.3)  # IDE 会提示所有参数

# 设置边界条件
cfg.set_dirichlet_boundary(id=4, value=[0.0, 0.0])  # IDE 会提示
cfg.set_neumann_boundary(id=2, value=[0.0, -1000.0])  # IDE 会提示
cfg.set_rhs([1.0, 0.0])  # IDE 会提示

# 所有方法都支持链式调用
cfg.set_material(E=2100, nu=0.3) \
   .set_dirichlet_boundary(id=4, value=[0.0, 0.0]) \
   .set_neumann_boundary(id=2, value=[0.0, -1000.0])
```

---

## Convenience Factories

所有 convenience factory 方法现在自动使用 classes：

```python
# Gravity - 自动使用 GravityParams
cfg = SimulationConfig.gravity(force=0.1, E=1e6, nu=0.3)
cfg.problem_params.force  # ✅ IDE 会提示

# Torsion - 自动使用 TorsionParams
cfg = SimulationConfig.torsion(axis_coordinate=2, n_turns=0.5)
cfg.problem_params.axis_coordinate  # ✅ IDE 会提示

# Flow - 自动使用 FlowParams
cfg = SimulationConfig.flow(inflow=1, outflow=3, inflow_amount=0.25)
cfg.problem_params.inflow_amount  # ✅ 正确的拼写！
```

---

## 向后兼容性

所有现有的代码都可以继续工作：

1. **Dict 输入仍然支持**：`SimulationConfig(materials={"E": 2100})` 仍然有效
2. **自动转换**：Dict 输入会自动转换为相应的 class（如果需要）
3. **to_dict() 方法**：所有 class 都有 `to_dict()` 方法，用于后端兼容


---

## 完整示例

### 示例 1：使用基础 Material 类

```python
from polyfempy.api.config import (
    SimulationConfig,
    Material,
    BoundaryConditions,
    GravityParams
)

# 方式 1：使用 class（推荐）
material = Material(E=2100, nu=0.3)
bc = BoundaryConditions()
bc.add_dirichlet(id=4, value=[0.0, 0.0])
gravity_params = GravityParams(force=0.1)

cfg = SimulationConfig(
    pde="LinearElasticity",
    discr_order=1,
    materials=material,
    boundary_conditions=bc,
    problem_type="Gravity",
    problem_params=gravity_params
)
```

### 示例 2：使用专门的材料类（推荐）

```python
from polyfempy.api.config import (
    SimulationConfig,
    NeoHookean,
    LinearElasticity,
    Stokes,
    BoundaryConditions
)

# NeoHookean 材料（E-nu 输入）
neo_hookean = NeoHookean(E=2100, nu=0.3, rho=1.0)
cfg1 = SimulationConfig(materials=neo_hookean)

# NeoHookean 材料（lambda-mu 输入）
neo_hookean_lame = NeoHookean(lambda_=1000, mu=800)
cfg2 = SimulationConfig(materials=neo_hookean_lame)

# LinearElasticity 材料
linear_elastic = LinearElasticity(E=2100, nu=0.3)
cfg3 = SimulationConfig(materials=linear_elastic)

# Stokes 流体材料
stokes = Stokes(viscosity=0.1, rho=1.0)
cfg4 = SimulationConfig(materials=stokes)

# 所有材料类都支持 IDE 自动补全
# 输入 material. 后，IDE 会提示所有可用属性
```

### 示例 3：使用便捷方法

```python
cfg = SimulationConfig()
cfg.set_material(E=2100, nu=0.3) \
   .set_dirichlet_boundary(id=4, value=[0.0, 0.0]) \
   .set_rhs([1.0, 0.0])
```

### 示例 4：使用 convenience factories

```python
cfg = SimulationConfig.gravity(force=0.1, E=1e6, nu=0.3)
```

### 示例 5：向后兼容（仍然支持）

```python
cfg = SimulationConfig(
    materials={"E": 2100, "nu": 0.3},  # ⚠️ IDE 无法提示
    boundary_conditions={"dirichlet_boundary": [...]},  # ⚠️ IDE 无法提示
    problem_params={"force": 0.1}  # ⚠️ IDE 无法提示
)
```

### 示例 6：复杂材料配置

```python
from polyfempy.api.config import (
    SimulationConfig,
    MooneyRivlin,
    UnconstrainedOgden,
    HookeLinearElasticity
)

# MooneyRivlin 材料
mooney_rivlin = MooneyRivlin(c1=0.5, c2=0.1, k=1000, rho=1.0)
cfg1 = SimulationConfig(materials=mooney_rivlin)

# UnconstrainedOgden 材料
ogden = UnconstrainedOgden(
    alphas=2.0,
    mus=[1.0, 0.5],
    Ds=[0.1, 0.2],
    rho=1.0
)
cfg2 = SimulationConfig(materials=ogden)

# HookeLinearElasticity 材料（elasticity_tensor 输入）
hooke = HookeLinearElasticity(
    elasticity_tensor=[...],  # 完整的弹性张量
    fiber_direction=[1, 0, 0]
)
cfg3 = SimulationConfig(materials=hooke)
```

---

## 相关文档

- [配置指南](config-guide.md) - SimulationConfig 的完整使用指南
- [API 架构文档](api-architecture.md) - 完整的 API 架构说明

