# PolyFEM Python 库介绍（面向导师汇报）

本文档用中文详细说明 **polyfempy**（PolyFEM 的 Python 绑定）是什么、目前能做什么、有什么优点，以及与机器学习（ML）相关的方面，便于向导师汇报和后续研究规划。**仅依赖本文件即可理解全貌，不引用其他文件。**

---

## 一、这个库是什么

### 1.1 基本定位

- **polyfempy** 是 PolyFEM 的 **Python 绑定**，在 Python 中通过 `import polyfempy as pf` 使用。
- **核心计算在 C++**：有限元求解器、刚度/质量矩阵组装、线性/非线性求解等都在 C++ 中完成；Python 层负责配置、调用和结果封装。
- **绑定技术**：使用 **nanobind** 编译出 C++ 扩展模块，Python 与 C++ 之间可零拷贝传递 NumPy 数组，兼顾性能与易用性。
- **版本**：当前为 **alpha** 阶段（如 0.8），API 可能仍有调整，但主入口（`solve`、`solve_differentiable`）已相对稳定。

### 1.2 与 PolyFEM 的关系

- PolyFEM 本身是一个 **C++ 有限元库**，支持多种 PDE（线性/非线性弹性、流体、接触等）、多种单元类型与求解器。
- **polyfempy** 不重写求解器，而是把 PolyFEM 的 Solver、网格、配置等通过 nanobind 暴露给 Python，使研究者可以用 Python 做：
  - 单次或批量仿真；
  - 可微分仿真（伴随法求梯度）；
  - 与 PyTorch/JAX 等框架结合做优化或机器学习。

### 1.3 技术栈概览

| 层级       | 技术 / 内容 |
|------------|--------------|
| 用户入口   | `polyfempy.api`：solve、SimulationConfig、Result、Selection、batch_solve、以及材料/边界/几何/求解器/输出等配置类 |
| 可微仿真   | `polyfempy.differentiable`：solve_differentiable、伴随法、与 PyTorch 的 autograd 集成 |
| 绑定层     | nanobind → C++ 扩展（polyfempy） |
| 核心求解   | PolyFEM C++（Eigen、各类线性/非线性求解器、接触与摩擦等） |

### 1.4 主要 API 入口一览

- **普通仿真**：`from polyfempy.api import solve` → `solve(vertices, cells, cfg)` 或 `solve(vertices=None, cells=None, cfg=cfg)`。
- **可微仿真**：`from polyfempy.differentiable import solve_differentiable` → `solve_differentiable(vertices, C, cfg, derivative_type=...)` 或仅传 `cfg`。
- **配置**：`SimulationConfig`、材料类（如 `LinearElasticity`、`NeoHookean`）、`BoundaryConditions`、`Geometry`/`GeometryMesh`、`Contact`、`Solver`/`LinearSolver`/`NonlinearSolver`、`Time`、`Output`/`ParaviewOutput` 等，均从 `polyfempy.api` 导入。
- **结果**：`Result`（含 `.u`、`.p`、`.vertices`、`.cells`、`point_data`/`cell_data`、`to_torch()` 等）；可微时返回 `DifferentiableResult`（`.u`、`.vertices` 为可反传的 Tensor）。
- **批量**：`batch_solve(jobs)`，其中每个 job 为 `(V, C, cfg)` 或 `(V, C, cfg, kwargs)`。
- **几何选择**：`Selection`，用于通过几何形状（球、盒、平面）指定边界或体，而不依赖网格文件中的 sideset ID。

---

## 二、目前可以做什么

### 2.1 主 API：普通仿真

- **统一入口**：`solve(vertices, cells, cfg)` 或 `solve(vertices=None, cells=None, cfg=cfg)`（后者从配置中的 geometry 加载网格）。
- **输入**：
  - 网格：`vertices`（顶点坐标，shape 一般为 `(n_vertices, dim)`，dim=2 或 3）、`cells`（单元连接，如三角形为 3 列、四面体为 4 列），支持 **NumPy / PyTorch / JAX** 数组，内部会统一为 C-contiguous NumPy 再交给 C++；
  - 配置：`cfg` 可以是 **SimulationConfig** 对象、**dict** 或 **JSON 文件路径**。
- **输出**：**Result** 对象，包含：
  - `result.vertices`、`result.cells`（网格）；
  - `result.u`（位移场，与顶点一一对应，shape `(n_vertices, dim)`）；
  - `result.p`（压力，若适用）；
  - `result.point_data` / `result.cell_data`（其它场，如应力、应变等）；
  - 若 C++ 端填入：`result.stress`、`result.strain`（per-vertex，Voigt 格式）。
- **结果来源**：采用「方案 A」——C++ 在 `solve()` 内直接组装好「结果包」（vertices、cells、u、p、meta 等），Python 只从该包构造 Result，不依赖多个 getter 拼装，契约清晰、易扩展。

### 2.2 支持的物理与配置能力

- **PDE 类型**（通过 `SimulationConfig` 或 JSON）：
  - 线性弹性（LinearElasticity）、非线性弹性（NonLinearElasticity）；
  - 多种本构：NeoHookean、Mooney-Rivlin、Saint-Venant、Stokes、Navier-Stokes、Electrostatics、IncompressibleLinearElasticity 等；
  - Poisson 等标量问题。
- **预定义问题**：重力（Gravity）、Franke、扭转（Torsion）、流动（Flow、DrivenCavity、FlowWithObstacle）等，便于教学与复现；可通过 `SimulationConfig.gravity(...)`、`SimulationConfig.franke(...)` 等工厂方法快速构建。
- **几何与接触**：
  - 多体几何：多个网格、平移/缩放/旋转变换、体积/表面选择（volume_selection、surface_selection）；
  - 接触（Contact）：开启后可做多体接触、摩擦；参数如 dhat、摩擦系数等可配置。
- **时间步进**：瞬态问题通过 `Time` 配置（tend、time_steps、积分器等）。
- **求解器**：线性求解器（如 Pardiso、Cholmod、SimplicialLDLT 等）、非线性（Newton、线搜索如 RobustArmijo）均可配置；还可设置高级选项（如 lump_mass_matrix）。
- **输出**：Paraview（.pvd）、JSON、以及通过 Result 的 `to_vtk(path)` 做后处理或可视化。

### 2.3 配置方式

- **方式一**：在代码中用 **API 类** 构建配置。例如：
  - `SimulationConfig(pde="LinearElasticity", discr_order=1, materials=[...], boundary_conditions={...}, geometry=...)`；
  - 材料可用类：`NeoHookean(E=..., nu=..., rho=...)`、`LinearElasticity(E=..., nu=...)` 等；
  - 边界：`BoundaryConditions()` 配合 `dirichlet_boundary` / `neumann_boundary` 的列表（每项含 selection id 与 value）；
  - 几何：`Geometry(meshes=[GeometryMesh(mesh=路径, surface_selection=[...], transformation={...})])`。
- **方式二**：使用 **完整 PolyFEM JSON 文件**（含 geometry、materials、boundary_conditions、solver、output 等），通过 `SimulationConfig.from_json_file(path)` 或直接 `solve(cfg=path)`。
- **自包含**：在 API 路径下，**不再依赖外部 common.json**；若配置里带 `"common"` 键会被丢弃，solver 只用当前传入的这一份配置，便于复现和 ML 数据生成。

### 2.4 批量与多后端

- **batch_solve(jobs)**：对多组 `(V, C, cfg)` 或 `(V, C, cfg, kwargs)` 顺序求解，返回与输入顺序一致的 Result 列表，适合批量数据生成。
- **多后端数组**：输入/输出支持 NumPy、PyTorch、JAX。内部在进 C++ 前转为 NumPy（零拷贝尽量保留）。**`solve()` 返回的 Result 也支持 `to_torch(include_mesh=True)`**，转后 `result.u`、`result.vertices` 等为 `torch.Tensor`，便于和 ML 管线、DataLoader 对接；`solve_differentiable()` 返回的 DifferentiableResult 的 `.u`、`.vertices` 已是 Tensor 且可 backward，无需再 to_torch。

### 2.5 可微分仿真（Differentiable）

- **入口**：`solve_differentiable(vertices, C, cfg, derivative_type=...)` 或仅传 `cfg`（从 geometry 加载网格，此时需在配置中提供 `_root_path` 或参数 `root_path` 以便解析网格路径）。
- **与 solve() 一致**：`cfg` 支持 JSON 路径、dict、SimulationConfig；不传 V/C 时从配置的 geometry 用内部 `load_mesh_from_settings()` 加载网格。
- **导数类型**（`derivative_type`）：
  - **shape**：标量 loss 对**顶点位置**的梯度 ∂L/∂V，用于形状优化、逆设计；
  - **periodic_shape**：周期边界下的形状导数；
  - **material**：对**弹性材料参数**（如 E、ν）的梯度，用于材料识别、材料优化；
  - **initial_velocity**：对**初始速度场**的梯度，用于动力学/初值优化。
- **实现方式**：通过 **伴随法（Adjoint Method）** 在 C++ 中求解伴随方程并计算导数，再通过 `torch.autograd.Function` 与 PyTorch 集成。
  - **为什么不用自动微分**：PDE 求解器自由度可达百万级，且多为迭代求解；若对整条求解链做自动微分，需要存储所有中间状态，内存与时间成本通常是前向的数倍，不可行。伴随法只需一次前向 + 一次伴随线性求解，梯度与设计变量维数无关，适合高维优化。
  - **伴随法简述**：前向解出状态 u 满足 F(u, θ)=0；定义伴随变量 λ 满足 F_u^T λ = -∂J/∂u；则 dJ/dθ = ∂J/∂θ + λ^T F_θ。库内在 forward 时设置 CacheLevel.Derivatives 缓存刚度矩阵等，backward 时调用 C++ 的 solve_adjoint 与对应的 derivative 接口（如 shape_derivative）得到梯度并写回 `result.vertices.grad` 等。
- **返回**：**DifferentiableResult**，其 `.u`、`.vertices` 等为 `torch.Tensor` 且可 `backward()`；梯度在 `result.vertices.grad`（形状）或对应可微参数的 grad 上。用完后可调用 `result.release_solver()` 释放 C++ solver 引用。

### 2.6 输出格式约定（Shape contract）

为与 ML 和优化循环对齐，Result / DifferentiableResult 遵循统一的 **Shape contract**：

| 字段       | Shape / 约定 |
|------------|----------------|
| vertices   | `(n_vertices, dim)`，dim=2 或 3 |
| cells      | `(n_cells, k)`，k 为每单元节点数（如 3=三角形，4=四面体） |
| u          | `(n_vertices, dim)`，与顶点一一对应 |
| p          | 压力，与求解器自由度一致 |
| stress     | per-vertex `(n_vertices, 6)` Voigt 顺序（2D 时为 σxx, σyy, σxy；3D 时为 σxx, σyy, σzz, σyz, σxz, σxy） |
| strain     | per-vertex `(n_vertices, 6)` Voigt 顺序同上 |
| energy     | 标量（若 C++ 端提供） |

这样神经算子、损失函数等可直接使用 `result.u`、`result.vertices` 等，无需再猜 shape。位移、应力、应变与「顶点」对齐，均为 (n_vertices, ...)，便于当作「每个顶点一个向量/张量」与 ML 接口一致。

### 2.7 典型使用场景简述

- **普通仿真**：构建或加载 cfg，传入网格（或仅 cfg 由配置加载网格），调用 `solve(...)`，从 Result 取 `result.u`、`result.vertices` 等；若需 PyTorch 可 `result.to_torch(include_mesh=True)`。
- **可微仿真**：构建 cfg，将顶点设为 `torch.tensor(..., requires_grad=True)`（或仅传 cfg 由配置加载网格），调用 `solve_differentiable(..., derivative_type="shape")`，定义 loss（如 `loss = (result.u**2).sum()`），`loss.backward()`，则 `result.vertices.grad` 即为形状梯度，可与 `torch.optim` 结合做形状优化。
- **批量数据生成**：多组 (V, C, cfg) 组成 jobs，调用 `batch_solve(jobs)` 得到 Result 列表，将 `result.u` 等收集为 dataset，供神经算子或代理模型训练。

---

## 三、优点总结

### 3.1 设计与工程

- **单一入口**：普通仿真用 `solve()`，要梯度用 `solve_differentiable()`，概念清晰。
- **配置自包含**：API 路径不依赖外部 common.json，适合脚本化、数据生成和复现。
- **结果契约清晰**：C++ 在绑定层组装好结果包，Python 只按固定键取数构造 Result，易扩展（例如将来 C++ 增加 stress/strain/energy 只需在 bundle 里多填）。
- **多后端**：NumPy/Torch/JAX 入出统一处理，与现有 ML 栈兼容；Result 提供 `.u`、`.p`、`to_torch()` 等，与 DifferentiableResult 用法一致（都用 `result.u` 取位移）。

### 3.2 性能与可扩展性

- **核心在 C++**：大规模组装与求解在 PolyFEM 中完成，Python 只做薄封装。
- **零拷贝**：NumPy 与 C++ 之间通过 nanobind 零拷贝；Torch 与 NumPy 在 CPU 上也可零拷贝，减少数据搬运。
- **伴随法**：梯度只需一次伴随求解，与设计变量维数无关，适合高维形状/参数优化。

### 3.3 研究与投稿

- **可复现**：自包含配置、固定 shape contract、建议在配置中设置随机种子等，便于写方法描述和附录。
- **易写论文**：已完成的改进（方案 A、去掉 common 依赖、shape contract、Result.u / to_torch、可微与 solve 对齐）以及可改进点、ML 方向，均可直接对应到方法章节和实验设置（如 derivative_type、可微 vs 普通 solve），便于与导师讨论和投稿规划。

---

## 四、ML 相关方面（详细）

### 4.1 已具备的 ML 相关能力

| 能力                 | 状态 | 说明 |
|----------------------|------|------|
| PyTorch 张量入/出    | ✅   | solve() / Result 支持；differentiable 全链路 Tensor |
| 可微求解（伴随法）   | ✅   | solve_differentiable()，shape / material / initial_velocity 等 |
| 与优化器结合         | ✅   | 可与 torch.optim 直接配合；另有 create_shape_optimizer、gradient_check 等辅助 |
| 多后端（JAX 数组）   | 部分 | 数组可进可出，但尚无 JAX 可微封装（无 custom_vjp） |
| 批量化               | 部分 | 单次求解；批量化通过 batch_solve 或 Python 层循环/并行实现 |

### 4.2 可微分物理与逆设计（Differentiable Physics & Inverse Design）

- **思路**：用可微 FEM 做逆问题或设计优化（形状、材料、载荷等），目标可为力学量（应力、位移、刚度）、制造约束或观测拟合。
- **与库的关系**：直接使用 `solve_differentiable()` + `torch.optim`，或封装「设计变量 → 网格/参数 → 可微仿真 → loss」的 pipeline。
- **可做**：形状优化（含接触/摩擦）的稳定梯度与算法；材料或参数识别；设计变量由神经网络输出、用可微仿真 loss 端到端训练。适合 NeurIPS 等顶会的可微分物理、physics-based optimization、inverse problems 方向。

### 4.3 神经算子与数据驱动代理（Neural Operators & Surrogates）

- **思路**：用 PolyFEM 生成大量解场数据（不同几何、材料、边界），训练神经算子（如 FNO、DeepONet）近似「参数 → 解」的映射，用于快速推理或不确定性量化。
- **与库的关系**：`solve()` 作为数据生成器；可封装为 PyTorch `Dataset`（参数/网格 → solve() → 返回场/标量）；Result 的 shape contract 与 `result.u`、`to_torch()` 便于 batch 与 DataLoader。
- **可做**：几何/参数 → 位移或应力场的神经算子及与 PolyFEM 的误差与泛化分析；多保真度学习；不确定性量化。对应 neural operators、surrogate models、scientific ML。

### 4.4 物理信息神经网络（PINNs）与 FEM 结合

- **思路**：用 FEM 解作为强/弱约束：在 PINNs 的 loss 中加入「在部分点/边界上满足 FEM 解」的项，或用 FEM 残差作为物理 loss；也可用 PolyFEM 解做 curriculum 或预训练。
- **与库的关系**：`solve()` 提供参考解或残差样本；`result.u`、`result.vertices`、`result.to_torch()` 等与 PyTorch 张量对齐方便。
- **可做**：在关键区域用 FEM 约束 PINNs；用 PolyFEM 生成大变形/接触难例以训练更鲁棒的 PINN；时间步进中 FEM 与 NN 交替或修正。对应 PINNs、physics-informed learning、hybrid methods。

### 4.5 强化学习 / 控制（RL & Control）

- **思路**：把仿真环境建在 PolyFEM 上（柔体、接触、多体），智能体通过动作改变边界、力或几何，用 RL 学控制或设计策略。
- **与库的关系**：`solve()` 作为 step 函数；若需策略可微，可局部使用 `solve_differentiable()` 或可微近似。
- **可做**：柔体/接触的连续控制；用可微仿真做 model-based RL 或 world model 的梯度更新。对应 RL + 物理仿真、sim-to-real、deformable objects。

### 4.6 最小「ML 友好」流程示例（可用于论文附录）

```python
# 数据生成：PolyFEM 作为数据源
from polyfempy.api import solve, SimulationConfig
for i in range(N):
    cfg = build_config(i)  # 几何/材料/边界
    result = solve(vertices=None, cells=None, cfg=cfg)
    u = result.u  # 或 result.to_torch(include_mesh=True)
    dataset.append((input_params, u))

# 可微优化：形状/参数
from polyfempy.differentiable import solve_differentiable
result = solve_differentiable(cfg=cfg, derivative_type="shape")
loss = criterion(result.u, target)
loss.backward()
# result.vertices.grad 用于更新设计变量
```

- **数据生成**：强调「真实 FEM」、自包含配置、无 common 依赖。
- **可微优化**：强调「伴随法梯度 + PyTorch」，无需有限差分。

### 4.7 导数类型与写论文时的表述

在方法或附录中可明确写出导数类型与物理含义，便于审稿人复现：

| derivative_type   | 物理含义           | 梯度对象 / 用途 |
|-------------------|--------------------|------------------|
| "shape"           | 形状导数           | ∂L/∂V，顶点位置；形状优化、逆设计 |
| "periodic_shape"  | 周期边界下的形状导数 | 同 shape，针对周期问题 |
| "material"        | 材料参数导数       | ∂L/∂(E, ν 等)；材料识别、材料分布优化 |
| "initial_velocity" | 初始速度导数     | ∂L/∂初速；动力学/初值优化 |

---

## 五、总结（汇报时可用的几句话）

- **是什么**：PolyFEM 的 Python 绑定（polyfempy），核心计算在 C++，Python 提供统一 API（solve、可微 solve_differentiable）和与 PyTorch/JAX 友好的结果格式。
- **能做什么**：线性/非线性弹性、接触、流体等多种 PDE；程序化或 JSON 配置；批量求解；**伴随法可微仿真**（形状、材料、初速等梯度）；结果与 ML 张量接口对齐（shape contract、result.u、to_torch）。
- **优点**：配置自包含、结果契约清晰、多后端支持、伴随法高效求梯度、适合做逆设计和神经算子/代理模型数据生成，便于写方法与实验章节。
- **ML 方面**：已支持可微物理与逆设计、神经算子数据生成、PINN-FEM 混合、RL/控制等方向；可与导师进一步选定一个具体问题（如接触形状优化、FNO 误差分析、PINN 约束）做实现与实验。

若导师希望看某一块的更多细节（例如伴随法推导、某类 PDE 的配置示例、或某个 ML 方向的实验设计），可以基于本文档对应小节展开说明。
