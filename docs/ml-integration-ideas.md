# PolyFEM Python API 与机器学习结合：方向与建议

本文档面向「希望把 PolyFEM Python API 与 ML 结合、并考虑投稿 NeurIPS 等会议」的师生，总结现有能力、可做的研究方向和 API 增强建议。**先写中文，便于讨论和后续扩展。**

---

## 一、现状：已有 ML 相关能力

### 1.1 多后端数组（NumPy / PyTorch / JAX）

- **`polyfempy.api`** 的 `solve()` 和 `Result` 支持 NumPy、PyTorch、JAX 数组：
  - 输入 `vertices`/`cells` 可以是 `torch.Tensor` 或 `jax.Array`，内部会转成 NumPy 再调 C++，结果可再转回原后端（`result.to_backend()`）。
  - 见 `api/tensor.py`：`detect_backend()`、`as_numpy()`、`to_backend()`，零拷贝尽量保留。
- **含义**：和 PyTorch/JAX 的「数据管线」已经兼容（例如用 `torch.Tensor` 做网格、用 DataLoader 喂数据），但**普通 `solve()` 本身不可微**，不能直接对网格或参数做 `loss.backward()`。

### 1.2 可微仿真模块（Differentiable）

- **`polyfempy.differentiable`** 提供基于**伴随法**的可微求解，与 PyTorch 深度集成：
  - **`solve_differentiable(V, C, cfg, ...)`**：输入可为 `torch.Tensor`（如 `requires_grad=True` 的网格），输出 `DifferentiableResult`，其 `.u` 等为可反传的 Tensor。
  - **梯度类型**：`derivative_type` 支持 `"shape"`（形状）、`"periodic_shape"`、`"material"`（材料参数）、`"initial_velocity"` 等，对应 C++ 的 `shape_derivative`、`elastic_material_derivative`、`initial_velocity_derivative`。
  - **实现**：通过 `torch.autograd.Function`（`PolyFEMFunction`）自定义 backward，内部调用 C++ 的 `solve_adjoint` + 对应 derivative 接口，避免在 Python 里展开整个求解器。
- **含义**：已经具备「FEM 仿真 + PyTorch 优化循环」的基础，适合做**梯度驱动的优化**（形状、材料、初值等）。

### 1.3 小结

| 能力           | 状态 | 说明 |
|----------------|------|------|
| PyTorch 张量入/出 | ✅   | `solve()` / `Result` 支持，differentiable 全链路 Tensor |
| 可微求解（伴随法） | ✅   | `solve_differentiable()`，形状/材料/初速导数 |
| 与优化器结合     | ✅   | `create_shape_optimizer`、`gradient_check` 等辅助 |
| JAX 支持        | 部分 | 数组可进可出，但**无可微 JAX 封装**（无 JAX 自定义 VJP） |
| 批量化 / GPU   | 部分 | 单次求解；若需 batch 需自己在 Python 层循环或并行 |

---

## 二、与 ML 结合的研究方向（适合 NeurIPS 的思路）

下面几类方向都可以在「PolyFEM 作为精确/可微物理引擎」的前提下做，并自然用到当前/扩展后的 API。

### 2.1 可微分物理与逆设计（Differentiable Physics & Inverse Design）

- **思路**：用可微 FEM 做**逆问题**或**设计优化**（形状、材料、载荷等），目标函数可以是力学量（应力、位移、刚度）、制造约束、观测拟合等。
- **与 API 的关系**：直接依赖 `solve_differentiable()` + `torch.optim`，或在此基础上封装「设计变量 → 网格/参数 → 可微仿真 → loss」的 pipeline。
- **NeurIPS 相关性**：可微分物理、physics-based optimization、inverse problems 一直是顶会方向；可强调**大规模 3D/接触/非线性**下的可微 FEM 与优化框架。

**可做的点**：
- 形状优化（拓扑/边界） + 接触/摩擦的稳定梯度与算法。
- 材料分布或参数识别（从位移/应力场反推 E、ν 等）。
- 与神经网络结合：设计变量由 NN 输出，NN 用「可微仿真 loss」端到端训练。

### 2.2 神经算子与数据驱动代理（Neural Operators & Surrogates）

- **思路**：用 PolyFEM 生成**大量解场数据**（不同几何、材料、边界），训练**神经算子**（如 FNO、DeepONet、U-Net 等）近似「参数 → 解」的映射，用于快速推理或不确定性量化。
- **与 API 的关系**：`solve()` 作为**数据生成器**；可封装为 PyTorch `Dataset`（参数、网格 → 调用 `solve()` → 返回场/标量）。未来若支持 batch 或 GPU 加速，可进一步提速数据生成。
- **NeurIPS 相关性**：neural operators、surrogate models、scientific ML 很常见；可强调**真实 FEM 数据 + 复杂几何/接触**而非简单 PDE。

**可做的点**：
- 几何/参数 → 位移场或应力场的神经算子；与 PolyFEM 的误差与泛化分析。
- 多保真度：少量高精度 PolyFEM 解 + 大量粗网格或简化模型，做多保真学习。
- 不确定性量化：用代理模型做 Monte Carlo 或贝叶斯推断。

### 2.3 物理信息神经网络（PINNs）与 FEM 结合

- **思路**：用 FEM 解作为**强约束或弱约束**：例如在 PINNs 的 loss 里加入「在部分采样点/边界上满足 FEM 解」的项，或用 FEM 残差作为物理 loss；也可用 PolyFEM 解做 curriculum / 预训练。
- **与 API 的关系**：`solve()` 提供**参考解或残差样本**；API 只需稳定输出场与网格（如 `Result.point_data`、`Result.vertices`），便于与 PyTorch 张量对齐。
- **NeurIPS 相关性**：PINNs、physics-informed learning、hybrid methods 持续有投稿；可强调「FEM + NN」的混合架构与理论/实验。

**可做的点**：
- 在关键区域用 FEM 约束 PINNs，减少违反物理的区域。
- 用 PolyFEM 生成「难例」（大变形、接触）训练更鲁棒的 PINN。
- 时间步进：用 PolyFEM 做若干步，NN 预测下一步或修正。

### 2.4 强化学习 / 控制（RL & Control）

- **思路**：把仿真环境建在 PolyFEM 上（例如柔体、接触、多体），智能体通过动作改变边界、力或几何，用 RL 学控制策略或设计策略。
- **与 API 的关系**：`solve()` 作为 **step 函数**：给定当前状态（网格、边界、载荷）→ 得到下一状态或观测。若需策略可微，可局部用 `solve_differentiable()` 或用可微近似。
- **NeurIPS 相关性**：RL + 物理仿真、sim-to-real、deformable objects 都有空间；可强调**高保真 FEM 环境**与可复现性。

**可做的点**：
- 柔体/接触的连续控制；与简化模型的对比。
- 用可微仿真做 model-based RL 或 world model 的梯度更新。

### 2.5 其他方向（简要）

- **降阶模型（ROM）**：用 PolyFEM 解做 POD/ROM 基，再配 NN 学系数或动力学，适合高维参数空间。
- **多保真 / 迁移**：不同网格密度、不同物理（线弹性 vs 接触）之间迁移或联合训练。
- **JAX 生态**：若在 API 层提供 JAX 可微封装（自定义 `jax.custom_vjp`），可与 JAX 的优化、PMAP、JAX-based 神经算子无缝结合，扩大受众。

---

## 三、API 增强建议（便于与 ML 结合）

在保持当前设计的前提下，下列增强能更好支撑上述方向，并方便写论文与开源。

### 3.1 统一入口与文档

- **统一入口**：在 `polyfempy` 或 `polyfempy.api` 的文档/示例中明确写出「普通仿真用 `solve()`，要梯度用 `solve_differentiable()`」，并给一个最小可运行示例（例如 2D 梁或立方体 + 形状梯度）。
- **可选**：在 `polyfempy.api` 的 `__all__` 或子模块里提供 `solve_differentiable` 的再导出（如 `from polyfempy.differentiable import solve_differentiable`），便于用户只记一个包名。

### 3.2 批量化与 DataLoader 友好

- **批量化**：当前 `solve()` / `solve_differentiable()` 为单次调用。可提供：
  - **Batch 示例**：在文档或 `examples/` 里给出「多组 (V, C, cfg) 循环调用 + 合并为 batch Tensor」的推荐写法，或封装一个 `batch_solve()` 返回 list 或 stacked Tensor。
  - **Dataset 示例**：一个 `torch.utils.data.Dataset` 示例：索引 → 参数/网格 → `solve()` → 返回 (input_tensor, output_tensor)，方便做神经算子/代理模型数据管线。
- **不要求立刻支持「单次调用多网格」**：先通过 Python 层批循环 + 可选多进程/多线程即可满足大部分 ML 数据生成需求。

### 3.3 可微接口与 `backend` 表述

- **与架构一致**：文档中已明确「无 backend 切换，只有 C++ 扩展」。`solve_differentiable()` 里的 `backend="nanobind"` 可保留为兼容，但文档说明「仅支持 C++ 扩展，无需选 backend」。
- **导数类型文档**：在 differentiable 文档或 docstring 中列出所有 `derivative_type` 及对应物理含义（形状、材料、初速等），方便写方法部分和实验。

### 3.4 结果格式与 PyTorch/JAX 对齐

- **Result 字段**：保证 `Result` / `DifferentiableResult` 的场（如 `u`、应力、应变）能稳定以 NumPy/PyTorch 形式取出，且 shape 一致（例如 (N, dim)），便于与 NN 输入输出对齐。
- **可选**：提供 `result.to_torch()` 或 `result.to_jax()` 的便捷方法（若已有 `to_backend` 可在此强调），减少用户手写转换。

### 3.5 JAX 可微（中长期）

- 若希望吸引 JAX 用户和 JAX 论文复现：可为 `solve_differentiable` 的「最小核心」（单次前向 + 形状梯度）提供 **JAX 封装**（`jax.custom_vjp`），输入输出为 JAX 数组，这样可与 JAX 的自动微分、`jax.grad`、`jax.vmap` 结合。可与 PyTorch 版并存，API 如 `solve_differentiable_jax(...)`。

### 3.6 可复现性与引用

- **随机种子**：配置中 `random_seed` 等已在文档中说明；在 ML 示例中显式设置并注明「便于复现」。
- **版本与引用**：在 README 或 docs 中注明「若在研究中使用了 PolyFEM Python API，请引用 …」，便于 NeurIPS 等论文的 reproducibility 和引用规范。

---

## 四、一个最小「ML 友好」示例思路（用于论文/附录）

下面是一个可直接用于「方法说明」或「附录实现细节」的最小流程（伪代码级），便于和老师/审稿人沟通：

```python
# 1) 数据生成：用 PolyFEM 生成神经算子训练数据
from polyfempy.api import solve, SimulationConfig
import torch
dataset = []
for i in range(N_samples):
    V, C, cfg = sample_geometry_and_config(i)  # 例如随机化几何/材料
    result = solve(V, C, cfg)
    u = torch.from_numpy(result.point_data["u"])  # 或 result.to_backend()
    dataset.append((input_params, u))
# 然后 torch.utils.data.TensorDataset / DataLoader 训练 FNO 等

# 2) 可微优化：形状/参数优化
from polyfempy.differentiable import solve_differentiable
V = torch.tensor(V, requires_grad=True)
result = solve_differentiable(V, C, cfg, derivative_type="shape")
loss = criterion(result.u, target)
loss.backward()
optimizer.step()  # 更新 V 或其它参数
```

- **数据生成**：强调「真实 FEM」；**可微优化**：强调「伴随法梯度 + PyTorch」，无需有限差分。

---

## 五、当前状态（已完成）

以下已经就绪，可直接使用：

1. **`solve_differentiable()` 与 `solve()` 对齐**  
   - `cfg` 支持三种形式：**JSON 路径（str）**、**dict**、**SimulationConfig**（与主 API 一致）。  
   - 不传 `V/C` 时：从 config 的 `geometry` 用 `load_mesh_from_settings()` 加载网格；需提供 `root_path`（传 JSON 路径时自动取配置所在目录；用类/dict 时设 `cfg.extras["_root_path"]` 或参数 `root_path=...`）。  
   - 传 `V/C` 时：走 `set_mesh(V, C)`，与原有可微流程一致。

2. **两种输入方式**  
   - **方式一**：`solve_differentiable(cfg="path/to/config.json", derivative_type="shape")`。  
   - **方式二**：用 `SimulationConfig`、`Geometry`、`GeometryMesh` 等 API 类构建 `cfg`，再 `solve_differentiable(cfg=cfg, root_path=str(data_dir), ...)`。  
   - 详见 `examples/differentiable_minimal.py`，脚本会分别跑两种方式并打印前向/反向结果，说明两种写法都可用。

3. **环境与验证**  
   - 需已安装 PyTorch，并已编译/加载 C++ 扩展（nanobind，且支持 differentiable）。  
   - 运行 `python examples/differentiable_minimal.py` 可验证前向+反向及 `result.vertices.grad`。  
   - C++ 端 log 已设为 off（`set_log_level(6)`），控制台输出已简化。

---

## 六、下一步建议（按优先级）

在「可微求解已跑通、API 与 solve() 对齐」的基础上，可按下面顺序推进：

| 优先级 | 事项 | 说明 |
|--------|------|------|
| 1 | **跑通最小示例** | 若尚未在本地跑通，先完成 C++ 扩展编译并运行 `examples/differentiable_minimal.py`。 |
| 2 | **选一个 ML 方向做小实验** | 从第二节选一个方向（如 2.1 形状优化、2.2 神经算子数据生成），用当前 API 做一个小规模实验（单问题/小网格），验证流程并熟悉 `result.u`、`result.vertices.grad` 的用法。 |
| 3 | **批量化 / DataLoader 示例** | 在 `examples/` 或文档中加一个「多组 (cfg 或 V,C,cfg) 循环 + 合并为 batch / Dataset」的示例，便于神经算子或代理模型的数据管线。 |
| 4 | **导数类型与配置文档** | 在 differentiable 文档或 docstring 中列出所有 `derivative_type`（shape、material、initial_velocity 等）及对应物理含义，方便写方法描述和实验。 |
| 5 | **论文/投稿细化** | 确定具体 track（如 ML4PS、NeurIPS 主会应用等），再细化「问题设定 + 方法 + 实验设计」；可把本文档中的「方向综述」与「API 改进清单」拆成两个短文档便于分工。 |

更细的待办清单见 [下一步清单](next-steps.md)。

---

## 七、总结

- **已有基础**：多后端数组（含 PyTorch/JAX 入出）、可微求解（伴随法、多种导数）、与 PyTorch 优化循环兼容；`solve_differentiable()` 已与 `solve()` 对齐（cfg 支持路径/ dict/SimulationConfig，两种输入方式均有示例）。
- **当前状态**：最小示例 `examples/differentiable_minimal.py` 已覆盖「JSON 路径」与「API 类构建 cfg」两种用法；C++ 输出已降噪。下一步见第六节与 [next-steps.md](next-steps.md)。
- **可加强点**：批量化与 DataLoader 示例、可微接口与导数类型文档、结果与 PyTorch/JAX 的稳定对齐；中长期可考虑 JAX 可微封装与引用规范。
- **研究落点**：在「高保真/可微 FEM + ML」交叉处选一个具体问题（如接触形状优化、神经算子误差分析、PINN-FEM 混合），结合当前 API 做实现与实验，即可形成完整故事。

如需，我可以根据你们具体想投的 track（例如 ML4PS、NeurIPS Datasets and Benchmarks、或主会应用 track）再细化一版「问题设定 + 方法 + 实验设计」的提纲，或把本文档拆成「方向综述」与「API 改进清单」两个短文档便于分工。
