# PolyFEM Python API：已做改进与 NeurIPS 方向路线图

本文档面向投稿 NeurIPS 等会议，总结**已完成的 API 改进**（方案 A、去掉 common.json）、**当前可进一步改进的点**，以及**与 ML 结合的未来方向**。便于写方法/实验章节和补充材料。

---

## 一、已完成的改进

### 1.1 方案 A：在绑定的时候把 result 设置好

**老师建议**：被计算出的 result 要在 Result class 里显示，且这件事要**在绑定的时候就设置好**——即从 C++算完到 Python 拿到的、能直接喂给 Result 的那份数据，在 C++ 绑定层就组装好；Python 只负责用这份数据构造 Result，不在 Python 里试多个 getter 再拼。

**我们做的**：


| 层级         | 实现                                                                                                                           |
| ---------- | ---------------------------------------------------------------------------------------------------------------------------- |
| **C++ 绑定** | `Solver::solve()` 在算完后，在绑定层组装一个 **结果包**（dict）：`vertices`、`cells`、`u`、`p`、`meta`（含 `from_bundle: true`），并 **return 该 dict**。  |
| **Python** | 若 `solver.solve()` 返回带 `_result_bundle` 的 dict，则**仅从该 dict** 取数构造 `Result`，**不再**调用 `_extract_additional_fields` 去试探 getter。 |


**带来的好处**：

- 契约清晰：解长什么样、对应 Result 的哪些字段，全部在绑定里约定。
- Python 逻辑简单：不再「试 get_solution / get_vertices / get_displacement / …」。
- 易扩展：以后 C++ 多一个场（如 stress、strain），只在绑定里多填一项，Python 按同一套键读取即可。

**演示**：运行 `python examples/python_config_5_cubes.py` 打印 Result from C++ bundle；运行 `python examples/differentiable_single_step.py` 演示可微单步与形状梯度。

**代码位置（给老师看）**：C++ `src/state/state.cpp` 约 435–451 行（`solve()` 内组装 bundle，return dict）；Python `polyfempy/api/solve.py` 约 351–374 行（仅从 `ret` 取数构造 Result，不调 `_extract_additional_fields`）。

**Before vs After（更能体现改动的好处）**：

| 旧流程（Before） | 现在（After，方案 A） |
|---|---|
| `solver.solve()` 返回 tuple/空；Python 侧再去**试探多个 getter**（`get_solution/get_vertices/get_elements/...`）拼 `fields/V/cells/meta` | C++ 绑定层 `solve()` **直接 return bundle dict**（`vertices/cells/u/p/meta/_result_bundle`） |
| `Result` 的“数学输出格式”主要靠 Python 端试探 + 拼装隐式形成 | `Result` 的输入契约在绑定层写死：Python 只做 **按键取数 → 构造 Result** |
| 接口不稳：C++ 改个 getter 名字，Python 可能拿不到 u/vertices | 接口稳：bundle key 是唯一契约；扩展新场（stress/strain/energy）只需在绑定层加 key |

---

### 1.2 去掉对 common.json 的依赖（API 路径）

**原状**：部分 C++ 或 JSON 工作流依赖外部 `common.json` 引用与合并，配置分散、路径依赖多。

**我们做的**：

- 在 **Python API 的配置处理** 中：若传入的配置（dict 或 SimulationConfig 转成的 dict）里含有 `"common"` 键，则**丢弃该键**，不再加载或合并任何外部 common 文件。
- 即：**API 路径下，solver 只使用「当前传入的这一份配置」**，不依赖外部 common.json，配置自包含、可复现。

**代码位置**：`polyfempy/api/solve.py` 中 `_process_json_config()`：`processed.pop("common", None)`。

**好处**：用 API 类（SimulationConfig、Geometry、材料、边界等）或单份 dict 即可完整描述问题，适合脚本化、数据生成和 ML 管线，也便于在论文中写「我们使用自包含配置，无外部 common 依赖」。

---

### 1.3 Shape contract 与 Result 统一接口（.u / to_torch）

**已落实**：

- **Shape contract**：见 [shape-contract.md](shape-contract.md)。核心字段写死：`vertices (n_vertices, dim)`、`cells (n_cells, k)`、`u (n_vertices, dim)`、stress/strain per-vertex `(n_vertices, 6)`、energy scalar。
- **Result 常用属性**：`Result` 提供 **`.u`**、**`.p`**、**`.stress`**、**`.strain`**（与 `point_data` 一致），与 **DifferentiableResult** 用法统一，都用 **`result.u`** 取位移。
- **to_torch()**：普通 `solve()` 返回的 Result 默认 NumPy；调用 **`result.to_torch(include_mesh=True)`** 可原地转为 PyTorch，之后 **`result.u`**、**`result.vertices`** 为 `torch.Tensor`。**DifferentiableResult** 无需 to_torch（本来就是 torch 且可 backward）。
- **示例**：`examples/python_config_5_cubes.py` 演示 `result.u` 与 `to_torch()`；`examples/differentiable_single_step.py` 使用 `result.u`（已是 torch，无需 to_torch）。

---

## 二、目前还可改进的点（让 API 更好）

在保持现有设计的前提下，下列改进能更好支撑研究与投稿：


| 优先级 | 改进项                         | 说明                                                                                                                                                  |
| --- | --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | **统一入口与文档**                 | 在 README / 文档中明确：普通仿真用 `solve()`，要梯度用 `solve_differentiable()`；给出最小可运行示例（如 2D 梁 + 形状梯度）。可选：在 `polyfempy.api` 再导出 `solve_differentiable`，方便用户只记一个包名。 |
| 2   | **批量化与 DataLoader 友好**      | 提供示例：多组 (cfg 或 V,C,cfg) 循环调用 `solve()` / `solve_differentiable()`，结果合并为 batch 或封装为 `torch.utils.data.Dataset`，便于神经算子/代理模型的数据生成。                     |
| 3   | **Result 与 PyTorch/JAX 对齐** | 见 [2.1](#21-shape-contract输出格式约定)–[2.3](#23-result-与-differentiableresult-用法统一)：shape contract、统一 Torch 用法、Result/DifferentiableResult 一致。 |
| 4   | **导数类型与配置文档**               | 见 [2.4](#24-导数类型与配置文档) |
| 5   | **可选：C++ bundle 扩展**        | 见 [2.5](#25-可选c-bundle-扩展) |
| 6   | **可复现性与引用**                 | 在配置/示例中显式设置随机种子；在 README 或 docs 中注明「若在研究中使用了 PolyFEM Python API，请引用 …」，满足 NeurIPS 等对 reproducibility 和引用的要求。                                        |


### 2.1 Shape contract（输出格式约定）

**目标**：把「API 的数学输出格式」写死，让 FEM 输出长得像现代 ML 张量接口。

**约定文档**：[shape-contract.md](shape-contract.md)。核心字段写死为：

| 字段 | Shape / 约定 |
|------|----------------|
| vertices | `(n_vertices, dim)` |
| cells | `(n_cells, k)` |
| u | `(n_vertices, dim)` |
| stress | **per-vertex** `(n_vertices, 6)` Voigt |
| strain | **per-vertex** `(n_vertices, 6)` Voigt |
| energy | **scalar**（总能量） |

实现与 C++ bundle 扩展时按此约定输出，NN 输入/输出即可稳定对应。

---

### 2.2 统一的 Torch 使用方式

**推荐用法**（文档明确推荐）：

```python
r = solve(...)
r.to_torch(include_mesh=True)   # 原地转为 PyTorch；或 to_backend(include_mesh=True) 若已传 torch 入 solve
u = r.u   # (n_vertices, dim)
```

- **`to_backend(include_mesh=True)`**：当 `Result.backend` 已是 `"torch"`（例如 `solve()` 时传入的是 `torch.Tensor`）时，把场和网格转为 Torch。
- **`to_torch(include_mesh=True)`**：不论当前 backend 是否为 numpy，**原地**转为 PyTorch 并返回 `self`，之后可直接用 **`r.u`**、**`r.vertices`** 等。适合「用 numpy 跑求解、事后统一转 Torch」的 ML 管线。

两种方式二选一即可；转完后 **`r.u`** 与 **`r.point_data["u"]`** 一致，shape 均为 `(n_vertices, dim)`。

---

### 2.3 Result 与 DifferentiableResult 用法统一

**目标**：普通 Result 和 DifferentiableResult 的用法尽量接近，避免「一个用 `point_data["u"]`、一个用 `.u`」的不统一感。

**当前**：

- **Result** 已提供常用属性访问：**`.u`**、**`.p`**、**`.stress`**、**`.strain`**（与 DifferentiableResult 对齐）。同时保留 **`point_data`** / **`cell_data`** 访问所有场。
- **DifferentiableResult** 继续用 **`.u`**、**`.vertices`**、可选的 **`.stress`**、**`.strain`**。

**统一约定**：

- 两者都推荐用 **`result.u`** 取位移；若需压力则 **`result.p`**。
- Result 的 **`.u`** / **`.p`** 等价于 **`result.point_data["u"]`** / **`result.point_data.get("p")`**，只是更短、与 DifferentiableResult 一致。
- 文档中明确：**Result** = 普通求解结果，支持 `.u` / `.p` / `point_data` / `cell_data`；**DifferentiableResult** = 可微求解结果，`.u` / `.vertices` 带梯度，另有 `release_solver()`、`to_numpy()`。

这样「让 FEM 输出长得更像现代 ML 张量接口」：统一用 **`.u`** 拿位移，shape 固定为 `(n_vertices, dim)`。

**直白理解**：这一块本质上是在做——**让 FEM 输出长得更像现代 ML 张量接口**：写死 shape contract、统一 Torch 入口（`to_torch()` / `r.u`）、Result 与 DifferentiableResult 都用 `.u` 取位移，便于 NN 与优化循环直接消费。

---

### 2.4 导数类型与配置文档

**目标**：在 differentiable 文档或 docstring 中**列出所有** `derivative_type` 及**物理含义**，便于写方法描述和实验设置（NeurIPS 方法/附录常用）。

**当前支持**（`solve_differentiable(..., derivative_type=...)`，对应 C++ 绑定在 `src/differentiable/adjoint.cpp` 与 backward 在 `polyfempy/differentiable/torch_integration.py`）：

| `derivative_type`   | C++ 接口 | 物理含义 | 梯度形式 / 用途 |
|---------------------|----------|----------|------------------|
| `"shape"`           | `shape_derivative(solver)` | 形状导数 | 标量 loss 对**顶点位置**的梯度 ∂L/∂V；用于形状优化、逆设计（顶点移动）。 |
| `"periodic_shape"`  | `periodic_shape_derivative(solver)`（若绑定存在） | 周期边界下的形状导数 | 与 `"shape"` 类似，针对周期边界问题；当前若未绑定则回退到 `shape_derivative` 并打 warning。 |
| `"material"`       | `elastic_material_derivative(solver)` | 材料参数导数 | 标量 loss 对**弹性材料参数**（如 E、ν）的梯度；用于材料识别、材料分布优化。 |
| `"initial_velocity"` | `initial_velocity_derivative(solver)` | 初始速度导数 | 标量 loss 对**初始速度场**的梯度；返回 dict，Python 侧展平为 1D 与 `vertices` 形状匹配；用于动力学/初值优化。 |

**建议落地**：

1. **Docstring**：在 `solve_differentiable()` 的 `derivative_type` 参数说明中，直接贴上上表或链接到文档，例如：`"shape" | "periodic_shape" | "material" | "initial_velocity"`，并各用一句话说明物理意义与梯度对象（顶点 / 材料参数 / 初速）。
2. **文档**：在 `docs/differentiable-guide.md`（及英文版）中开一小节「导数类型（derivative_type）」，列出上表，并注明：backward 时根据 `derivative_type` 调用对应 C++ 接口，梯度写回 `result.vertices.grad`（shape）或对应可微参数的 grad。
3. **实验设置**：论文中若做形状优化，写「我们使用 `derivative_type=\"shape\"`，通过伴随法计算 ∂L/∂V」；若做材料或初值优化，可写 `\"material\"` / `\"initial_velocity\"` 及对应物理量。这样审稿人可复现。

---

### 2.5 可选：C++ bundle 扩展

**目标**：若 PolyFEM 的 `State` 在 C++ 中暴露 `get_stress` / `get_strain` / `get_energy`（或等价接口），在 **C++ 的 `solve()` 里**把这类量一并填入结果包；Python 侧**仅从 Result 读**，不再在 Python 里补任何 getter。

**当前状态**：

- **C++**（`src/state/state.cpp`，约 435–451 行）：`solve()` 在绑定层组装 `bundle`，目前包含：`vertices`、`cells`、`u`、`p`、`_result_bundle`、`meta`。文件中已有注释：「当 polyfem::State 提供 get_stress/get_strain/get_energy 等时，在此处填入 bundle」。
- **Python**（`polyfempy/api/solve.py`，约 351–374 行）：当 `ret` 为结果包时，已对 `stress`、`strain`、`v`、`energy` 做**可选读取**：若 `ret.get(key)` 存在且非空，则写入 `fields`，再由 `Result` 的 `_split_fields()` 按行数落入 `point_data` 或 `cell_data`。

**扩展步骤**（当 C++ State 提供相应 API 时）：

1. **C++**：在 `state.cpp` 的 `solve()` 中，在现有 `bundle["u"]`、`bundle["p"]` 之后：
   - 若存在 `s.get_stress(sol, stress)`（或按实际 API 签名），将 `stress` 填入 `bundle["stress"]`；
   - 同理 `get_strain` → `bundle["strain"]`；
   - 若存在标量或场形式的能量（如 `get_energy(sol)`），填入 `bundle["energy"]`。
2. **Python**：无需改逻辑；`solve.py` 已从 `ret` 读 `stress`、`strain`、`energy` 并写入 `fields`，用户通过 `result.point_data["stress"]` 或 `result.cell_data["stress"]`（取决于行数是 n_vertices 还是 n_cells）即可使用。
3. **Shape 约定**：按 [shape-contract.md](shape-contract.md)，应力/应变为 **per-vertex** `(n_vertices, 6)` Voigt；便于与 NN 或后处理对齐。

**好处**：所有「解相关」数据都在绑定层统一输出，方案 A 的契约保持单一；后续若增加更多场（如损伤、塑性应变），只需在 C++ bundle 中增加键值，Python 端自动可用。

---

## 三、未来可做的 ML 相关方向（NeurIPS 相关）

在「PolyFEM 作为高保真/可微物理引擎」的前提下，下列方向均可基于当前/扩展后的 API 做，并自然用到方案 A 与自包含配置带来的清晰契约。

### 3.1 可微分物理与逆设计（Differentiable Physics & Inverse Design）

- **思路**：用可微 FEM 做**逆问题**或**设计优化**（形状、材料、载荷等），目标可为力学量（应力、位移、刚度）、制造约束或观测拟合。
- **与 API**：直接使用 `solve_differentiable()` + `torch.optim`，或封装「设计变量 → 网格/参数 → 可微仿真 → loss」的 pipeline。
- **NeurIPS 相关性**：可微分物理、physics-based optimization、inverse problems 常见于顶会；可强调** 3D/接触/非线性**下的可微 FEM 与优化框架。

**可做点**：形状优化（含接触/摩擦）的稳定梯度与算法；材料或参数识别；设计变量由 NN 输出、用可微仿真 loss 端到端训练。

### 3.2 神经算子与数据驱动代理（Neural Operators & Surrogates）

- **思路**：用 PolyFEM 生成**大量解场数据**（不同几何、材料、边界），训练**神经算子**（如 FNO、DeepONet）近似「参数 → 解」的映射，用于快速推理或不确定性量化。
- **与 API**：`solve()` 作为数据生成器；封装为 PyTorch `Dataset`（参数/网格 → `solve()` → 返回场/标量）。批量化示例见第二节改进项。
- **NeurIPS 相关性**：neural operators、surrogate models、scientific ML；可强调**真实 FEM 数据 + 复杂几何/接触**。

**可做点**：几何/参数 → 位移或应力场的神经算子及与 PolyFEM 的误差与泛化分析；多保真度学习；不确定性量化。

### 3.3 物理信息神经网络（PINNs）与 FEM 结合

- **思路**：用 FEM 解作为强/弱约束：在 PINNs 的 loss 中加入「在部分点/边界上满足 FEM 解」的项，或用 FEM 残差作为物理 loss；也可用 PolyFEM 解做 curriculum 或预训练。
- **与 API**：`solve()` 提供参考解或残差样本；`Result.point_data`、`Result.vertices` 等稳定输出，便于与 PyTorch 张量对齐。
- **NeurIPS 相关性**：PINNs、physics-informed learning、hybrid methods；可强调「FEM + NN」的混合架构。

**可做点**：在关键区域用 FEM 约束 PINNs；用 PolyFEM 生成大变形/接触难例以训练更鲁棒的 PINN；时间步进中 FEM 与 NN 交替或修正。

### 3.4 强化学习 / 控制（RL & Control）

- **思路**：把仿真环境建在 PolyFEM 上（柔体、接触、多体），智能体通过动作改变边界、力或几何，用 RL 学控制或设计策略。
- **与 API**：`solve()` 作为 step 函数；若需策略可微，可局部用 `solve_differentiable()` 或可微近似。
- **NeurIPS 相关性**：RL + 物理仿真、sim-to-real、deformable objects；可强调高保真 FEM 环境与可复现性。

**可做点**：柔体/接触的连续控制；用可微仿真做 model-based RL 或 world model 的梯度更新。

### 3.5 最小「ML 友好」流程（可用于论文附录）

```python
# 数据生成：PolyFEM 作为数据源
from polyfempy.api import solve, SimulationConfig
for i in range(N):
    cfg = build_config(i)  # 几何/材料/边界
    result = solve(vertices=None, cells=None, cfg=cfg)  # 结果来自绑定结果包
    u = result.point_data["u"]  # 或 result.to_backend("torch")
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

---

## 四、总结


| 类别        | 内容                                                                                   |
| --------- | ------------------------------------------------------------------------------------ |
| **已做**    | 方案 A（result 在绑定设置好）；去掉 common.json 依赖；可微与 solve() 对齐；**Shape contract**（[shape-contract.md](shape-contract.md)）；**Result.u / .p / to_torch()** 与 DifferentiableResult 用法统一；示例已更新（python_config_5_cubes 含 to_torch 演示）。 |
| **可改进**   | 统一入口与文档、批量化/DataLoader 示例、导数类型文档、C++ bundle 扩展（stress/strain/energy）、可复现与引用规范。             |
| **ML 方向** | 可微物理与逆设计、神经算子与代理、PINN-FEM 混合、RL/控制；均可基于当前 API 与方案 A 的清晰契约做实现与实验。                     |


详细 ML 方向与 API 建议见 [ml-integration-ideas.md](ml-integration-ideas.md)。

---

## 五、建议下一步与相关文档

**建议下一步**（按优先级）：(1) 跑通 `examples/differentiable_single_step.py`；(2) 选一个 ML 方向做小实验；(3) 批量化/DataLoader 示例；(4) 导数类型文档；(5) 论文/投稿细化。若报错见 [how_to_view_cpp_backend.md](how_to_view_cpp_backend.md)。

**相关文档**：

| 文档 | 内容 |
|------|------|
| [shape-contract.md](shape-contract.md) | 输出格式约定（vertices、u、stress、strain、to_torch） |
| [differentiable-guide.md](differentiable-guide.md) | 可微功能指南 |
| [api-architecture.md](api-architecture.md) | API 整体架构与 Route A |
| [config-guide.md](config-guide.md) | 配置与 SimulationConfig |
| [how_to_view_cpp_backend.md](how_to_view_cpp_backend.md) | C++ 扩展查看/调试 |
| [ml-integration-ideas.md](ml-integration-ideas.md) | ML 方向与 API 增强建议 |