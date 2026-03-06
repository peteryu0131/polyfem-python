# 下一步清单（当前状态 + 建议顺序）

本文档是「可微 API 已与 solve() 对齐」之后的**待办与建议顺序**，便于你按条推进。详细背景与 ML 方向见 [ml-integration-ideas.md](ml-integration-ideas.md)。

---

## 项目最低要求达成情况（可汇报/交付）

| 类别 | 最低要求 | 现状 | 对应示例 |
|------|----------|------|----------|
| **主 API (solve)** | 能用配置跑通一次仿真，并有用 API 类构建配置的示例 | ✅ 已达成 | `contact_5_cubes.py`（JSON/config）；`python_config_5_cubes.py`（SimulationConfig、Geometry、材料、接触等） |
| **Differentiable** | 能跑通单次可微（前向 + loss + 反向），得到形状梯度 | ✅ 已达成 | `differentiable_single_step.py`（单次、给老师演示）；`differentiable_minimal.py`（两种 cfg 输入）；`differentiable_shape_optimization.py`（单次 + 多步尝试） |
| **API 对齐** | 可微入口与 solve() 一致（cfg 支持路径/dict/SimulationConfig） | ✅ 已达成 | solve_differentiable(cfg=...) 与 solve() 同形式，见 differentiable_minimal |
| **可演示** | 有可直接运行的例子，能说明「能算、能求出东西」 | ✅ 已达成 | 主 API：运行 contact / python_config_5_cubes；可微：运行 differentiable_single_step |

结论：**当前已满足项目最低要求**——主 API 与 Differentiable 均有可运行示例，可微单次链路稳定，可向老师演示或作为交付基线。后续为增强（批量化、文档细节、论文方向等），见下方「建议下一步」。

---

## 当前已完成

- **solve_differentiable() 与 solve() 对齐**：`cfg` 支持 str（JSON 路径）、dict、SimulationConfig；不传 V/C 时从 config 的 geometry 加载网格，支持 `root_path` / `extras["_root_path"]`。
- **两种输入方式示例**：`examples/differentiable_minimal.py` 演示「JSON 路径」与「API 类构建 cfg」，并分别打印前向+反向结果。
- **C++ 输出降噪**：内部 `set_log_level(6)`，控制台输出已简化。

---

## 建议下一步（按优先级）

### 1. 本地跑通最小示例（若尚未）

- 确保已编译并加载 C++ 扩展（nanobind，且支持 differentiable）。
- 在仓库根目录执行：`python examples/differentiable_minimal.py`。
- 若报错：C++ 未编译 → 先完成 CMake 构建；缺 PyTorch → `pip install torch`；其他见报错信息或 [how_to_view_cpp_backend.md](how_to_view_cpp_backend.md)。

### 2. 选一个 ML 方向做小实验

- 从 [ml-integration-ideas.md 第二节](ml-integration-ideas.md#二与-ml-结合的研究方向适合-neurips-的思路) 选一个方向（如形状优化、神经算子数据生成）。
- 用当前 API 做一个小规模实验：单问题、小网格，熟悉 `result.u`、`result.vertices`、`loss.backward()` 与 `result.vertices.grad` 的用法。
- 目标：验证「设计变量 → 可微仿真 → loss → 梯度」整条链路在自己环境里可行。
- **形状优化小实验**：可直接运行 `python examples/differentiable_shape_optimization.py`。脚本会做（1）单次前向 + loss + 反向并打印 `result.u`、`result.vertices`、`result.vertices.grad`；（2）若能从 solver 取到 cells，再跑 3 步梯度下降，演示多步优化。

### 3. 批量化 / DataLoader 示例（可选）

- 在 `examples/` 或文档中增加一个示例：多组 (cfg 或 V,C,cfg) 循环调用 `solve()` 或 `solve_differentiable()`，结果合并为 batch 或封装为 `torch.utils.data.Dataset`。
- 便于后续做神经算子、代理模型等需要大量解场数据的工作。

### 4. 文档与 API 细节

- 在 differentiable 文档或 docstring 中列出所有 `derivative_type`（shape、material、initial_velocity 等）及对应物理含义。
- 若需要，在 README 或 docs 中加一句「若在研究中使用了 PolyFEM Python API，请引用 …」。

### 5. 论文/投稿细化（若已定方向）

- 确定目标 track（如 ML4PS、NeurIPS 主会应用等）。
- 细化「问题设定 + 方法 + 实验设计」；可将 [ml-integration-ideas.md](ml-integration-ideas.md) 拆成「方向综述」与「API 改进清单」两个短文档便于分工。

---

## 相关文档


| 文档                                                       | 内容                        |
| -------------------------------------------------------- | ------------------------- |
| [ml-integration-ideas.md](ml-integration-ideas.md)       | ML 方向、API 增强建议、当前状态与下一步概述 |
| [differentiable-guide.md](differentiable-guide.md)       | 可微功能完整指南（含两种 cfg 输入方式）    |
| [api-architecture.md](api-architecture.md)               | API 整体架构与 Route A 说明      |
| [how_to_view_cpp_backend.md](how_to_view_cpp_backend.md) | C++ 扩展查看/调试说明             |


