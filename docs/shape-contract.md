# PolyFEM Python API：输出格式约定（Shape contract）

本文档定义 **Result / DifferentiableResult 的数学输出格式**，即「你的 API 的数学输出长什么样」。所有与解、网格、场相关的 shape 和存放位置都按此约定，便于与 ML 张量接口、神经算子、优化循环对齐。

---

## 核心字段（写死）

| 字段 | Shape / 存放 | 说明 |
|------|----------------|------|
| **vertices** | `(n_vertices, dim)` | 网格顶点坐标。`dim` = 2 或 3。 |
| **cells** | `(n_cells, k)` | 单元拓扑。`k` = 每单元节点数（如 3=三角形，4=四面体）。多块时按 `[(cell_type, array), ...]`，每块 `array.shape == (n_cells_i, k)`。 |
| **u** | `(n_vertices, dim)` | 位移场，与顶点一一对应。 |
| **p** | `(n_vertices,)` 或 压力自由度形状 | 压力（若有）。与求解器自由度一致。 |
| **stress** | **per-vertex** `(n_vertices, 6)` | 应力（若由 C++ bundle 提供）。Voigt 顺序：2D 为 (σxx, σyy, σxy)，3D 为 (σxx, σyy, σzz, σyz, σxz, σxy)。与顶点对应，便于与 NN 输入输出对齐。 |
| **strain** | **per-vertex** `(n_vertices, 6)` | 应变（若由 C++ bundle 提供）。Voigt 顺序同 stress。per-vertex。 |
| **energy** | **scalar**（或可选 field） | 总能量为标量；若将来提供场则单独约定（如 per-cell 或 per-vertex）。 |

**原则**：位移、应力、应变等与「顶点」对齐时，均为 `(n_vertices, ...)`，方便统一当作「每个顶点一个向量/张量」，与现代 ML 张量接口一致。

---

## 在代码中的位置

- **Result**：`vertices` 为属性；`u`、`p`、`stress`、`strain` 可通过 **`result.u`**、**`result.p`** 或 **`result.point_data["u"]`** 等访问；`cells` 为 **`result.cells`**（list of (type, array)）。
- **DifferentiableResult**：**`result.u`**、**`result.vertices`**（及可选的 `stress`、`strain`）为属性，与 Result 的常用属性一致。
- 未提供的场（如 C++ 尚未填入 stress/strain）为 `None` 或不存在于 `point_data`/`cell_data`。

---

## PyTorch 用法

- **普通 Result**（`solve()` 返回）：默认 NumPy。若需 PyTorch，调用 **`result.to_torch(include_mesh=True)`**，之后 **`result.u`**、**`result.vertices`** 为 `torch.Tensor`，可直接喂给 NN 或 loss（不可微）。
- **DifferentiableResult**（`solve_differentiable()` 返回）：**`result.u`**、**`result.vertices`** 已是 `torch.Tensor` 且可 `backward()`，**不需要** `to_torch()`。
- 详见 [neurips-api-and-ml-roadmap.md](neurips-api-and-ml-roadmap.md) 第二节「统一的 Torch 使用方式」与「Result 与 DifferentiableResult 用法统一」。

---

## 与 ML 的对应

- 神经算子输入/输出：通常为「场」在顶点或网格上，即 `(n_vertices, dim)` 或 `(n_vertices, 6)`。
- 本约定保证：**`result.u`** 与 **`result.point_data["u"]`** 的 shape 均为 `(n_vertices, dim)`；若提供 **`result.stress`**，则为 `(n_vertices, 6)`。可直接用于 `criterion(result.u, target)` 或作为 DataLoader 的输出。
