# 哪里触发了“真正的计算”（C++ / PolyFEM）

本仓库的核心结论：

- **真正的数值计算都在 C++ 扩展模块 `polyfempy` 里发生**（即 PolyFEM 的 `polyfem::State` / `AdjointTools` 等）。
- Python 层（`polyfempy/api/*`、`polyfempy/differentiable/*`）主要负责 **数据规整（NumPy/torch/jax → NumPy）、配置序列化（dict → JSON string）、结果包装（Result/DifferentiableResult）**。
- 由于 C++ 绑定实现为 **`Solver.solve() -> (sol, pressure)`**，Python 侧必须 **接收并解析返回值**，而不是假设存在 getter。

---

## 快速判定规则

### ✅ 一定触发真实 C++ 计算

- 任意路径里出现：
  - `import polyfempy as pf`
  - `solver = pf.Solver()`（或 `pf.State()`）
  - `solver.solve()`（或 `solve_problem`/`run` 的绑定）

### ❌ 不做真实计算（Python glue）

- 只涉及：
  - `tensor.py`（转换/contiguous/astype）
  - `config.py`（构建 dict/JSON，校验）
  - `result.py`（保存/导出/简单后处理）
  - `batch.py`（循环调用）

---

## 路径 A（统一路径）：`polyfempy.api.solve.solve(...)`

### Python 入口（glue）

- 文件：`polyfempy/api/solve.py`
- 函数：`solve(vertices, cells, cfg, ...)`

Python 侧做的事：

- 将输入 `vertices/cells` 归一化成 **CPU、NumPy、C-contiguous**，并强制 `cells=int32`
- 将 `cfg` 归一化成 **JSON 字符串**（因为 C++ 侧 `json::parse(str(settings))`）
- 调用 C++ `Solver` 方法完成：
  - `set_settings(JSON)`
  - `load_mesh_from_settings()`（JSON 模式）或 `set_mesh(V, C)`（数组模式）
  - `solve()`
- **解析 `solve()` 返回的 `(sol, pressure)`** 并包装为 `Result`

### C++ 真实计算入口

- 文件：`src/state/state.cpp`
- 绑定方法：`Solver.solve(...)`
- 核心计算发生在：
  - `s.build_basis()`
  - `s.assemble_rhs()` / `s.assemble_mass_mat()`
  - `s.solve_problem(sol, pressure)`

也就是说，**只要你最终调用到了 `pf.Solver().solve()`，就是 PolyFEM 真算**。

---

## 可微分路径：`polyfempy.differentiable.solve.solve_differentiable(...)`

### Python 入口（glue + PyTorch 包装）

- 文件：`polyfempy/differentiable/solve.py`
- 函数：`solve_differentiable(V, C, cfg, ...)`

流程要点：

- `pf.Solver()` + `solver.set_settings(JSON)` + `solver.set_mesh(...)`
- `solver.build_basis()` + `solver.assemble()`（预处理）
- 调用 `PolyFEMFunction.apply(...)` 进入 PyTorch 自定义算子

### forward：真实 C++ 求解 + 缓存

- 文件：`polyfempy/differentiable/torch_integration.py`
- 方法：`PolyFEMFunction.forward(...)`

触发真实计算的调用：

- `solver.set_cache_level(pf.CacheLevel.Derivatives)`（启用可微分缓存）
- **`ret = solver.solve()`**（真实求解）
- 优先解析 `ret[0]` 作为 solution（兼容返回 tuple）

### backward：真实 C++ 伴随求解 + 导数

- 文件：`polyfempy/differentiable/torch_integration.py`
- 方法：`PolyFEMFunction.backward(...)`

触发真实计算的调用：

- `ctx.solver.solve_adjoint(grad_output_np)`（C++ 解伴随方程）
- `pf.shape_derivative(ctx.solver)` / `pf.elastic_material_derivative(...)` / `pf.initial_velocity_derivative(...)`

对应 C++ 绑定位置：

- `src/state/state.cpp`：`Solver.solve_adjoint(...)`, `Solver.set_cache_level(...)`, `Solver.get_solution_cache(...)`
- `src/differentiable/adjoint.cpp`：`shape_derivative`, `elastic_material_derivative`, ...

---

## 其它 C++ 侧“真实计算但不是主 solve()”的例子

这些也是“真算”，只是用途不同：

- `pf.apply_slim(...)`（网格优化工具）→ `src/differentiable/utils.cpp`
- `pf.create_objective(...)` / `Objective.derivative(...)` → `src/differentiable/objective.cpp`
- `Solver.step_in_time(...)` / `init_timestepping(...)`（瞬态）→ `src/state/state.cpp`

---

## 一句话总结

- **真正计算**：`pf.Solver().solve()`、`solve_adjoint(...)`、`pf.*_derivative(...)`
- **Python 主要做 glue**：转换数据、生成 JSON、解析 `(sol, pressure)`、包装结果

