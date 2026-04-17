# 可微 Python 侧改动 — 发给「老环境 / 另一份仓库」的移植说明

不需要改 `state.cpp`、不需要重编 C++ 扩展（除非老分支连 `solve(log_level=...)` 或 `get_solutions()` 都没有，那种情况才要对齐绑定）。

下面几处均为 **纯 Python**，对齐后通常能解决：

1. 终端看不到与 `polyfem.log` 一致的 `[debug]` / Newton 行  
2. `loss.backward()` 报 `Invalid adjoint_rhs shape!`（瞬态尤其明显）

本仓库已实现：见 `solve_diff.py` 与 `torch_integration.py`。老分支可按下列清单做 diff 合并。

---

## 1. `polyfempy/differentiable/solve_diff.py`

### 1.1 `_console_log_level_from_settings(settings)`

从 `settings["output"]["log"]["level"]` 读字符串（`trace` … `off`），映射成整数 `0`–`6`，逻辑与 `polyfempy/api/solve.py` 里传给 `solver.solve(log_level=...)` 的那段一致。

### 1.2 `solve_differentiable(..., *, quiet_polyfem_setup: bool = True)`

若老代码没有该参数：加上 **keyword-only** 参数 `quiet_polyfem_setup=True`。

构造 `Solver()` 之后：仅当 `quiet_polyfem_setup` 为 **True** 时才调用原来的 `set_log_level(6)`（或 `_solver_set_log_level_off`）；为 **False** 则跳过，便于终端日志与 JSON 一致。

### 1.3 调用 `PolyFEMFunction.apply` 时多传一格 log level

在 `settings = config.to_dict()` 已有之后：

```python
solve_log_level = _console_log_level_from_settings(settings)
solutions = PolyFEMFunction.apply(solver, V_torch, derivative_type, solve_log_level)
```

（老代码若是 `apply(solver, V_torch, derivative_type)`，要改成上面这样，并改 `forward` 签名，见下一节。）

### 1.4 Config + 磁盘 mesh：不要在 `forward` 之前再 `assemble` / `init_timestepping`

在 **`use_load_mesh`** 分支（`set_settings` → `load_mesh_from_settings` → 取顶点）里，**故意不要**在进入 `PolyFEMFunction.forward` 之前再跑：

- `assemble()`  
- `set_cache_level(Derivatives)`  
- `init_timestepping(...)`  

否则会和后面 `Solver.solve()` → `solve_problem` 里自己的时间步初始化 **重复或乱序**，在大 **NeoHookean + 瞬态** 上可能把 **State** 弄不一致，甚至 **SIGSEGV**。

**约定**：`solve_differentiable` 在该分支只做 `set_settings` → `load_mesh_from_settings` → `build_basis`（以及可选的 `quiet_polyfem_setup`）；**真正的** `assemble`、导数缓存、`solve(log_level=...)` 内的时间推进，一律在 **`PolyFEMFunction.forward`** 里按唯一顺序执行（`set_vertices` → `build_basis` → `assemble` → `set_cache_level` → `solve`），与 **`api.solve`** 侧「不要提前 `init_timestepping`」的思路对齐。

**这不是**对「可微 + 接触」SIGSEGV 的根治；那一类见 `experiment/new_experiment_02/ISSUES_AND_ENVIRONMENT.md`（规避 + 归因 C++ 可微接触路径）。

---

## 2. `polyfempy/differentiable/torch_integration.py` — `PolyFEMFunction`

### 2.1 `forward` 增加参数 `solve_log_level: int = 2`

调用 C++ 时：

```python
try:
    ret = solver.solve(log_level=solve_log_level)
except TypeError:
    ret = solver.solve()  # 极老绑定没有关键字参数时回退
```

原因：绑定里 `solve` 默认 `log_level=3`（warn），会在 `solve` 里 `set_log_level`，把终端 Newton debug 压掉；而文件 log 仍可能是 debug。

### 2.2 取位移 `u` 时 **优先** `solver.get_solutions()`

在 `solver.solve(...)` 之后，组 `solutions_np` 的顺序应为：

1. 若有 `get_solutions`：`solutions_np = np.asarray(solver.get_solutions())`（可 `try/except`，失败或空则继续）  
2. 否则再用返回的 `dict` `ret["u"]` / `tuple` `ret[0]` / cache 兜底  

原因：`solve_adjoint(grad_output)` 要求 `grad_output` 形状为 `(diff_cached.u(0).size(), diff_cached.size())`，与 `get_solutions()` 的矩阵一致。仅用 bundle 里的 `u` 在瞬态上可能与 `diff_cached` 布局不一致 → `Invalid adjoint_rhs shape!`。

### 2.3 `backward` 的返回值个数

`forward` 若多了 `solve_log_level` 一个非 tensor 参数，`backward` 最后要 **多返回一个 `None`**：

```python
return None, grad_tensor, None, None
#     solver      vertices   derivative_type  solve_log_level
```

（与 `apply` 的 4 个输入一一对应：无梯度的槽位用 `None`。）

---

## 3. 应用层（可选）

若有 `run_experiment02.py` 之类调用 `solve_differentiable`：

- 需要终端日志时：`quiet_polyfem_setup=False`（或与 CLI 绑定）。  
- 默认保持 `True` 则与旧行为一致、控制台更安静。

本仓库：`run_experiment02.py` 支持 `--polyfem-console-log`，等价于 `quiet_polyfem_setup=False`。

---

## 4. 验证清单

- [ ] `solve_differentiable` 里 `PolyFEMFunction.apply` 传入 `solve_log_level`。  
- [ ] `torch_integration.forward` 使用 `solver.solve(log_level=solve_log_level)`（`TypeError` 回退 `solve()`）。  
- [ ] `get_solutions()` **优先**于 bundle 的 `u`。  
- [ ] `backward` 返回 4 个值（若 `forward` 为 4 个输入）。  
- [ ] 瞬态 case：`loss.backward()` 不再报 `Invalid adjoint_rhs shape!`。

---

## 5. 若老环境仍失败

- 确认 Python 加载的是你改过的 `polyfempy`（`pip install -e .` 或 `PYTHONPATH` 指向含 `polyfempy` 的仓库根）。  
- 若 `solver.solve(log_level=...)` **一直** `TypeError`，说明扩展太旧，要对齐 `src/state/state.cpp` 里 `solve` 的绑定签名（**需重编**）。  
- 若根本没有 `get_solutions`，同样需要更新绑定或手写与 `diff_cached` 一致的 `u` 形状（不推荐，**优先升级绑定**）。

可将本文件整体粘贴给另一个 Cursor，并附上老分支里 `solve_diff.py` / `torch_integration.py` 的当前内容，让对方按 diff 合并。
