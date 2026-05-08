# Differentiable API Contract

这个文件记录 Phase 4 的 differentiable / optimization API contract。目标是让
reviewer 和未来用户能快速判断：

- 什么时候用 one-shot differentiable solve；
- 什么时候用 prepared optimization problem；
- shape / material / parameterized shape 三条路径有什么区别；
- `run_optimization(..., return_result=True)` 返回什么；
- 哪些名字是推荐 public API，哪些只是 advanced / compatibility。

## 推荐 Public API

新用户应该先看这一组入口：

```python
from polyfempy.differentiable import (
    make_optimizer,
    make_von_mises_loss,
    prepare_differentiable_simulation,
    prepare_optimization_problem,
    prepare_parameterized_shape_problem,
    run_optimization,
)
```

如果只需要导出训练样本或读取 body-specific gradient，可以再用：

```python
from polyfempy.differentiable import (
    save_training_sample,
    shape_gradient_for_body,
)
```

`polyfempy.differentiable.__all__` 只包含推荐 public API。旧 helper 和诊断函数仍然
可以显式 import，但它们被归到 `COMPATIBILITY_API`，并通过 lazy import 加载；
它们不再作为新用户第一入口，也不会让顶层 facade 一次性暴露所有高级名字。

## 三种核心 Workflow

### 1. One-Shot Differentiable Solve

适合：只想跑一次 forward + backward，拿一个 gradient。

```python
result = prepare_differentiable_simulation(
    cfg=cfg,
    derivative_type="shape",
)
loss = make_von_mises_loss(result=result, body=1, time="smooth_max")
loss.backward()
```

这个 workflow 不一定需要 optimizer。常见用途：

- debug gradient；
- 导出一个 training sample；
- 检查 loss 是否可微；
- 比较不同 objective / body selection。

### 2. Prepared Optimization Problem

适合：要反复 solve/backward/optimizer.step。

推荐统一入口：

```python
problem = prepare_optimization_problem(cfg=cfg, kind="shape")
```

或 material：

```python
E_lattice = torch.nn.Parameter(torch.tensor(20.0))

problem = prepare_optimization_problem(
    cfg=cfg,
    kind="material",
    body_id=1,
    E_parameter=E_lattice,
    parameter_name="E_lattice_MPa",
    bounds=(1.0, None),
    E_unit="MPa",
)
```

然后：

```python
optimizer = make_optimizer(problem, name="adam", lr=1e-2)
loss_fn = make_von_mises_loss(body=1, time="smooth_max")

run = run_optimization(
    problem,
    steps=5,
    optimizer=optimizer,
    loss_fn=loss_fn,
    return_result=True,
)
```

### 3. Parameterized Shape Optimization

适合：设计变量不是所有 vertices，而是少量物理/几何参数，例如 `h` 和
`theta_deg`。

```python
h = torch.nn.Parameter(torch.tensor(0.04))
theta_deg = torch.nn.Parameter(torch.tensor(90.0))

problem = prepare_parameterized_shape_problem(
    cfg=cfg,
    parameters=[h, theta_deg],
    parameter_names=["h", "theta_deg"],
    bounds={"h": (0.03, 0.07), "theta_deg": (60.0, 110.0)},
    vertex_map=h_theta_vertex_map,
)
```

`vertex_map` 是用户提供的 PyTorch map：

```text
named parameters -> vertices
```

PolyFEM 仍然对 vertices 求 shape derivative；PyTorch 负责把 `dL/dX` chain 回
`dL/dh`、`dL/dtheta_deg`。

## `OptimizationRunResult` Contract

`run_optimization(...)` 默认保留旧行为，返回 step list：

```python
steps = run_optimization(...)
```

新代码推荐：

```python
run = run_optimization(..., return_result=True)
```

稳定字段：

| 字段 | 语义 |
| --- | --- |
| `run.steps` | completed step objects。 |
| `run.iterations` | 已完成 step 数量。 |
| `run.final_step` | 最后一个 step；没有 step 时为 `None`。 |
| `run.final_loss` | 最后 loss 的 Python float。 |
| `run.best_step` | loss 最小的 step。 |
| `run.best_loss` | 最小 loss 的 Python float。 |
| `run.best_iteration` | best step 的 iteration。 |
| `run.final_gradient` | final step 上保存的 gradient，如果该 step type 支持。 |
| `run.success` | 是否完成请求的 step 数。 |
| `run.message` | 人读状态，例如 `completed 5 optimization steps`。 |
| `run.summary()` | JSON-friendly summary dict。 |

推荐写法：

```python
summary = run.summary()
print(summary["final_loss"])
print(summary["best_loss"])
```

`summary()` 至少包含：

```text
problem_type
success
message
optimization_steps
final_iteration
final_loss
best_iteration
best_loss
workspace
summary_path
history_summary_path
gradient_dir
```

如果 step 类型提供更多信息，summary 还会包含：

```text
final_step_norm
final_max_vertex_update
final_gradient_path
final_E_value
final_E_unit
```

## Backward Compatibility

Phase 4 不改变旧默认行为：

```python
steps = run_optimization(...)
```

仍然返回 list。这样旧 experiment scripts 不会因为返回类型改变而坏。

Compatibility helpers 仍然显式可 import，例如：

```python
from polyfempy.differentiable import solve_differentiable
```

但它们不在 `polyfempy.differentiable.__all__` 里，并且只在显式请求时 lazy-load。
新 docs 和 examples 不应该把它们写成推荐入口。

## Examples 对应关系

| Example | Contract |
| --- | --- |
| `examples/03_shape_gradient.py` | one-shot shape gradient。 |
| `examples/04_scalar_E_gradient.py` | one-shot scalar material gradient。 |
| `examples/05_parameterized_vertex_map.py` | parameterized shape + `OptimizationRunResult`。 |
| `examples/06_dataset_one_case.py` | one differentiable run -> training sample。 |

不要为了展示 API 而把所有 examples 都改成 optimizer loop。gradient example 和
dataset export example 本来就应该保持直接。

## 不应该破坏的语义

后续 cleanup 必须保护：

- `run_optimization(...)` 默认返回 list；
- `run_optimization(..., return_result=True)` 返回 `OptimizationRunResult`；
- `OptimizationRunResult.summary()` 是 JSON-friendly dict；
- `best_loss` 从 completed steps 中按最小 loss 选择；
- parameterized shape step snapshots 使用用户设计变量名字，例如 `h` /
  `theta_deg`，而不是退回 `param_0`；
- compatibility names 显式 import 仍然可用，但不进入 `__all__`。

## 推荐测试

改 differentiable public surface 后：

```bash
python -m pytest tests/test_import_public_api.py
```

改 optimization result contract 后：

```bash
python -m pytest \
  tests/test_optimization_run_result.py \
  tests/test_parameterized_shape_problem.py
```

改 examples 后，如果 backend 和 PyTorch 可用，至少跑 touched example：

```bash
python examples/05_parameterized_vertex_map.py
```
