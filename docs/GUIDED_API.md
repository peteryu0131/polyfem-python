# Guided API Contract

`polyfempy.api.guided` 是配置 authoring layer。它的目标是让用户不用直接手写
完整 PolyFEM JSON，也能构造一个 solver-facing `SimulationConfig`。

推荐 import：

```python
import polyfempy.api.guided as g
```

典型链路：

```text
section factories
  -> g.simulation_template(...)
  -> g.build_config(template, workspace)
  -> SimulationConfig
  -> solve(cfg=cfg)
```

## 最小例子

```python
from pathlib import Path

from polyfempy.api import solve
import polyfempy.api.guided as g

body = g.body_section(
    name="lattice",
    mesh="triangular_lattice.msh",
    material=g.material_section(
        model="NeoHookean",
        E=20.0,
        E_unit="MPa",
        nu=0.45,
    ),
    fixed_surface=g.fixed_surface_section(
        side="y_min",
        value=(0.0, 0.0),
    ),
)

template = g.simulation_template(
    problem=g.problem_section(pde="NonLinearElasticity"),
    bodies=g.bodies_section(body),
    loads=g.loads_section(rhs=(0.0, 980.0)),
    time=g.time_section(t0=0.0, tend=0.02, dt=0.01),
    solver=g.solver_section(),
    contact=g.contact_section(mode="frictionless"),
    results=g.results_section(requested_fields=["u", "von_mises"]),
)

cfg = g.build_config(template, Path("runs/example"))
result = solve(cfg=cfg)
```

## 推荐 Section Factories

`g.__all__` 只包含推荐的 section factory / builder functions。Section
dataclasses、Literal type aliases 和旧名字仍然可以显式访问，例如
`g.BodySection`、`g.MaterialModelName`、`g.experiment_template`，但它们不属于
推荐 star-import surface。

| Factory | 返回 | 用途 |
| --- | --- | --- |
| `g.problem_section(...)` | `ProblemSection` | PDE 和 problem-level 设置。 |
| `g.units_section(...)` | `UnitsSection` | length/mass/time 单位。 |
| `g.material_section(...)` | `MaterialSection` | material model 和参数。 |
| `g.body_section(...)` | `BodySection` | 单个 body 的 geometry/material/BC。 |
| `g.bodies_section(...)` | `list[BodySection]` | 多 body 组合。 |
| `g.fixed_surface_section(...)` | `FixedSurfaceSection` | 常用 Dirichlet fixed surface。 |
| `g.loads_section(...)` | `LoadsSection` | RHS / body force。 |
| `g.space_section(...)` | `SpaceSection` | discretization settings。 |
| `g.time_section(...)` | `TimeSection` | transient time settings。 |
| `g.solver_section(...)` | `SolverSection` | linear/nonlinear/contact solver settings。 |
| `g.contact_section(...)` | `ContactSection` | contact model/options。 |
| `g.results_section(...)` | `ResultsSection` | Python-facing requested fields。 |
| `g.output_section(...)` | `OutputSection` | solver file output options。 |
| `g.simulation_template(...)` | `SimulationTemplate` | 顶层 guided config container。 |
| `g.build_config(...)` | `SimulationConfig` | 把 guided template 转成 solver-facing config。 |

## Template Names

推荐新代码使用：

```python
g.simulation_template(...)
```

兼容旧代码的名字：

```python
g.experiment_template(...)
```

两者当前返回同一种 template object。`experiment_template` 只是旧 example flow 的
compatibility alias；文档和新 examples 应该使用 `simulation_template`。

## `body_section(...)` Contract

`body_section(...)` 必须且只能给一种 geometry source。

文件 mesh：

```python
g.body_section(
    name="body",
    mesh="mesh.msh",
    material=g.material_section(...),
)
```

array-backed mesh：

```python
g.body_section(
    name="body",
    vertices=V,
    cells=C,
    material=g.material_section(...),
)
```

`faces` 是 `cells` 的 alias：

```python
g.body_section(
    name="body",
    vertices=V,
    faces=F,
    material=g.material_section(...),
)
```

必须保留的语义：

- `mesh` 和 `vertices/cells` 不能同时传；
- array-backed body 必须同时有 `vertices` 和 `cells`/`faces`；
- `cells` 和 `faces` 不能同时传；
- array-backed bodies 当前不能和 mesh-file bodies 混用；
- array-backed bodies 当前不支持 `transformation`、`advanced`、`n_refs`。

array-backed template 会在 `build_config(...)` 时把 payload 存到：

```python
cfg.extras["_mesh_array_mode"]
```

之后 `solve(...)` 会在 mesh normalization 阶段读取它。

## `build_config(...)` Contract

`g.build_config(template, workspace)` 做这些事：

1. 要求 `template.bodies` 至少有一个 body；
2. 创建 `SimulationConfig()`；
3. 写入 PDE、units、space、time、solver、contact、output；
4. 把每个 `BodySection` 转成 geometry + material + boundary conditions；
5. 如果是 array-backed body，构造 `_mesh_array_mode` payload；
6. 把 `results_section(requested_fields=...)` 接到 output result request；
7. 返回 `SimulationConfig`，不运行 solver。

它不做这些事：

- 不 call `solve(...)`；
- 不 build differentiable problem；
- 不计算 loss；
- 不做 optimization；
- 不保存训练样本。

## Output vs Results

`output_section(...)` 和 `results_section(...)` 不是一回事。

| Section | 关注点 |
| --- | --- |
| `output_section(...)` | solver-facing 文件输出，例如 Paraview/VTU/log/data files。 |
| `results_section(...)` | Python-facing result extraction，例如 `u`, `stress`, `von_mises`。 |

常用 examples 应该优先写：

```python
results = g.results_section(requested_fields=["u", "von_mises"])
```

如果需要保存 VTU/log，再配合 runtime helpers：

```python
from polyfempy.api.runtime import result_output, terminal_log

result_output(cfg, directory=str(workspace), save_vtu=True)
terminal_log(cfg, print_terminal=False)
```

## 不建议用户直接依赖的实现细节

这些可以存在，但不应该作为新用户的第一入口：

- `polyfempy.api.guided_sections`：compatibility facade for guided internals；
- `polyfempy.api._guided_config`：guided template -> `SimulationConfig` translation；
- `polyfempy.api._guided_array_mesh`：internal payload builder；
- `add_body_from_section(...)` / `build_material(...)` 等 builder helpers。

## 测试保护

修改 guided API 后至少跑：

```bash
python -m pytest tests/test_import_public_api.py
python -m pytest tests/test_guided_config_builder.py
python -m pytest tests/test_config_typed_blocks.py tests/test_geometry_transformations.py
```

如果改了 array-backed body path，还要跑 solve pipeline 相关 tests：

```bash
python -m pytest tests/test_pipeline_normalize.py tests/test_pipeline_sampled_fallback.py
```
