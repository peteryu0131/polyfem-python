# PolyFEM Python API 使用指南

这份文档面向当前仓库里的 `polyfempy.api`。

目标不是覆盖全部 PolyFEM 数学细节，而是回答下面这些最实际的问题：

- 从 Python 里应该怎么启动一次仿真
- `cfg` 到底可以传什么
- 如果我先读 JSON，再在 class 里改参数，最后到底用谁
- `Result` 里能直接拿什么，拿不到什么
- `stress` / `von_mises` 现在推荐怎么拿
- `Output` 现在该怎么理解，怎样同时控制“文件输出”和“Python 返回结果”
- 哪些地方已经很好用了，哪些地方现在还有坑

---

## 1. API 分层

当前 Python 侧大致分两层。

### `polyfempy.api`

这是普通前向求解接口，适合：

- 跑一次仿真
- 拿位移、压力、速度、应变等结果
- 读 mesh / 写结果
- 做非可微的后处理

最核心的入口有：

- `polyfempy.api.solve`
- `polyfempy.api.SimulationConfig`
- `polyfempy.api.Result`
- `polyfempy.api.Output`

### `polyfempy.differentiable`

这是可微仿真接口，适合：

- 形状优化
- 材料参数优化
- 在 PyTorch 训练循环里把 PolyFEM 当成一个可微模块来调用

一句话区分：

- 只想跑 forward，先看 `polyfempy.api`
- 想 `loss.backward()`，看 `polyfempy.differentiable`

---

## 2. 最常用入口：`solve(...)`

最常见的调用方式是：

```python
from polyfempy.api import solve

result = solve(cfg=...)
```

当前 `solve` 支持两大输入模式：

1. JSON / mesh-file 模式
2. array 模式

### 2.1 JSON / mesh-file 模式

mesh 从配置里读取：

```python
result = solve(cfg="config.json")
```

或者：

```python
from polyfempy.api import SimulationConfig, solve

cfg = SimulationConfig.from_json_file("config.json")
result = solve(cfg=cfg)
```

这是当前最稳、最接近你现有实验脚本的用法。

### 2.2 array 模式

你已经在 Python 里有顶点和单元：

```python
result = solve(vertices=V, cells=C, cfg=cfg)
```

适合：

- mesh 已经在内存里
- 不想让 JSON 去找文件
- 想把 PolyFEM 嵌到更纯的 Python 流程里

如果 mesh 没有合适的 sideset / boundary id，有时还需要 `sidesets_func` 帮你补边界编号。

---

## 3. `cfg` 可以怎么写

`cfg` 目前支持 3 种主流形式。

### 3.1 直接传 JSON 路径

```python
result = solve(cfg="config.json")
```

优点：

- 最贴近现有 PolyFEM 配置习惯
- 对复杂场景最稳
- 最适合复用已有实验目录

### 3.2 传 Python `dict`

```python
cfg = {
    "pde": "LinearElasticity",
    "materials": [{"type": "LinearElasticity", "E": 2100.0, "nu": 0.3}],
    "geometry": [{"mesh": "mesh.msh"}],
    "boundary_conditions": {
        "dirichlet_boundary": [{"id": 4, "value": [0.0, 0.0]}],
    },
}

result = solve(cfg=cfg)
```

优点：

- 适合脚本里动态改参数
- 不必先写一个 JSON 文件

### 3.3 传 `SimulationConfig`

这是更 Python 风格、IDE 自动补全更好的写法：

```python
from polyfempy.api import (
    solve,
    SimulationConfig,
    Material,
    BoundaryConditions,
    Geometry,
    GeometryMesh,
    Solver,
    LinearSolver,
    Time,
    Output,
)

bc = BoundaryConditions()
bc.add_dirichlet(id=4, value=[0.0, 0.0])

cfg = SimulationConfig(
    pde="LinearElasticity",
    materials=Material(E=2100.0, nu=0.3),
    boundary_conditions=bc,
    geometry=Geometry(meshes=[GeometryMesh(mesh="mesh.msh")]),
    solver=Solver(linear=LinearSolver(solver_type="Eigen::SparseLU")),
    time=Time(tend=0.1, dt=0.01),
    output=Output(directory="out"),
)

result = solve(cfg=cfg)
```

优点：

- key 不容易拼错
- IDE 体验更好
- 更适合进一步封装成自己的 scene builder / helper API

---

## 4. JSON 与 class 覆盖规则

这是当前最重要的一条行为规则：

**如果你先 `from_json_file(...)`，再在 Python 里改 `cfg` 上的字段，运行时会优先使用你后来改的 class 值。**

也就是说：

- JSON 里 `E = 20`
- Python 里你后来改成 `E = 50`
- 最终 `solve(cfg=cfg)` 会按 `E = 50` 跑

例如：

```python
from polyfempy.api import SimulationConfig, solve

cfg = SimulationConfig.from_json_file("config.json")
cfg.materials[0]["E"] = {"value": 50, "unit": "MPa"}

result = solve(cfg=cfg)
```

这条覆盖规则现在适用于最常改的几类字段：

- `materials`
- `time`
- `output`
- `solver`
- `geometry`
- `contact`

这背后的设计原则是：

- JSON 负责保留完整原始配置
- Python 对象负责表达“当前真正要跑的配置”

### 4.1 JSON 序列化要用哪个接口

如果你想把当前 `SimulationConfig` 真正完整地导出成 JSON，再以后读回来，应该用：

```python
s_dict = cfg.to_full_json_dict()
s = cfg.to_full_json_str()
cfg2 = SimulationConfig.from_full_json_str(s)
```

这里：

- `to_full_json_dict()` 适合你还想在 Python 里继续改这个配置字典
- `to_full_json_str()` 适合你要把它写到文件、缓存成字符串、或者传给别的系统

它们都会基于完整当前配置导出。

旧的 `to_json_str()` 还保留着，但它只是一个历史上的 minimal export，只包含较少的核心字段，不适合做完整 round-trip。现在如果你真的要走这条兼容路径，应该显式配对 `to_minimal_json_str()` / `from_minimal_json_str()`。

最常见的使用场景有这些：

1. 你先 `from_json_file("config.json")`，再在 Python 里改了 `E/tend/output`，这时想把“改完之后真正要跑的版本”完整保存下来。
2. 你想做参数扫描，用一个 JSON 模板生成很多变体，每个变体都先导成完整 dict 再统一落盘。
3. 你想在 notebook 或脚本里打印、比较两个配置到底差了哪些字段，这时 `to_full_json_dict()` 比字符串更方便。
4. 你想把当前配置重新喂给 `SimulationConfig.from_json_dict(...)` 或别的 JSON 管线做 round-trip。

---

## 5. `SimulationConfig` 的主要组成

`SimulationConfig` 是顶层容器。最常见的子对象如下。

### 5.1 材料

最基础的是：

- `Material(E=..., nu=..., rho=...)`

也支持更具体的本构类，比如：

- `NeoHookean`
- `IsochoricNeoHookean`
- `SaintVenant`
- `MooneyRivlin`
- `LinearElasticity`

如果只是普通线弹性问题，通常先用 `Material` 或 `LinearElasticity` 就够了。

### 5.2 边界条件

最常用的是：

- `BoundaryConditions`
- `DirichletBoundary`
- `NeumannBoundary`

例如：

```python
bc = BoundaryConditions()
bc.add_dirichlet(id=4, value=[0.0, 0.0])
bc.add_neumann(id=2, value=[0.0, -1000.0])
bc.rhs = [0.0, -980.0]
```

### 5.3 几何

最常用的是：

- `Geometry`
- `GeometryMesh`

例如：

```python
geom = Geometry(
    meshes=[
        GeometryMesh(mesh="triangular_lattice.msh", volume_selection=1),
        GeometryMesh(mesh="falling_weight_block.msh", volume_selection=2),
    ]
)
```

### 5.4 求解器

最常用的是：

- `Solver`
- `LinearSolver`
- `NonlinearSolver`

例如：

```python
solver = Solver(
    linear=LinearSolver(solver_type="Eigen::SparseLU"),
    nonlinear=NonlinearSolver(solver_type="Newton", max_iterations=100),
    max_threads=1,
)
```

### 5.5 时间

瞬态问题用：

```python
time = Time(t0=0.0, tend=0.1, dt=0.01)
```

注意：

- `Time` 要么给 `dt`
- 要么给 `time_steps`
- 不能两个都给

### 5.6 接触

```python
from polyfempy.api import Contact

contact = Contact(enabled=True, dhat=0.01, mu=0.0)
```

### 5.7 几何选择

如果 mesh 没有现成的 sideset id，可以用 `Selection` 做几何选择：

```python
from polyfempy.api import Selection

sel = Selection()
sel.select_sideset_with_box(id=1, box_min=[0, 0], box_max=[1, 0.1])
```

这更适合“按几何位置选边界”，而不是“按 mesh 标号选边界”。

---

## 6. `Output` 现在怎么理解

`Output` 现在有两层含义，需要明确区分。

### 6.1 solver-facing 输出

这是传统意义上的文件输出配置，比如：

- `directory`
- `paraview`
- `json`
- `log`
- `advanced`

例如：

```python
from polyfempy.api import Output, ParaviewOutput

output = Output(
    directory="out",
    json=False,
    paraview=ParaviewOutput(file_name="sim.pvd"),
)
```

### 6.2 Python-facing 结果请求

这是现在新加进 `Output` 里的 Python 侧运行控制，用来表达：

- 我想让 `solve()` 最后尽量返回哪些字段
- 如果字段拿不到，是否允许 fallback
- 临时 VTU 是放 RAM 还是 disk

这部分由两个子对象表达：

- `ResultOutput`
- `FallbackOutput`

或者直接用 `Output` 的便利方法：

```python
output = (
    Output(directory="out", json=False, save_paraview=False)
    .request_results(["u", "stress", "von_mises"], strict=False)
    .configure_fallback(sampled_vtu="auto", temp_storage="ram")
)
```

这是现在最推荐的写法。

### 6.3 `save_paraview=False`

`save_paraview=False` 是一个 Python 侧便利开关，它会自动做两件事：

- 清掉 `paraview.file_name`
- 把 `advanced.save_time_sequence` 关掉

这样用户不用自己手动去改两个位置。

---

## 7. 推荐的 `Output` 用法

### 7.1 只要位移，不要 VTU

```python
cfg.output = Output(
    directory="out",
    json=False,
    save_paraview=False,
).request_results(["u"])
```

### 7.2 希望尽量拿 `stress` 和 `von_mises`，但不保留 VTU

```python
cfg.output = (
    Output(directory="out", json=False, save_paraview=False)
    .request_results(["u", "stress", "von_mises"], strict=False)
    .configure_fallback(sampled_vtu="auto", temp_storage="ram")
)
```

这表示：

- 文件层面不保留 Paraview 时序
- Python 返回层面尽量拿 `u/stress/von_mises`
- 如果原生结果没有这些字段，就临时导一个 VTU，再读回来

### 7.3 拿不到就报错

```python
cfg.output = (
    Output(directory="out", json=False, save_paraview=False)
    .request_results(["u", "stress", "von_mises"], strict=True)
    .configure_fallback(sampled_vtu="auto")
)
```

如果最终仍然缺字段，`solve()` 会抛异常。

---

## 8. `solve()` 返回的 `Result`

`Result` 是 `polyfempy.api` 里的核心结果容器。

最常用的属性有：

- `result.vertices`
- `result.cells`
- `result.u`
- `result.p`
- `result.stress`
- `result.strain`
- `result.von_mises`
- `result.fields`
- `result.meta`

### 8.1 最常直接拿的字段

```python
u = result.u
stress = result.stress
vm = result.von_mises
```

### 8.2 看有哪些字段

```python
print(result.field_names())
print(result.meta)
```

### 8.3 访问任意字段

```python
velocity = result.field("v")
energy = result.field("energy")
```

### 8.4 转成 torch

```python
result.to_torch(include_mesh=True)
u = result.u
```

注意：

- `Result.to_torch()` 只是把数据容器转到 torch
- 它**不会**把这次 forward solve 自动变成可微 graph

如果你想真的让 PolyFEM 参与 autograd，还是应该用 `polyfempy.differentiable`

---

## 9. `stress` 和 `von_mises` 现在怎么拿

这是最常被问到的部分。

### 9.1 最理想情况

如果底层 native result bundle 直接给了 `stress`，那么：

```python
stress = result.stress
vm = result.von_mises
```

这里的 `result.von_mises` 会优先：

1. 复用已有 `von_mises` 字段
2. 没有的话，从 `result.stress` 现算

### 9.2 如果 native result 没有 `stress`

当前 `solve()` 支持临时 VTU fallback：

```python
cfg.output = (
    Output(directory="out", json=False, save_paraview=False)
    .request_results(["u", "stress", "von_mises"])
    .configure_fallback(sampled_vtu="auto", temp_storage="ram")
)
```

这时 `solve()` 会：

1. 正常跑 solver
2. 如果缺 `stress` / `von_mises`
3. 临时导一个 VTU
4. 用 `meshio` 读回 sampled 字段
5. 把它们塞回 `Result`

### 9.3 重要限制

通过临时 VTU 读回的 `stress` / `von_mises`，通常是 **sampled visualization mesh** 上的字段。

所以它们的长度可能和 `result.u` 对不上，例如：

- `u.shape = (n_vertices, dim)`
- `stress.shape = (n_sample_points, 3 or 6)`

这不是 bug，而是两套 mesh 语义不同。

最安全的理解是：

- `u` 更接近原始解空间 / 原始 mesh
- sampled `stress` / `von_mises` 更接近可视化或采样结果

所以：

- 可以拿来做统计、分析、画图
- 但不要假设它和 `result.vertices` 是一一对应的 nodal field

---

## 10. `Result.meta` 里现在会记录什么

如果你用了结果请求和 fallback，`Result.meta` 里会尽量记录一些诊断信息，比如：

- `requested_fields`
- `missing_requested_fields`
- `sampled_vtu_fallback`
- `sampled_vtu_temp_storage`
- `stress_source`
- `stress_location`
- `von_mises_source`
- `von_mises_location`

这很适合用来判断：

- 我想要的字段到底有没有拿到
- 是原生拿到的，还是 fallback 拿到的

例如：

```python
print(result.meta.get("requested_fields"))
print(result.meta.get("missing_requested_fields"))
print(result.meta.get("sampled_vtu_fallback"))
```

---

## 11. 推荐工作流

### 11.1 复用现有实验配置

这是当前最推荐的主线：

```python
from polyfempy.api import SimulationConfig, Output, solve

cfg = SimulationConfig.from_json_file("config.json")
cfg.output = (
    Output(directory="out", json=False, save_paraview=False)
    .request_results(["u", "stress", "von_mises"], strict=False)
    .configure_fallback(sampled_vtu="auto", temp_storage="ram")
)

result = solve(cfg=cfg)
```

适合：

- 你已经有一套 JSON case
- 想用 Python 改少量参数
- 想避免保留大批 VTU 文件

### 11.2 先读 JSON，再覆盖参数

```python
cfg = SimulationConfig.from_json_file("config.json")
cfg.time.tend = 0.1
cfg.output = Output(directory="out", json=False, save_paraview=False)
cfg.materials[0]["E"] = {"value": 50, "unit": "MPa"}

result = solve(cfg=cfg)
```

适合：

- 把 JSON 当模板
- 再用 Python 批量扫参数

### 11.3 纯 class API

```python
cfg = SimulationConfig(
    pde="LinearElasticity",
    materials=Material(E=2100.0, nu=0.3),
    geometry=Geometry(meshes=[GeometryMesh(mesh="mesh.msh")]),
    boundary_conditions=BoundaryConditions(),
    output=Output(directory="out", json=False, save_paraview=False),
)

result = solve(cfg=cfg)
```

适合：

- 新项目
- 想完全用 Python 表达场景

---

## 12. 辅助模块

除了 `solve/config/result`，还有几个常用辅助模块。

### 12.1 `io.py`

最主要是：

- `read_mesh(path)`
- `Mesh`

例如：

```python
from polyfempy.api import read_mesh

mesh = read_mesh("mesh.msh")
result = solve(vertices=mesh.vertices, cells=mesh.cells, cfg=cfg)
```

### 12.2 `selection.py`

用于用几何方式选 sideset / body，而不是依赖 mesh 里现成 marker。

### 12.3 `tensor.py`

用于在 `numpy / torch / jax` 之间做轻量转换。

当前 `solve()` 自己也会用它来：

- 把输入正规化成 NumPy
- 把结果转回原 backend

### 12.4 `batch.py`

提供一个最简单的顺序 batch solve：

```python
from polyfempy.api import batch_solve

results = batch_solve([
    (V1, C1, cfg1),
    (V2, C2, cfg2),
])
```

注意：

- 现在是顺序执行
- 不是并行 batch

---

## 13. 当前已知限制

这是当前 API 最值得记住的限制。

### 13.1 `Result` 里不同字段不一定来自同一套 mesh

尤其是用了 sampled VTU fallback 以后：

- `u` 和 `vertices`
- `stress` 和 `von_mises`

未必是一一对应的。

### 13.2 `result.force` 不是当前标准字段

现在 `Result` 默认没有一个稳定统一的：

- `result.force`
- `result.contact_force`
- `result.reaction_force`

如果以后要支持，建议单独设计语义，而不是直接塞进现有 point field。

### 13.3 `solve()` 本身还是比较大

它现在同时承担了：

- cfg 规范化
- JSON / array 模式切换
- solver 设置
- 求解
- 结果提取
- fallback

所以后续维护时，最好继续拆分。

### 13.4 `Output` 还是一个“混合对象”

现在虽然用户层只用一个 `Output` 更方便了，但内部其实仍然有两层语义：

- solver-facing 文件输出
- python-facing 结果请求

这对用户是简化了，但对实现层来说仍然需要小心区分。

---

## 14. 当前最推荐的最小模板

如果你今天就想稳定开始用，我最推荐这条：

```python
from polyfempy.api import SimulationConfig, Output, solve

cfg = SimulationConfig.from_json_file("config.json")
cfg.output = (
    Output(directory="out", json=False, save_paraview=False)
    .request_results(["u", "stress", "von_mises"], strict=False)
    .configure_fallback(sampled_vtu="auto", temp_storage="ram")
)

result = solve(cfg=cfg)

print(result.field_names())
print(result.meta)
print(result.u.shape if result.u is not None else None)
print(result.stress.shape if result.stress is not None else None)
print(result.von_mises.shape if result.von_mises is not None else None)
```

这条模板最适合：

- 保留现有 JSON case
- Python 里做少量覆盖
- 尽量不保留 VTU 时序文件
- 但还是想尽量拿到 stress / von Mises

---

## 15. 一句话总结

当前 `polyfempy.api` 最适合的使用方式是：

- 用 JSON case 当模板
- 用 `SimulationConfig.from_json_file(...)` 读进来
- 再用 Python/class 覆盖少量参数
- 用 `Output(...)` 同时控制文件输出和结果请求
- 用 `Result` 统一接收位移、stress、von Mises 和元信息

如果你要做优化或 PyTorch autograd，再切到 `polyfempy.differentiable`。

---

## 16. 我建议后续优先改进的地方

如果后面要继续打磨这套 API，我最建议优先做这几件事。

### 16.1 把 `solve()` 继续拆小

当前 `solve()` 同时负责：

- cfg 规范化
- JSON / array 模式切换
- solver 初始化
- mesh 加载
- 求解
- 结果提取
- VTU fallback

这对用户是方便的，但对维护不太友好。后续很适合拆成：

- config normalization
- solver setup
- forward solve
- result extraction
- fallback extraction

### 16.2 明确区分 native result mesh 和 sampled result mesh

现在 `Result` 里虽然统一挂字段很方便，但 sampled `stress/von_mises` 和原始 `u/vertices` 经常不是同一套网格。

后续更干净的设计可以是：

- `result.native`
- `result.sampled`

或者至少在 `Result` 内部把字段来源和 mesh 语义分层。

### 16.3 给 `force / reaction / contact_force` 一个正式接口

现在用户最自然会问：

- 能不能直接拿 `force`
- 能不能直接拿接触力
- 能不能直接拿 reaction force

这部分现在还没有稳定标准字段。后续最好单独设计，而不是把它们硬塞成普通 point field。

### 16.4 让 `Output` 的双重语义更显式

现在 `Output` 已经比以前好用，但内部仍然混着两类东西：

- solver-facing 文件输出
- python-facing 结果请求与 fallback

从用户角度这已经够顺手了，但实现层最好继续把这两层边界写清楚。

### 16.5 给常见工作流做一层更薄的高层 wrapper

当前 API 已经够用，但对新用户来说还是有点“配置系统味道太重”。

后面如果想让它更像真正的 Python API，可以再包一层更小的入口，比如：

- `load_case("config.json")`
- `case.override(E=50, tend=0.1)`
- `case.solve(request=["u", "stress", "von_mises"])`

这样会更接近普通 Python 用户的直觉。
