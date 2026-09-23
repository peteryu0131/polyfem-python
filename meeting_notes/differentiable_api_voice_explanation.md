# Differentiable API Voice Explanation Notes

这份文档是给语音讲解用的，不是最终 API 承诺。它的目标是把会议后的 differentiable API 思路讲清楚，尤其是老师指出的两点：

- forward setup 仍然用 `polyfem.model()`，不要重新发明一套 forward 写法。
- differentiable setup 应该从完成后的 forward model/state 开始：`diff_model = diff.model([model])`，不是把用户推到裸 JSON payload 上。

建议把这份文档发给 ChatGPT 语音，然后说：

```text
请用中文语音，用对话方式给我讲这份文档。
请多用大白话，不要跳过中间逻辑。
如果遇到代码，请先讲它在做什么，再讲为什么这样设计。
```

## 1. 一句话总结

会议后的主线应该是：

```text
polyfem.model()
    负责 forward 建模，和现有 generated forward examples 保持一致

diff.model([model])
    把一个或多个 forward model/state 包进 differentiable context

diff.ShapeOpt.apply(...)
    是 PyTorch autograd 边界

diff.MaxStress / diff.StressNorm / diff.VonMises
    表达 PolyFEM 需要求导的物理 objective

torch
    负责 params、optimizer、loop，以及用户自己写的外层变换
```

最重要的变化是：第一个真实 shape/stress example 不应该只说“对 solution 随便写一个 loss 就能自动优化 stress”。对 stress / von Mises / max stress 这类物理目标，backend backward 需要知道具体 objective 是什么。

## 2. 会议后的 API 形状

forward 部分保持现在 generated API 的样子：

```python
from polyfempy.generated_api import generated_api as polyfem
from polyfempy import differentiable_api as diff

model = polyfem.model()

rubber = polyfem.neo_hookean(E=1e5, nu=0.3, rho=1150)

body = model.mesh(mesh="beam.msh")
body.material(rubber)

model.config(
    rhs=[0, -9.8],
    time_tend=0.02,
    time_dt=0.01,
)
```

老师提到 `config` 这个名字以后可能可以再讨论，`rhs` / `time` 也可能变成更清楚的 helper。但这不影响核心结论：

```text
forward model 的搭建仍然在 polyfem.model() 层。
diff API 不重新设计 forward 建模语言。
```

然后进入 differentiable：

```python
diff_model = diff.model([model])
```

这里用 list 是故意的：

```python
diff_model = diff.model([state_a, state_b])
```

MVP 只需要一个 state，但 list 形状让以后 multi-state problem 不用换 API。

## 3. 为什么是 `diff.model([model])`

`diff.model([model])` 不是 solve，也不是 optimizer。

它只是一个 differentiable container：

```text
我已经用 polyfem.model() 建好了一个 forward problem。
现在请把它登记成一个可微分问题。
后面的 ShapeOpt / MaterialOpt / BCOpt 都从这里拿 forward 设置。
```

大白话：

```text
polyfem.model() 是普通仿真模型。
diff.model([model]) 是把普通仿真模型放进可微分世界。
```

为什么不用裸 config 变量作为主线？

```text
因为用户现在写 forward example 时，最自然看到的是 model / body / material。
老师也明确说 diff.model(...) 应该 with model，不应该让 meeting example 变成只围绕 raw config payload。
```

实现上，底层仍然可以在 `diff.model(...)` 里面调用 `model.config()`，得到 backend 需要的 dict。这个转换属于 wrapper 内部，不应该暴露成用户主线。

## 4. 第一个真实目标：shape + stress objective

老师会议里的重点不是“做一个随便返回 solution 的函数”，而是：

```text
shape optimization 要知道优化的物理目标是什么。
比如 MaxStress、StressNorm、von Mises。
```

所以第一个 stress-oriented API 更应该长这样：

```python
objective = diff.MaxStress(selection=body)


def shape_loss(vertices: torch.Tensor) -> torch.Tensor:
    return diff.ShapeOpt.apply(
        model=diff_model,
        selection=body,
        tensor=vertices,
        objective=objective,
    )
```

大白话：

```text
vertices 是当前 mesh 顶点。
ShapeOpt.forward 用这些 vertices 跑 PolyFEM，并计算 objective value。
ShapeOpt.backward 调 PolyFEM adjoint，返回 objective 对 vertices 的梯度。
```

这比“只返回 solution u，然后用户随便写 loss”更适合第一个 stress example，因为 stress objective 不是普通 solution tensor 上的简单数学函数。它依赖 PolyFEM 内部应力、单元、材料、时间等信息。

## 5. `ShapeOpt` 到底是什么

`ShapeOpt` 应该是一个 `torch.autograd.Function`。

它的概念结构类似：

```python
class ShapeOpt(torch.autograd.Function):
    @staticmethod
    def forward(ctx, model, selection, tensor, objective):
        ...

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        ...
```

它不是 optimizer。

它不做：

```text
Adam
loop count
learning rate
stop condition
```

这些仍然由 PyTorch 做：

```python
optimizer = torch.optim.Adam([vertices], lr=1e-2)

for step in range(50):
    optimizer.zero_grad()
    loss = shape_loss(vertices)
    loss.backward()
    optimizer.step()
```

`ShapeOpt` 真正负责的是：

```text
forward:
    vertices -> PolyFEM solve/objective -> scalar loss

backward:
    scalar loss gradient -> PolyFEM adjoint -> dloss/dvertices
```

这里有一个非常重要的说法要讲准确：

```text
不是“不用 adjoint”。

而是：
    用户 API 里不直接暴露 adjoint。
    实现层的 ShapeOpt.backward() 必须调用 PolyFEM adjoint。
```

也就是说，用户看到的是：

```python
loss = diff.ShapeOpt.apply(
    model=diff_model,
    selection=body,
    tensor=vertices,
    objective=objective,
)
loss.backward()
```

用户不写：

```python
adjoint(...)
```

但内部真实发生的是：

```text
loss.backward()
    -> ShapeOpt.backward()
    -> PolyFEM objective-specific adjoint
    -> dObjective/dvertices
```

老师强调的不是“别用 adjoint”，而是：

```text
不能只给 backend 一个 shape variable，然后期待它算任意 gradient。
必须告诉 backend objective 是什么，比如 MaxStress。
```

所以正确理解是：

```text
Objective J + shape vertices x
        -> adjoint
        -> dJ/dx
```

而不是：

```text
shape vertices x
        -> generic gradient
```

## 6. 老师的 `def loss(params)` 在说什么

老师写的核心结构是：

```python
def loss(params):
    x = ...  # slice params, transform params, etc
    sol = diff.ShapeOpt.apply(...)
    return ...
```

这里最重要的是 `params -> x`。

大白话：

```text
用户不一定直接优化所有 mesh vertices。
用户可能优化 angle、thickness、height 这种低维参数。
然后用 PyTorch 把这些参数变成 mesh vertices。
```

例如：

```python
params = torch.tensor(
    [initial_angle, initial_thickness, initial_height],
    dtype=torch.float64,
    requires_grad=True,
)


def build_vertices_from_params(params: torch.Tensor) -> torch.Tensor:
    angle, thickness, height = params

    x = reference_vertices.clone()
    x[:, 0] = x[:, 0] * thickness
    x[:, 1] = x[:, 1] * height

    rotation = torch.stack(
        [
            torch.stack([torch.cos(angle), -torch.sin(angle)]),
            torch.stack([torch.sin(angle), torch.cos(angle)]),
        ]
    )
    return x @ rotation.T
```

然后：

```python
def loss(params: torch.Tensor) -> torch.Tensor:
    x = build_vertices_from_params(params)
    return diff.ShapeOpt.apply(
        model=diff_model,
        selection=body,
        tensor=x,
        objective=diff.MaxStress(selection=body),
    )
```

这条链是：

```text
loss -> objective -> vertices/x -> params
```

分工是：

```text
PolyFEM adjoint:
    objective -> vertices/x

PyTorch:
    vertices/x -> params
```

所以 `build_vertices_from_params` 必须全程用 torch operations。不要写：

```python
x.detach()
x.numpy()
x.item()
with torch.no_grad():
    ...
```

这些会断开 PyTorch 计算图。图断了以后，梯度就不能从 vertices 继续传回 angle/thickness/height。

## 7. 老师写的两个 `return` 是什么

老师写过类似：

```python
return torch.linalg.norm(sol[:, -1]) * torch.linalg.norm(sol[:, -1])
# or
return torch.nn.functional.mse_loss(sol, target_u)
```

这不是让你两个都写。意思是：

```text
如果某个 differentiable operation 返回 solution sol，
用户可以用 PyTorch 写不同的 scalar loss。
```

### 7.1 solution norm loss

```python
def loss_from_solution_norm(vertices: torch.Tensor) -> torch.Tensor:
    sol = diff.ShapeSolve.apply(diff_model, body, vertices)
    return torch.linalg.norm(sol[:, -1]) * torch.linalg.norm(sol[:, -1])
```

大白话：

```text
取 solution 的某一部分，算 norm，把它变成一个 scalar。
```

这里 `sol[:, -1]` 只是老师举例。真实代码要看 backend 返回的 solution layout。

### 7.2 target matching loss

```python
target_u = ...


def loss_from_target_solution(vertices: torch.Tensor) -> torch.Tensor:
    sol = diff.ShapeSolve.apply(diff_model, body, vertices)
    return torch.nn.functional.mse_loss(sol, target_u)
```

`target_u` 不是必需文件。它只是目标匹配问题里的目标 displacement / solution。

它可以来自：

```text
实验测量
另一次 simulation
老师给的数据
用户自己构造的目标
```

会议后的建议是：

```text
这些 solution-returning loss 作为 future / optional route 保留。
第一个真实 stress example 先走 objective-aware ShapeOpt。
```

## 8. 为什么不能只说“任意 PyTorch loss 都行”

这句话一半对，一半容易误导。

对的部分：

```text
如果 backend operation 返回的是普通 torch tensor，
用户确实可以用 PyTorch 写 scalar loss。
```

容易误导的部分：

```text
stress / max stress / von Mises 不一定只是 solution tensor 上的普通函数。
它可能需要 PolyFEM 内部状态。
adjoint backward 也需要知道正在对哪个物理 objective 求导。
```

所以更准确的说法是：

```text
Route A, primary for stress MVP:
    ShapeOpt receives an objective and returns scalar objective loss.

Route B, optional/future:
    ShapeSolve returns solution tensor, then user writes PyTorch loss from sol.
```

## 9. 以前的 `make_von_mises_loss(...)` 属于哪里

你以前的旧 example 有：

```python
loss_fn = make_von_mises_loss(body=OBJECTIVE_BODY_ID, time="smooth_max")
```

这个概念不应该丢。

它对应新 API 里的：

```python
objective = diff.MaxStress(selection=body)
```

或者未来：

```python
objective = diff.VonMises(selection=body, time="smooth_max")
objective = diff.StressNorm(selection=body, power=8)
```

区别是：

```text
旧 API:
    polyfempy.differentiable + old guided/api/config stack

新 API:
    polyfem.model()
    diff.model([model])
    diff.ShapeOpt.apply(..., objective=objective)
```

也就是说，旧 API 的 objective 思路可以保留，但不要复制旧架构。

## 10. `objective-spec.json` 和 diff objectives 的关系

PolyFEM 里有：

```text
polyfem/json-specs/objective-spec.json
```

它描述 objective / functionals 的 JSON 结构，例如：

```text
max_stress
stress_norm
compliance
volume
```

所以长期更好维护的方向是：

```text
objective-spec.json
    -> generated or validated objective helper layer
    -> diff.MaxStress(...)
    -> diff.StressNorm(...)
```

当前 repo 里的：

```text
polyfempy/differentiable_api/objectives.py
```

只是一个小的 handwritten helper。它可以先用于表达 API 方向，但长期应该尽量和 `objective-spec.json` 对齐，避免手写 helper 和 C++ 支持能力慢慢跑偏。

## 11. `input-spec.json` 和 forward API 的关系

forward generated API 来自：

```text
polyfem/json-specs/input-spec.json
```

所以这些东西属于 forward layer：

```python
polyfem.model()
polyfem.neo_hookean(...)
polyfem.solver(...)
polyfem.output(...)
model.config(...)
```

differentiable API 不应该替代它们。它应该接在后面：

```python
model = polyfem.model()
...
model.config(...)

diff_model = diff.model([model])
```

这也是为什么 meeting example 应该看起来像 forward example 的自然延伸，而不是像另一个库。

## 12. 当前实现和目标 API 的差别

当前 repo 已经有一些真实代码：

```text
polyfempy/differentiable_api/model.py
polyfempy/differentiable_api/torch_ops.py
polyfempy/differentiable_api/objectives.py
polyfempy/differentiable_api/shape.py
```

当前已经比较接近的部分：

```text
diff.model([...])
diff.ShapeOpt.apply(...)
basic objective helper objects
```

但是会议后的目标 API 里，下面这些仍然是 proposed / future：

```text
import polyfem
import polyfem.diff as diff

diff.ShapeOpt.apply(..., objective=objective)
diff.MaxStress(...)
diff.VonMises(...)
diff.shape_vertices(...)
clean generated objective helper API
full backend objective-aware adjoint path
```

所以汇报时要说清楚：

```text
这是目标 API shape。
当前 MVP 正在把 diff.model / ShapeOpt / objective helper 这些边界搭出来。
```

不要说：

```text
这些全部已经完成。
```

## 13. 一份更适合开会展示的主 example

```python
from polyfempy.generated_api import generated_api as polyfem
from polyfempy import differentiable_api as diff

import torch

model = polyfem.model()

rubber = polyfem.neo_hookean(E=1e5, nu=0.3, rho=1150)

body = model.mesh(mesh="beam.msh")
body.material(rubber)

model.config(
    rhs=[0, -9.8],
    time_tend=0.02,
    time_dt=0.01,
)

diff_model = diff.model([model])

vertices = diff.shape_vertices(
    diff_model,
    selection=body,
    requires_grad=True,
)

objective = diff.MaxStress(selection=body)


def loss(vertices: torch.Tensor) -> torch.Tensor:
    return diff.ShapeOpt.apply(
        model=diff_model,
        selection=body,
        tensor=vertices,
        objective=objective,
    )


optimizer = torch.optim.Adam([vertices], lr=1e-2)

for step in range(50):
    optimizer.zero_grad()
    value = loss(vertices)
    value.backward()
    optimizer.step()
```

这段代码表达的是：

```text
forward:
    polyfem.model() 建模拟问题

differentiable:
    diff.model([model]) 包装 forward model

shape variable:
    vertices 是要优化的 mesh 顶点

objective:
    MaxStress 是 PolyFEM 要对 shape 求导的物理目标

backward:
    value.backward() 触发 ShapeOpt.backward 和 PolyFEM adjoint

optimizer:
    Adam 更新 vertices
```

## 14. 低维参数版本

如果用户真正想优化的是 angle / thickness / height，而不是所有 vertices：

```python
params = torch.tensor(
    [initial_angle, initial_thickness, initial_height],
    dtype=torch.float64,
    requires_grad=True,
)

reference_vertices = diff.shape_vertices(
    diff_model,
    selection=body,
    requires_grad=False,
)


def build_vertices_from_params(params: torch.Tensor) -> torch.Tensor:
    angle, thickness, height = params

    x = reference_vertices.clone()
    x[:, 0] = x[:, 0] * thickness
    x[:, 1] = x[:, 1] * height

    rotation = torch.stack(
        [
            torch.stack([torch.cos(angle), -torch.sin(angle)]),
            torch.stack([torch.sin(angle), torch.cos(angle)]),
        ]
    )
    return x @ rotation.T


def loss(params: torch.Tensor) -> torch.Tensor:
    x = build_vertices_from_params(params)
    return diff.ShapeOpt.apply(
        model=diff_model,
        selection=body,
        tensor=x,
        objective=objective,
    )


optimizer = torch.optim.Adam([params], lr=1e-2)
```

这就是老师说的：

```text
params -> x
```

PolyFEM 不需要知道 angle / thickness / height 的语义。PolyFEM 只需要知道当前 vertices `x`，并返回 objective 对 `x` 的梯度。PyTorch 负责把这个梯度继续传回 `params`。

## 15. 可以怎么跟老师汇报

可以这样说：

```text
I updated the API direction so the forward setup stays in the generated
polyfem.model() layer. The differentiable layer now starts from the completed
forward model with diff.model([model]), not from a raw config variable in the
meeting example.

For the first shape/stress MVP, I think ShapeOpt should be objective-aware:
diff.ShapeOpt.apply(..., objective=diff.MaxStress(...)). This matches your
point that the adjoint path needs to know which physical quantity is being
differentiated.

The params-to-vertices mapping stays in PyTorch. Users can optimize low-level
vertices directly, or optimize high-level parameters such as angle/thickness
and build vertices with torch operations. Then PyTorch handles the chain rule
from vertices back to the user parameters.

I still keep the solution-returning PyTorch-loss route as a future option,
for cases such as MSE against target displacement, but I would not make that
the first stress optimization example.
```

中文大意：

```text
forward 仍然用 polyfem.model()。
diff.model([model]) 只是把 forward model 包进 differentiable context。
第一个 stress/shape MVP 应该让 ShapeOpt 接 objective，比如 MaxStress。
用户如果有 angle/thickness/height，就用 torch 把这些参数变成 vertices。
PolyFEM 负责 objective 对 vertices 的梯度，PyTorch 负责 vertices 对 params 的梯度。
solution-returning 的任意 PyTorch loss 路线可以保留，但不作为第一个 stress demo 的主线。
```

## 16. 最容易混淆的点

### 16.1 `diff.model([model])` 不运行仿真

它只是包装 forward model/state。

真正运行发生在：

```python
diff.ShapeOpt.apply(...)
```

### 16.2 `ShapeOpt` 不是 optimizer

`ShapeOpt` 是 autograd operation。Adam / loop / learning rate 仍然在 PyTorch。

### 16.3 `objective` 不是装饰品

对 stress / von Mises 这种目标，objective 告诉 backend：

```text
这次 backward 要对哪个物理量求导。
```

### 16.4 `target_u` 不是必须的

`target_u` 只用于 target matching，例如 MSE。没有目标 displacement 数据时，不需要它。

### 16.5 `params` 和 `vertices` 可以不是同一个东西

最简单时：

```text
params = vertices
```

更常见的设计优化时：

```text
params = angle/thickness/height
vertices = torch function(params)
```

## 17. 最短版本

只记这一段：

```text
用户先用 polyfem.model() 按 forward example 的方式建模型。
然后用 diff.model([model]) 把 forward model 包成可微分模型。
第一个 shape/stress MVP 应该是 objective-aware ShapeOpt：
ShapeOpt 接收 vertices 和 MaxStress / StressNorm / von Mises 这类 objective，
forward 返回 scalar objective value，backward 调 PolyFEM adjoint 返回 shape gradient。
如果用户优化的是 angle/thickness/height，就用 torch 把 params 变成 vertices，
这样 PyTorch 可以用 chain rule 把梯度从 vertices 继续传回 params。
老师写的 sol norm 或 mse_loss(target_u) 是 solution-returning 路线的可选例子，
可以保留，但不应该作为第一个 stress optimization demo 的主线。
```
