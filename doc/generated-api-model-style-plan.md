# Generated Examples Model Style Status

## 目标

classic generated examples 尽量使用用户更容易读的写法：

```python
model = polyfem.model()

body = model.mesh(...)
body.material(...)

surface = body.surface_axis(...)
surface.dirichlet(...)

polyfem_config = model.config(...)
```

这个方向是合理的，因为 `polyfem.model()` 本身也是 generated API 的一部分。它不是 example-only helper。它把 JSON 里的 id 关系转成更自然的 API 调用：先创建 geometry/body，再绑定 material / initial condition / space order，再创建 surface 或 point selection，最后绑定 boundary condition。

但不应该手动改每个 generated example。正确维护方式是：

- generator 能无损表达时，自动生成 model style；
- generator 不能保证 source JSON parity 时，保留 direct `polyfem.config(...)` 或 `SourcePayloadConfig(...)` fallback；
- 重新运行 generator 后，examples 仍然保持同样风格，不依赖手改 generated files。

## 当前结果

当前 `examples/classic_example` 里共有 92 个 generated examples：

| Style | Count |
| --- | ---: |
| `model = polyfem.model()` | 91 |
| fallback | 1 |

唯一没有使用 model style 的 example：

- `examples/classic_example/3D/contact_3d_static_two_cubes_generated_api.py`

它对应：

- `polyfem-data/contact/examples/3D/static/two-cubes.json`

这个文件当前不是 model-style renderer 的主要问题，而是 generated API/schema coverage 问题：source JSON 里有 `solver.nonlinear.use_grad_norm`，但当前 generated API 的 `Root.Solver.Nonlinear.__init__()` 不接受这个 keyword。

验证过：

- 原始 payload 调用 `polyfem.config(**payload)` 会失败：
  - `Root.Solver.Nonlinear.__init__() got an unexpected keyword argument 'use_grad_norm'`
- 删除 `solver.nonlinear.use_grad_norm` 后，同一个 payload 可以被 `polyfem.config(...)` 构造。

所以这个剩余 fallback 应该等 polysolve JSON spec 合入 generator 输入后再自动解决，不应该在 generated examples 里 hard-code workaround。

## 已完成的 model-style 扩展

### 1. Obstacle mesh

已支持：

```python
model.obstacle_mesh(
    mesh=...,
    surface_selection=0,
)
```

输出 JSON：

```json
{
  "type": "mesh",
  "mesh": "...",
  "surface_selection": 0,
  "is_obstacle": true
}
```

重点：

- obstacle mesh 不创建 `volume_selection`；
- `surface_selection=0` / `point_selection=0` 会原样保留；
- positive `surface_selection` 可以返回 `SelectionHandle`，用于绑定 obstacle displacement。

### 2. mesh_array

已支持：

```python
model.mesh_array(...)
model.obstacle_mesh_array(...)
```

如果 `mesh_array` 有正整数 `volume_selection`，它返回 `BodyHandle`，可以继续使用：

- `body.material(...)`
- `body.velocity(...)`
- `body.solution(...)`
- `body.acceleration(...)`
- `body.discr_order(...)`

如果没有正整数 `volume_selection`，它只作为 geometry item 加入 model，不会自动创建 volume id。

### 3. no-volume normal geometry

已支持：

```python
geometry = model.geometry_mesh(...)
surface = geometry.surface_all(id=1)
surface.dirichlet(...)
```

这个用于 source JSON 里 normal mesh 没有 `volume_selection` 的情况。

为什么不能直接用 `model.mesh(...)`：

- `model.mesh(...)` 的语义是创建 body；
- 如果 source JSON 没有 `volume_selection`，`model.mesh(...)` 会自动分配一个新的 volume id；
- 这样 generated payload 会比 source JSON 多字段，破坏 source parity。

所以新增 `geometry_mesh(...)` 表示“只加入 geometry，不创建 body id”。这是 generic builder 能力，不是某个 example 的 hard-code。

### 4. obstacle surface displacement

已支持 obstacle 上的 positive `surface_selection` 绑定 `obstacle_displacements`：

```python
roller_surface = model.obstacle_mesh(
    mesh=...,
    surface_selection=1000,
)
roller_surface.obstacle_displacement(value=[0, "-t", 0])
```

输出 JSON：

```json
{
  "geometry": [
    {
      "type": "mesh",
      "mesh": "...",
      "surface_selection": 1000,
      "is_obstacle": true
    }
  ],
  "boundary_conditions": {
    "obstacle_displacements": [
      {
        "id": 1000,
        "value": [0, "-t", 0]
      }
    ]
  }
}
```

这个能力让 `3D/friction/ball-rollers.json`、`armadillo-roller.json`、`microstructure.json`、`squeeze-out.json`、`trash-compactor-*` 等 examples 自动转成 model style。

### 5. single object selection

已支持 source JSON 里的单个 selection object：

```json
"surface_selection": {
  "id": 1,
  "axis": "-y",
  "position": 1e-05,
  "relative": true
}
```

生成写法：

```python
surface = body.surface_axis(
    id=1,
    axis='-y',
    position=1e-05,
    relative=True,
    append=False,
)
```

这里 `append=False` 的意思是保持 source JSON 的 single-object shape。如果不写 `append=False`，默认仍然 append 到 list，保持原有 helper 语义。

这个能力让 `3D/unit-tests/2-cubes.json` 也能 model-style，同时保持 source JSON parity。

### 6. mesh_sequence

已支持：

```python
model.geometry_mesh_sequence(...)
model.obstacle_mesh_sequence(...)
```

这用于 `type == "mesh_sequence"` 的 geometry item。例如：

```python
model.obstacle_mesh_sequence(
    mesh_sequence='../../../meshes/3D/obstacles/kick-sequence/',
    fps=24,
)
```

这个能力让 `3D/mesh-sequence/kick.json` 自动转成 model style。

## 这次改了哪里

### `python-from-jse/generator/model_builder.py`

扩展 generic `ModelBuilder`：

- `mesh_array(...)`
- `geometry_mesh(...)`
- `geometry_mesh_array(...)`
- `geometry_mesh_sequence(...)`
- `obstacle_mesh(...)`
- `obstacle_mesh_array(...)`
- `obstacle_mesh_sequence(...)`
- obstacle positive `surface_selection` 返回 `SelectionHandle`
- selection helper 支持 `append=False`，用于保持 single-object selection shape
- selection helper rule 从 generated manifest 里保留 `class_path`，需要 single-object shape 时可以构造对应 generated class object

这些都是 generic builder runtime 能力，不是针对某个 example 的 hard-code。

### `tools/generate_classic_contact_examples.py`

扩展 `render_builder_config(...)`：

- 允许 `type == "mesh"` / `"mesh_array"` / `"mesh_sequence"`；
- 允许 `is_obstacle=True` geometry；
- 允许无 `volume_selection` 的 normal geometry 用 `geometry_mesh(...)` 表达；
- 允许 obstacle positive `surface_selection` 绑定 `obstacle_displacements`；
- 允许 single dict selection 生成 `append=False` helper；
- 保留 `surface_selection=0` / `point_selection=0`，避免丢字段。

### `tests/test_generated_api_example.py`

新增/扩展 tests：

- builder-safe examples 必须使用 `model = polyfem.model()`；
- all generated examples 必须和 `polyfem-data` source JSON 保持 parity；
- obstacle mesh 不创建 volume selection；
- no-volume `geometry_mesh(...)` 不创建 volume selection；
- mesh_array 保留 source payload；
- obstacle surface displacement 可绑定；
- single dict selection 保持 object shape；
- mesh_sequence 保持 payload。

### `python-from-jse/tests/test_model_builder.py`

新增 generic builder tests：

- no-volume geometry 可以创建 surface selection；
- selection helper 可以用 `append=False` 保持 single-object shape；
- selection helper rules 从 manifest 保留 `class_path`。

## id relationship 为什么重要

`generator-config/id_relationships.json` 是 model API 自动可维护的核心。

例如：

```json
{
  "namespace": "volume",
  "producer": "geometry[*].volume_selection",
  "consumer": "materials[*].id",
  "builder_api": "body.material",
  "status": "required"
}
```

意思是：

- `geometry[*].volume_selection` 生产一个 volume id；
- `materials[*].id` 消费这个 volume id；
- 用户 API 可以写成 `body.material(...)`。

再例如：

```json
{
  "namespace": "surface",
  "producer": "geometry[*].surface_selection",
  "consumer": "boundary_conditions.obstacle_displacements[*].id",
  "builder_api": "surface.obstacle_displacement",
  "status": "required"
}
```

意思是：

- `geometry[*].surface_selection` 生产 surface id；
- `boundary_conditions.obstacle_displacements[*].id` 消费这个 id；
- 用户 API 可以写成 `surface.obstacle_displacement(...)`。

这次 obstacle displacement 能自动转 model style，就是因为已有 id relationship 可以表达这个关系，只需要让 model builder 在 obstacle positive `surface_selection` 时返回 `SelectionHandle`。

## 当前验证结果

已跑：

```powershell
python tools\generate_classic_contact_examples.py
python -m pytest python-from-jse\tests\test_model_builder.py -q
python -m pytest tests\test_generated_api_example.py tests\test_generated_example_backend_tool.py tests\test_generated_payload.py tests\test_pipeline_normalize.py -q
python -m py_compile tools\generate_classic_contact_examples.py python-from-jse\generator\model_builder.py
```

结果：

```text
Generated 86 classic contact example(s); skipped 6 hand-written file(s).
25 passed in 0.06s
96 passed in 4.64s
py_compile passed
```

还跑了 representative backend comparisons：

```powershell
conda activate polyfem
python tools\check_generated_example_backend.py --example examples\classic_example\3D\contact_3d_friction_ball_rollers_generated_api.py --source-json polyfem-data\contact\examples\3D\friction\ball-rollers.json --output-root build\model-style-obstacle-displacement-ball-rollers-check-20260813 --generated-source-tolerance 1e-5 --require-tests-match
```

结果：

- generated output vs source JSON output: PASS
- generated output vs source JSON tests: PASS

```powershell
conda activate polyfem
python tools\check_generated_example_backend.py --example examples\classic_example\3D\contact_3d_unit_tests_2_cubes_generated_api.py --source-json polyfem-data\contact\examples\3D\unit-tests\2-cubes.json --output-root build\model-style-single-dict-selection-2-cubes-check-20260813 --generated-source-tolerance 1e-5 --require-tests-match
```

结果：

- generated output vs source JSON output: PASS
- source JSON has no tests block, so tests comparison skipped

```powershell
conda activate polyfem
python tools\check_generated_example_backend.py --example examples\classic_example\3D\contact_3d_mesh_sequence_kick_generated_api.py --source-json polyfem-data\contact\examples\3D\mesh-sequence\kick.json --output-root build\model-style-mesh-sequence-kick-check-20260813 --generated-source-tolerance 1e-5
```

结果：

- generated output vs source JSON output: PASS
- source JSON has no tests block, so tests comparison skipped

还检查了 `2D/static/friction-slope.json`：

- generated payload 和 source payload 保持 parity；
- 但 backend generated run reached iteration limit；
- 手动跑 source reduced JSON 也同样 reached iteration limit；
- 所以这不是 model-style regression，不应该算作 generated API payload 错误。

## 后续建议

当前不建议为了最后一个 `3D/static/two-cubes.json` 写 workaround。

建议下一步：

1. 把 polysolve 的 JSON spec 正式并入 generated API 输入。
2. 重新生成 generated API。
3. 确认 `Root.Solver.Nonlinear` 是否接受 `use_grad_norm`。
4. 再运行：

```powershell
python tools\generate_classic_contact_examples.py
python -m pytest tests\test_generated_api_example.py -q
```

如果 `use_grad_norm` 被 schema/API 支持，`3D/static/two-cubes.json` 应该可以继续走 model-style renderer。否则它仍然应该保留 `SourcePayloadConfig(...)` fallback，因为 source JSON shape 还不是 generated API 可表达的 shape。

当前可以对老师汇报的核心结论：

- generated examples 已经从 79/92 model style 提升到 91/92；
- 新增支持不是手写每个 example，而是扩展 generator 和 generic `ModelBuilder`；
- source JSON parity 单元测试通过；
- representative backend comparisons 通过；
- 唯一剩余 fallback 是 `solver.nonlinear.use_grad_norm` 的 generated API/schema coverage 问题，不是 model-style renderer 的问题。
