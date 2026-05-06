# Result Contract

`Result` 是 `solve(...)` 的 structured output contract。

推荐 import：

```python
from polyfempy.api import Result
```

通常用户不会手动构造 `Result`，而是从 solve 得到：

```python
from polyfempy.api import solve

result = solve(cfg=cfg)
```

## 核心字段

常用字段：

| 字段 | 语义 |
| --- | --- |
| `result.vertices` / `result.V` | native mesh vertices。 |
| `result.cells` | native mesh cell blocks。 |
| `result.u` | displacement field，如果 solver 返回。 |
| `result.p` | pressure field，如果存在。 |
| `result.stress` | stress field，如果存在。 |
| `result.strain` | strain field，如果存在。 |
| `result.von_mises` | von Mises field，或从 stress 计算得到。 |
| `result.history` | transient per-step frames，如果 backend 提供。 |
| `result.meta` | field source、solver info、missing fields 等 metadata。 |

## 三个 Field Namespace

`Result` 有三个 field namespace：

| Namespace | 对齐对象 | 说明 |
| --- | --- | --- |
| `point_data` | `result.vertices` | per-vertex/native point data。 |
| `cell_data` | `result.cells` | per-cell/native cell data。 |
| `sampled_data` | sampled/probe mesh | fallback 或 solver 输出的 sampled VTU data，不保证和 native mesh 对齐。 |

查找顺序：

```text
result.field(name)
  -> point_data
  -> cell_data
  -> sampled_data
```

所以：

```python
result.stress
result.von_mises
```

即使来自 fallback sampled data，也可能返回非 `None`。如果需要判断来源，应看：

```python
result.meta
```

例如 `stress_source` / `von_mises_source`。

如果需要明确只查某一个 namespace，不希望 fallback，可以用：

```python
result.point_field("u")
result.cell_field("material_id")
result.sampled_field("von_mises")
```

查看当前有哪些 fields：

```python
result.field_names()
result.available_fields()
```

`field_names()` 返回所有 namespace 合并后的 field 名字；`available_fields()`
按 `point_data` / `cell_data` / `sampled_data` 分组，适合写 report 或 debug。

## Native Data vs Sampled Data

`point_data` 和 `cell_data` 是 native mesh-aligned data。它们可以安全写进
`to_meshio()` / `write(...)` 输出。

`sampled_data` 是另一张 sampled/probe mesh 上的数据。它可以用于：

- visualization summary；
- statistics；
- diagnostics；
- per-body sampled stress/von Mises inspection。

但不能假设：

```text
sampled_data rows == result.vertices rows
```

因此 `to_meshio()` 不会把 `sampled_data` 附到 native mesh 上。这是为了避免把
数据写到错误的 mesh topology 上。

## Field Mutators

native mesh-aligned data：

```python
result.set_field("stress", stress)
```

规则：

- length 等于 `n_cells` 且不等于 `n_vertices` 时，存入 `cell_data`；
- 其他情况默认存入 `point_data`。

sampled/probe data：

```python
result.set_sampled_field("stress", sampled_stress)
```

这个永远写入 `sampled_data`。

## Von Mises 语义

`result.von_mises` 调用 `get_von_mises_numpy()`。

优先级：

1. 如果已有 `von_mises` field，直接返回；
2. 如果已有 `von_mises_avg` field，直接返回；
3. 如果有 `stress`，从 Voigt stress 计算；
4. 否则返回 `None`。

支持 stress shape：

```text
(n, 6): [sxx, syy, szz, sxy, syz, szx]
(n, 3): [sxx, syy, sxy]
```

## History Contract

`result.history` 是 `HistoryView`。如果 backend 提供 transient frames，常用形状是：

```text
history.u:      (n_steps, n_sampled, dim)
history.vm:     (n_steps, n_sampled)
history.vm_avg: (n_steps, n_sampled)
history.stress: (n_steps, n_sampled, tensor_width)
history.times:  (n_steps,)
```

`result.body_ids` 优先从 `result.field("body_ids")` 找；如果没有，会尝试使用
`result.history.body_ids`。

## Per-Body Field Split

如果 sampled fallback 提供了 `body_ids`，可以按 body 拆 field：

```python
stress_by_body = result.field_by_body("stress")
for body_id, stress in stress_by_body.items():
    print(body_id, stress.shape)
```

限制：

- `body_ids` 必须存在；
- field rows 必须和 `body_ids` rows 一致；
- 当前这通常用于 sampled mesh fields，不用于 native `u`。

## Mesh I/O

写文件：

```python
result.write("result.vtu")
```

需要 `meshio`。写出的 mesh 包含：

- `vertices`
- `cells`
- mesh-aligned `point_data`
- mesh-aligned `cell_data`

不会写入 `sampled_data`，因为 sampled data 不在 native mesh 上。

读文件：

```python
result = Result.read("result.vtu")
```

`from_meshio(...)` 会把 meshio point/cell data 转成 `point_data` /
`cell_data`。

## ML Conversion

```python
result.to_torch(include_mesh=True)
```

这会把已存字段转换成 Torch tensor，但不会让普通 forward solve 变成
autograd-backed solve。需要可微求解时，应该使用：

```python
from polyfempy.differentiable import prepare_differentiable_simulation
```

或 optimization helpers。

## 不应该破坏的语义

后续 cleanup 必须保护：

- `field(...)` 查找顺序：point -> cell -> sampled；
- `sampled_data` 不写入 native mesh output；
- `von_mises` 从已有 field 或 stress 计算的优先级；
- `history` 的 per-step shape；
- `field_by_body(...)` 对 body_ids/field rows 的一致性检查；
- `Result` 能作为 `solve(...)` 的稳定返回对象。

## 推荐测试

改 `Result` 或 result reporting 后，至少跑：

```bash
python -m pytest \
  tests/test_result_history.py \
  tests/test_result_meshio_roundtrip.py \
  tests/test_result_report.py \
  tests/test_result_sampled_data.py \
  tests/test_pipeline_extract_outputs.py \
  tests/test_pipeline_sampled_fallback.py
```
