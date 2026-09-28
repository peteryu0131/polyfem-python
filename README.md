# PolyFEM-Python

This README is intentionally minimal while the Python API documentation is being
rewritten.

The current work-in-progress public API direction is:

- generated Python configuration helpers from `polyfem/json-specs/`
- PolyFEM-specific generator config under `generator-config/`
- `polyfempy.runtime.solve(...)` as the forward solve entry point
- generated/model-builder examples from the `examples/` submodule
- `solve(cfg=...)` accepts generated config objects, backend-shaped dicts, or JSON paths

The repository is being split into the Python package plus fixed submodule
checkouts for upstream source, generator, data, and examples:

- `polyfempy/runtime/`: handwritten solve/runtime layer.
- `polyfempy/generated_api/`: packaged generated config authoring API.
- `src/`: C++ extension sources.
- `polyfem/`: `polyfem/polyfem` submodule for PolyFEM backend source and
  canonical JSON specs.
- `python-from-jse/`: `polyfem/python-from-jse` submodule for the generic
  JSON-spec-to-Python generator and dummy examples.
- `generator-config/`: PolyFEM-specific generator config.
- `polyfem-data/`: `polyfem/polyfem-data` submodule for data, meshes, source
  JSON examples, and expected test data.
- `differentiability-data/`: optional `polyfem/differentiability-data`
  submodule for differentiable API examples and gradient regression fixtures.
- `examples/`: `polyfem/python_data` submodule, with `classic_example/` as the
  current generated-API example collection.

For a fresh checkout, clone with submodules:

```powershell
git clone --recurse-submodules https://github.com/polyfem/polyfem-python.git
cd polyfem-python
```

If the repository was already cloned without submodules, initialize them before
generating, building, or running example parity tests:

```powershell
git submodule update --init --recursive
```

The heavier differentiable gradient checks also use the
`differentiability-data/` submodule by default. If you keep that data checkout
somewhere else, set `POLYFEMPY_DIFFDATA_ROOT` to that path instead.

## Current differentiable API status

The new `polyfempy.differentiable_api` is currently a shape differentiable MVP.
The user-facing path is `diff.shape_opt(...)`:

```python
diff_model = diff.model([forward_config])
shape_opt = diff.shape_opt(
    diff_model,
    vertices,
    objective=diff.Objective.STRESS_NORM,
)
loss = shape_opt()
loss.backward()
```

For optimization, configure the same shape problem with a PyTorch optimizer:

```python
result = (
    shape_opt
    .optimizer(torch.optim.Adam, lr=1e-7)
    .steps(100)
    .optimize()
)
```

The reference regression data comes from
`differentiability-data/input/neohookean-stress-3d-opt.json`, which maps to the
current `shape` parameter plus `stress_norm` objective path. The matching
forward-style examples are:

- `differentiable_example/shape/neohookean_stress_3d_opt.py`: one objective
  evaluation plus `loss.backward()`, matching the reference opt spec.
- `differentiable_example/shape/neohookean_stress_3d_optimization.py`: a
  standalone PyTorch `torch.optim.Adam` example with the same forward setup
  and `diff.shape_opt(...)` call.

`diff.solve(state=..., parameters=...)` is the intended generic parameter API
shape, but it is not the completed backend-backed entry point yet. Treat it as
contract scaffolding until its backend session methods are implemented. The
next API expansion target is elastic material parameters from the
`elastic-material-smoothing-2d/3d-opt.json` differentiability-data fixtures.

Generate the packaged API from the repository root:

```powershell
python tools\generate_polyfem_api.py
```

Run the backend-free generation and PolyFEM API parity checks:

```powershell
python tools\generate_polyfem_api.py --check
```

The generator submodule owns its own standalone test suite. The
`polyfem-python` checks only verify that the generated PolyFEM API is produced
and remains compatible with this package's examples/tests.

`tools\generate_polyfem_api.py` reads the schema from the `polyfem/` submodule.
If PolyFEM's schema references solver specs owned by the pinned PolySolve
dependency, the wrapper caches those linked specs under `build/` automatically.
When a PolyFEM build exports the complete embedded/resolved schema, prefer that
single source of truth instead:

```powershell
python tools\generate_polyfem_api.py --resolved-schema-file path\to\polyfem-resolved-spec.json --check
```

Using `--resolved-schema-file` skips automatic PolySolve cache resolution and
generates from the same full schema that PolyFEM's JSON validation uses.

Submodule revisions are compatibility pins. Do not update `polyfem/` by simply
pulling the latest commit unless `python-from-jse`, `generator-config/`, the
regenerated API, and the backend-free checks are updated together.

Older guided API, artifact, experiment, and example documentation has been
removed because it no longer matches the current API direction.
