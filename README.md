# PolyFEM-Python

This README is intentionally minimal while the Python API documentation is being
rewritten.

The current work-in-progress public API direction is:

- generated Python configuration helpers from `external/polyfem/json-specs/`
- PolyFEM-specific generator config under `generator-config/`
- `polyfempy.runtime.solve(...)` as the forward solve entry point
- generated/model-builder examples under `examples/classic_example/`
- `solve(cfg=...)` accepts generated config objects, backend-shaped dicts, or JSON paths

The repository currently keeps the generator, data, and examples in one working
tree. The intended split is:

- `polyfempy/runtime/`: handwritten solve/runtime layer.
- `polyfempy/generated_api/`: packaged generated config authoring API.
- `src/`: C++ extension sources.
- `external/polyfem/`: PolyFEM backend source and canonical JSON specs.
- `python-from-jse/`: generic JSON-spec-to-Python generator and dummy examples.
- `generator-config/`: PolyFEM-specific generator config.
- `polyfem-data/`: data, meshes, source JSON examples, and expected test data.
- `examples/`: generated-API examples, with `classic_example/` as the current
  example collection.

For a fresh checkout, initialize submodules before generating or building:

```powershell
git submodule update --init --recursive
```

Generate the packaged API from the repository root:

```powershell
python tools\generate_polyfem_api.py
```

Run the backend-free generation and parity checks:

```powershell
python tools\generate_polyfem_api.py --check
```

PolyFEM's raw JSON spec may reference linked solver specs that are provided by
the C++ dependency/spec setup rather than `external/polyfem/json-specs/` itself.
When those files are available in a local directory, pass it explicitly:

```powershell
python tools\generate_polyfem_api.py --include-spec-dir path\to\linked-specs --check
```

Older guided API, artifact, experiment, and example documentation has been
removed because it no longer matches the current API direction.
