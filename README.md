# PolyFEM-Python

This README is intentionally minimal while the Python API documentation is being
rewritten.

The current work-in-progress public API direction is:

- generated Python configuration helpers from `python-from-jse/json-specs/`
- PolyFEM-specific generator config under `generator-config/`
- `polyfempy.api.solve(...)` as the forward solve entry point
- generated/model-builder examples under `examples/classic_example/`
- `solve(cfg=...)` accepts generated config objects, backend-shaped dicts, or JSON paths

The repository currently keeps the generator, data, and examples in one working
tree. The intended split is:

- `polyfempy/` and `src/`: Python binding package and C++ extension.
- `python-from-jse/`: generic JSON-spec-to-Python generator.
- `generator-config/`: PolyFEM-specific generator config.
- `polyfem-data/`: data, meshes, source JSON examples, and expected test data.
- `examples/`: generated-API examples, with `classic_example/` as the current
  example collection.

Older guided API, artifact, experiment, and example documentation has been
removed because it no longer matches the current API direction.
