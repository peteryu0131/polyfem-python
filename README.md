# PolyFEM-Python

This README is intentionally minimal while the Python API documentation is being
rewritten.

The current work-in-progress public API direction is:

- generated Python configuration helpers from `python-from-jse/json-specs/`
- `polyfempy.api.solve(...)` as the forward solve entry point
- generated/model-builder examples under `examples/classic_example/`
- `solve(cfg=...)` accepts generated config objects, backend-shaped dicts, or JSON paths

Older guided API, artifact, experiment, and example documentation has been
removed because it no longer matches the current API direction.
