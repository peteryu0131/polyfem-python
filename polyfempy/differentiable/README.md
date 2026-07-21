# Differentiable Reference Code

This directory contains the previous differentiable API implementation.

It is kept as experimental reference code and is not part of the current supported polyfempy.runtime interface.
The supported forward-solve path is:

```text
generated API configuration
  -> polyfempy.runtime.solve(...)
  -> C++ PolyFEM backend
  -> polyfempy.runtime.Result
```

Some modules in this directory may still refer to removed APIs such as
`SimulationConfig`. Future differentiable work should rebuild this layer on top
of the generated JSON/model-builder API rather than restoring the old
handwritten configuration API.
