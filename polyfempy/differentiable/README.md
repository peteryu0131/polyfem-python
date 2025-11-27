# Differentiable Simulations

This module provides differentiable simulation support for PolyFEM, enabling automatic gradient computation through the adjoint method.

## Quick Start

```python
import torch
import numpy as np
from polyfempy.differentiable import solve_differentiable

# Setup mesh
V = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]], dtype=np.float64)
C = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

# Configuration
cfg = {
    "materials": [{"type": "LinearElasticity", "E": 1e6, "nu": 0.3}],
    "boundary_conditions": {
        "dirichlet_boundary": [{"selection": 1, "value": [0.0, 0.0]}],
        "neumann_boundary": [{"selection": 2, "value": [0.0, -1000.0]}],
    },
    "output": {"directory": "out", "paraview": {"file_name": "solution"}}
}

# Make vertices differentiable
vertices = torch.tensor(V, requires_grad=True)

# Run simulation
result = solve_differentiable(vertices, C, cfg)

# Compute loss and gradient
loss = torch.norm(result.u)
loss.backward()
grad = vertices.grad  # Automatic gradient computation!
```

## Features

- **Automatic gradient computation**: No need to manually write `torch.autograd.Function`
- **Unified API**: Uses the same configuration format as `solve()`
- **Multiple derivative types**: Support for shape, material, initial condition derivatives
- **PyTorch integration**: Seamless integration with PyTorch's autograd

## API Reference

### `solve_differentiable()`

Main function for differentiable simulations.

```python
result = solve_differentiable(
    V,                    # Vertices (numpy or torch.Tensor)
    C,                    # Connectivity
    cfg,                  # Configuration (dict or SimulationConfig)
    differentiable_params=["geometry"],  # Which parameters are differentiable
    derivative_type="shape",            # Type of derivative
    backend="nanobind"                  # Must be "nanobind"
)
```

### `PolyFEMFunction`

Low-level PyTorch Function wrapper for advanced use cases.

### `DifferentiableResult`

Result container with PyTorch tensors that support `.backward()`.

## Requirements

- PyTorch (for gradient computation)
- PolyFEM C++ module (nanobind backend)

## Notes

- Differentiable simulations require the `nanobind` backend (not `dummy`)
- The C++ module must be built with differentiable support
- Currently uses the old `pf.Solver()` API internally for direct access to differentiable features

