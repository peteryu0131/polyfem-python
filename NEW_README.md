# PolyFEM Python API

A clean, neutral Python API for PolyFEM simulations that works today with a Dummy backend and will seamlessly connect to a nanobind C++ backend.

## Features

- **Simplified API**: Minimal interface with `solve(V, C, cfg, callbacks, backend)`
- **Type-safe configuration**: `SimulationConfig` dataclass with validation
- **Multiple backends**: Dummy (available now), nanobind (coming soon)
- **Error isolation**: Batch processing with proper error handling
- **Deterministic output**: Seed-based reproducibility

## Installation

### Basic Installation

Install in development mode:

```bash
pip install -e .
```

Or install for use:

```bash
pip install .
```

### Differentiable Simulation Support (Optional)

For gradient-based optimization and PyTorch integration, install with the `differentiable` extra:

```bash
pip install -e .[differentiable]
```

Or install PyTorch directly:

```bash
pip install torch>=1.9.0
```

**Note**: PyTorch is only required if you need differentiable simulations (shape optimization, material optimization, etc.). Most users don't need this.

See [Differentiable Guide](docs/differentiable-guide.md) for more details.

## Quick Start

### Basic Usage

```python
import numpy as np
from polyfempy.api import solve, SimulationConfig

# Create a simple 2D mesh (unit square with 2 triangles)
V = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
C = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

# Configure the simulation
cfg = SimulationConfig(
    pde="linear_elasticity",
    max_iters=10,
    random_seed=42
)

# Solve
result = solve(V, C, cfg, backend="dummy")

# Examine results
print(result.summary())
print(f"Solution shape: {result.u.shape}")
print(f"Meta: {result.meta}")
```

### Configuration

```python
from polyfempy.api import SimulationConfig

# Default configuration
cfg = SimulationConfig()

# Custom configuration
cfg = SimulationConfig(
    pde="linear_elasticity",      # or "poisson"
    discr_order=2,                  # polynomial order
    stiffness=1.5,                 # stiffness parameter
    max_iters=50,                  # max solver iterations
    random_seed=42,                # for reproducibility
    materials={"E": 2100, "nu": 0.3},  # material properties
    bc={},                          # boundary conditions
    time={},                        # time-related params
    extras={}                        # backend-specific options
)

# Convert to dict
settings = cfg.to_dict()
```

### With Callbacks

```python
def before_solve(meta):
    print("Starting solve...")

def after_iter(iter_num, residual, meta):
    print(f"Iteration {iter_num}: residual = {residual}")

def after_solve(meta):
    print("Solve complete!")

callbacks = {
    "before_solve": before_solve,
    "after_iter": after_iter,
    "after_solve": after_solve,
}

result = solve(V, C, cfg, callbacks=callbacks)
```

### Batch Processing

```python
from polyfempy.api import batch_solve

# Multiple jobs
jobs = [
    (V1, C1, cfg1, None),
    (V2, C2, cfg2, None),
    (V3, C3, cfg3, None),
]

results = batch_solve(jobs)

# Results maintain order, errors are isolated
for i, res in enumerate(results):
    if isinstance(res, Exception):
        print(f"Job {i} failed: {res}")
    else:
        print(f"Job {i} succeeded: {res.meta}")
```

### Differentiable Simulations (Optional)

For gradient-based optimization, use the differentiable module:

```python
from polyfempy.differentiable import solve_differentiable
import torch
import numpy as np

# Prepare data
V = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]], dtype=np.float64)
C = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

cfg = {
    "materials": [{"type": "LinearElasticity", "E": 1e6, "nu": 0.3}],
    "boundary_conditions": {
        "dirichlet_boundary": [{"selection": 1, "value": [0.0, 0.0]}],
        "neumann_boundary": [{"selection": 2, "value": [0.0, -1000.0]}],
    }
}

# Use differentiable solver (5 lines!)
vertices = torch.tensor(V, requires_grad=True)
result = solve_differentiable(vertices, C, cfg)
loss = torch.norm(result.u)
loss.backward()
grad = vertices.grad  # Automatic gradient computation!
```

See [Differentiable Guide](docs/differentiable-guide.md) for more details.

## Examples

Run the Dummy backend example:

```bash
python -m polyfempy.api.examples.run_dummy_elasticity
```

This demonstrates:
- Creating a mesh (unit square with 2 triangles)
- Configuring simulation parameters
- Running the solver with Dummy backend
- Displaying results and metadata

## Backend Status

### Dummy Backend (Available Now)

The Dummy backend provides:
- ✅ Full API contract compliance
- ✅ Deterministic pseudo-random output
- ✅ Callback lifecycle support
- ✅ Input validation with clear errors
- ✅ Supports 2D (triangles, quads) and 3D (tets, hexes) meshes

Use with: `backend="dummy"` (default)

### Nanobind Backend (Coming Soon)

The Nanobind backend will connect to the C++ PolyFEM solver:

- ⏳ Currently: `backend="nanobind"` raises `NotImplementedError`
- 🔨 Requires: Building the C++ module `polyfem_nb`
- 📚 See: `docs/for_cpp_dev.md` for C++ implementation guide

When available, simply switch `backend="nanobind"` to use the real solver.

## API Reference

### Main Functions

- **`solve(V, C, cfg, callbacks=None, backend="dummy")`**: Main solver entrypoint
- **`batch_solve(jobs, backend="dummy")`**: Batch processing with error isolation
- **`SimulationConfig`**: Configuration dataclass with `to_dict()`
- **`Result`**: Result container with `u`, `strain`, `stress`, `meta`

### Error Types

- **`ValueError`**: Invalid input (with `INPUT:` prefix)
- **`TypeError`**: Invalid callback return (with `CALLBACK:` prefix)
- **`RuntimeError`**: Backend failure (with `BACKEND:` prefix)

## Testing

Run all tests:

```bash
pytest tests/ -q
```

Test categories:
- `test_dummy_contract.py`: API contract validation
- `test_determinism.py`: Reproducibility tests
- `test_backend_switch.py`: Backend switching tests
- `test_batch_isolation.py`: Batch processing isolation tests
- `test_contact_examples_api.py`: Compatibility tests with polyfem-data examples
- `test_common_json_merge.py`: common.json merge functionality tests

## Compatibility

### PolyFEM Data Examples

✅ **Full compatibility** with [polyfem-data/contact/examples](https://github.com/polyfem/polyfem-data/tree/main/contact/examples):

- **86/86 examples** (28 2D + 58 3D) can be loaded and configured
- **All parameters supported**: geometry, materials, boundary_conditions, time, contact, solver, output, space
- **common.json support**: Automatic deep merge of referenced common.json files
- **Transient problems**: Support for time-stepping configurations

See [API Compatibility Documentation](docs/api-contact-examples-compatibility.md) for details.

### Loading PolyFEM JSON Configurations

The API supports loading full PolyFEM JSON configurations:

```python
from polyfempy.api import SimulationConfig, solve

# Load from file (automatically handles common.json references)
cfg = SimulationConfig.from_json_file("data/contact/examples/2D/unit-tests/5-squares.json")

# Or load from dict
import json
with open("config.json") as f:
    config_dict = json.load(f)
cfg = SimulationConfig.from_json_dict(config_dict)

# Solve (for static problems)
result = solve(cfg=cfg)
```

## Documentation

- **[API Architecture](docs/api-architecture.md)**: Complete API design and architecture
- **[API Compatibility](docs/api-contact-examples-compatibility.md)**: Compatibility with polyfem-data examples
- **[Differentiable Guide](docs/differentiable-guide.md)**: Gradient-based optimization and PyTorch integration
- **[Legacy vs New API](docs/legacy-vs-new-api-comparison.md)**: Comparison between old and new APIs
- **[API Completion Assessment](docs/api-completion-assessment.md)**: Feature completion status

## Development

### Requirements

- Python >= 3.9
- NumPy
- pytest (for testing)
- PyTorch >= 1.9.0 (optional, for differentiable simulations)

### Project Structure

```
polyfempy/
  api/                    # Main API package
    ├── config.py         # SimulationConfig
    ├── result.py         # Result container
    ├── errors.py          # Unified error model
    ├── backend_base.py   # Backend SPI definition
    ├── backend_dummy.py  # Dummy implementation
    ├── backend_nanobind.py  # Nanobind adapter
    ├── solve.py          # Main entrypoint
    └── batch.py          # Batch processing
  examples/               # Example scripts
    └── run_dummy_elasticity.py
  test/                   # Legacy tests
  tests/                  # New contract tests
    ├── test_dummy_contract.py
    ├── test_determinism.py
    ├── test_backend_switch.py
    ├── test_batch_isolation.py
    ├── test_contact_examples_api.py
    └── test_common_json_merge.py
docs/
  ├── binding-spec.md    # API specification
  ├── for_cpp_dev.md     # C++ implementation guide
  └── TODO.md            # Development roadmap
```

## Design Principles

1. **Neutral API**: No dependency on specific backends until runtime
2. **Strict validation**: Inputs are checked before backend dispatch
3. **Error isolation**: Bad tasks don't break good ones in batch mode
4. **Determinism**: Same inputs → same outputs (seed-based)
5. **Clear errors**: All errors have informative prefixes
6. **Future-proof**: Ready to connect C++ backend without API changes

## License

[Check LICENSE file for details]

## Contact

[Add contact information if needed]


