# Differentiable Examples

This directory contains runnable examples demonstrating how to use the differentiable API.

## Examples

### 1. `simple_shape_optimization.py`
**Simple shape optimization using the new API**

Demonstrates the simplest use case:
- Basic shape optimization
- Comparison with old API
- Shows 75% code reduction

**Run:**
```bash
python -m polyfempy.differentiable.examples.simple_shape_optimization
```

### 2. `multi_solver_shape_optimization.py`
**Multi-solver shape optimization**

Shows how to:
- Combine multiple simulations
- Compute combined gradients
- Verify gradients using finite differences

**Run:**
```bash
python -m polyfempy.differentiable.examples.multi_solver_shape_optimization
```

## Migration from Old Code

The old differentiable code in `test/` directory has been migrated here:

- `test/diffSimulator.py` → `simple_shape_optimization.py` (updated to use new API)
- `test/test_diff.py` → `multi_solver_shape_optimization.py` (updated to use new API)
- `test/test_differentiable.ipynb` → Can be converted to a Python script

## Key Improvements

### Old Way (20+ lines)
```python
class Simulate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, solver, vertices):
        solver.mesh().set_vertices(vertices)
        solver.set_cache_level(pf.CacheLevel.Derivatives)  # Easy to forget!
        solver.solve()
        return torch.tensor(solver.get_solutions())
    # ... backward method ...
```

### New Way (5 lines)
```python
from polyfempy.differentiable import solve_differentiable

vertices = torch.tensor(V, requires_grad=True)
result = solve_differentiable(vertices, C, cfg)
loss = torch.norm(result.u)
loss.backward()
grad = vertices.grad  # Automatic!
```

## Requirements

- PyTorch
- PolyFEM C++ module (nanobind backend)
- NumPy

