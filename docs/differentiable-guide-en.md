# Differentiable Feature Complete Guide

This document provides a comprehensive guide to the differentiable simulation functionality in PolyFEM Python API, including design philosophy, usage methods, technical details, and best practices.

> Note (Route A): Python should always `import polyfempy as pf` (stable C++ extension module name).
> nanobind vs pybind11 is a build-time choice and must not affect imports. The C++ binding’s
> `Solver.solve()` returns `(sol, pressure)`, so Python callers must capture and parse it.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Overview](#overview)
3. [Technical Principles](#technical-principles)
4. [API Design](#api-design)
5. [New Implementation vs Old Implementation](#new-implementation-vs-old-implementation)
6. [Supported Derivative Types](#supported-derivative-types)
7. [Complete Examples](#complete-examples)
8. [Advanced Usage](#advanced-usage)
9. [Implementation Details](#implementation-details)

---

## Quick Start

### Simplest Example (5 Lines of Code)

```python
from polyfempy.differentiable import solve_differentiable
import torch
import numpy as np

# Prepare mesh and configuration
V = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]], dtype=np.float64)
C = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
cfg = {
    "materials": [{"type": "LinearElasticity", "E": 1e6, "nu": 0.3}],
    "boundary_conditions": {
        "dirichlet_boundary": [{"selection": 1, "value": [0.0, 0.0]}],
        "neumann_boundary": [{"selection": 2, "value": [0.0, -1000.0]}],
    }
}

# Run differentiable simulation (only 5 lines!)
vertices = torch.tensor(V, requires_grad=True)
result = solve_differentiable(vertices, C, cfg)
loss = torch.norm(result.u)
loss.backward()
grad = vertices.grad   # Gradient is computed automatically!
```

**It's that simple!** No need to manually configure solver, no need to manually set cache, no need to manually call adjoint. All details are automatically handled by `solve_differentiable()`.

### Config input (aligned with solve())

`solve_differentiable()` accepts the same `cfg` styles as the main `solve()` API:

**Option 1: JSON path** (when you have a config file and mesh files)

```python
result = solve_differentiable(cfg="path/to/config.json", derivative_type="shape")
# When V, C are omitted, mesh is loaded from config's geometry via load_mesh_from_settings();
# root_path is set to the JSON file's directory.
```

**Option 2: API classes** (building config in code)

```python
from polyfempy.api import SimulationConfig, Geometry, GeometryMesh
cfg = SimulationConfig(
    pde="LinearElasticity",
    geometry=Geometry(meshes=[GeometryMesh(mesh="square.obj", surface_selection=[...])]),
    materials=[...],
    boundary_conditions={...},
)
cfg.extras["_root_path"] = str(data_dir)   # or pass root_path=str(data_dir)
result = solve_differentiable(cfg=cfg, derivative_type="shape")
# After backward, gradients are in result.vertices.grad
```

**Minimal runnable example**: `examples/differentiable_minimal.py` runs both options and prints forward/backward results; C++ log level is set to off for minimal output.

---

## Overview

### What is Differentiable?

Differentiable functionality allows you to compute gradients of simulation results with respect to input parameters, which is crucial for the following scenarios:

- **Shape Optimization**: Optimize structural shape to minimize stress or displacement
- **Material Parameter Optimization**: Find optimal material parameters
- **Inverse Problem Solving**: Infer parameters from observed data
- **Machine Learning**: Integrate physical simulation into neural network training

**Typical Use Case**:
```python
# Goal: Find optimal vertices that minimize displacement
def objective(vertices):
    result = solve_differentiable(vertices, C, cfg)  # Differentiable simulation
    return torch.norm(result.u)  # Compute loss

# Need to compute gradient: d(objective)/d(vertices)
# This enables gradient descent optimization
```

### Design Principles

1. **Default Simplicity**: `solve()` API remains simple, no differentiable involved
2. **Optional Enhancement**: Provides independent differentiable module
3. **Progressive Complexity**: From simple to advanced, users choose as needed
4. **Backward Compatibility**: Existing complex usage still supported

### Why Can't We Use PyTorch's Automatic Differentiation Directly?

**PDE solvers are too large, automatic differentiation is not suitable:**

- Millions of degrees of freedom: Storing all intermediate computations requires huge memory
- Iterative solving: Nonlinear solvers require multiple iterations, automatic differentiation records every step
- Efficiency issues: Automatic differentiation time complexity is typically 3-5 times that of forward computation

### Solution: Adjoint Method

**Adjoint method only needs one backward solve, highly efficient:**

1. **Forward Pass**: Run one simulation, solve PDE
2. **Backward Pass**: Solve one adjoint equation, directly get gradient
3. **Memory Efficiency**: Only need to store necessary intermediate results, not all computations

This is why we need `torch.autograd.Function` to customize backward propagation!

---

## Technical Principles

### Core Idea: Layered Design + Separation of Concerns

```
User Layer (Simple and Easy)
    ↓
solve_differentiable() - High-level Wrapper
    ↓
PolyFEMFunction - PyTorch Integration Layer (Custom Backward)
    ↓
Low-level C++ API (pf.Solver) - Actual Computation
```

### Why Must We Use `torch.autograd.Function`?

`torch.autograd.Function` allows us to:

1. **Customize Forward Pass**: Run PolyFEM simulation
2. **Customize Backward Pass**: Use adjoint method to compute gradients (instead of automatic differentiation)
3. **Integrate with PyTorch**: Make PolyFEM simulation part of PyTorch computation graph

**Key Points**:
- Must inherit from `torch.autograd.Function`
- Must implement `forward()` and `backward()` static methods
- Number of gradients returned by `backward()` must match number of inputs to `forward()`

### Module Architecture

#### 1. `__init__.py` - Module Entry and Dependency Management

**Responsibilities**:
- ✅ Dependency checking: Check if PyTorch is installed
- ✅ Unified exports: Provide clear API interface
- ✅ Graceful degradation: If PyTorch is not available, provide clear error message

#### 2. `solve.py` - High-level Wrapper (Main User Interface)

**Responsibilities**:
- ✅ Unified interface: Provide API style consistent with `solve()`
- ✅ Input processing: Automatically handle numpy/torch type conversion
- ✅ Configuration management: Uniformly handle SimulationConfig and dict
- ✅ Solver lifecycle management: Create, configure, run, cleanup

#### 3. `torch_integration.py` - PyTorch Integration Layer

**Responsibilities**:
- ✅ Wrap PyTorch Function: Implement forward/backward
- ✅ Automatic cache handling: Automatically set CacheLevel.Derivatives
- ✅ Automatic adjoint call: Automatically call solve_adjoint
- ✅ Automatic derivative computation: Automatically call derivative functions

#### 4. `result.py` - Result Container

**Responsibilities**:
- ✅ Wrap results: Contains solution, solver, metadata
- ✅ PyTorch support: Contains torch.Tensor, supports .backward()
- ✅ Type conversion: Provides to_numpy() method

#### 5. `helpers.py` - Helper Tools

**Responsibilities**:
- ✅ Reduce boilerplate: Provide wrappers for common scenarios
- ✅ Gradient verification: Provide gradient checking tools

### Adjoint Method (Adjoint Method) Detailed Explanation

The adjoint method is the core algorithm for computing gradients of PDE-constrained optimization problems. It is more efficient than directly computing gradients or using automatic differentiation.

#### Why Do We Need the Adjoint Method?

**Problem Scenario**: We need to compute gradients of objective function with respect to parameters, but the objective function depends on the solution of the PDE.

For optimization problem:
```
minimize J(u(θ), θ)
subject to F(u(θ), θ) = 0  (PDE constraint)
```

Where:
- `u` is the solution of the PDE (displacement field)
- `θ` is the optimization parameter (e.g., vertex coordinates, material parameters, etc.)
- `J` is the objective function (e.g., total displacement, stress, etc.)
- `F` is the PDE residual (e.g., linear elasticity equation)

**Problem with Direct Gradient Computation**:
```
dJ/dθ = ∂J/∂θ + (∂J/∂u) · (du/dθ)
```

To compute `du/dθ`, we need to solve a linear system for each parameter:
```
F_u · (du/dθ) = -F_θ
```

If there are N parameters, we need to solve N linear systems, with computational cost O(N × solve cost), which is very expensive!

#### Mathematical Principles of the Adjoint Method

**Core Idea**: Introduce adjoint variable `λ` (also called Lagrange multiplier) to transform constrained optimization problem into unconstrained problem.

**Adjoint Equation**:
```
F_u^T · λ = -∂J/∂u
```

Where `F_u^T` is the transpose of the Jacobian matrix of PDE residual with respect to solution (i.e., the adjoint operator).

**Gradient Formula**:
```
dJ/dθ = ∂J/∂θ + λ^T · F_θ
```

**Key Advantages**:
- ✅ **Only need to solve adjoint equation once**, regardless of number of parameters!
- ✅ Computational cost is O(1 × solve cost), not O(N × solve cost)
- ✅ For problems with millions of parameters, efficiency improvement is enormous

#### Implementation in PolyFEM

The adjoint method is implemented in PolyFEM through `torch.autograd.Function`, divided into forward pass and backward pass:

##### 1. Forward Pass

Forward pass is just running one ordinary simulation:

```python
def forward(ctx, solver, vertices, derivative_type):
    # 1. Update mesh vertices
    solver.mesh().set_vertices(vertices)
    
    # 2. Enable derivative cache (critical!)
    # This caches intermediate results like stiffness matrix, residual, etc., for backward pass
    solver.set_cache_level(pf.CacheLevel.Derivatives)
    
    # 3. Run simulation, solve PDE
    # This solves F(u, θ) = 0, obtaining solution u
    solver.solve()
    
    # 4. Save solver to ctx (for backward)
    # ctx is used to pass data between forward and backward
    ctx.solver = solver
    ctx.derivative_type = derivative_type
    
    # 5. Return solution (as PyTorch Tensor)
    return torch.tensor(solver.get_solutions())
```

**Key Points**:
- `set_cache_level(CacheLevel.Derivatives)` caches intermediate results needed for computing derivatives (stiffness matrix, residual, etc.)
- If cache is not set, backward pass cannot compute gradients

##### 2. Backward Pass

Backward pass uses adjoint method to compute gradients:

```python
def backward(ctx, grad_output):
    # grad_output = d(loss)/d(solution) = ∂J/∂u
    # This is the gradient backpropagated from loss function to solution
    
    # 1. Solve adjoint equation: F_u^T · λ = -∂J/∂u
    # solve_adjoint() uses cached stiffness matrix transpose from forward pass
    ctx.solver.solve_adjoint(grad_output.numpy())
    
    # 2. Compute parameter derivative: dJ/dθ = ∂J/∂θ + λ^T · F_θ
    # Use adjoint solution λ and cached F_θ to compute gradient
    if ctx.derivative_type == "shape":
        grad = pf.shape_derivative(ctx.solver)  # Shape derivative
    elif ctx.derivative_type == "material":
        grad = pf.elastic_material_derivative(ctx.solver)  # Material derivative
    elif ctx.derivative_type == "initial_velocity":
        grad = pf.initial_velocity_derivative(ctx.solver)  # Initial velocity derivative
    
    # 3. Return gradient
    # Return tuple, corresponding to each input of forward
    return None, torch.tensor(grad), None
    #     ↑      ↑                ↑
    #   solver  vertices    derivative_type
```

**Key Points**:
- `solve_adjoint()` solves adjoint equation, obtaining adjoint solution `λ`
- `shape_derivative()` and similar functions use adjoint solution and cached intermediate results to compute gradients
- Regardless of number of parameters, only need to solve adjoint equation once

#### Performance Advantages

**Compared to Automatic Differentiation**:
- **Automatic Differentiation**: Need to store all intermediate computations, huge memory consumption (O(number of computation steps))
- **Adjoint Method**: Only cache necessary intermediate results (stiffness matrix, residual, etc.), small memory consumption (O(problem size))

**Compared to Direct Computation**:
- **Direct Computation**: Each parameter needs to solve one linear system, computational cost O(N)
- **Adjoint Method**: Only need to solve adjoint equation once, computational cost O(1)

**Actual Effect**:
- For shape optimization problem with 1000 parameters, adjoint method is 1000 times faster than direct computation
- For problems with millions of parameters, adjoint method is the only feasible solution

#### Summary

The adjoint method is the core algorithm of Differentiable functionality, achieving efficient gradient computation through:

1. **Forward Pass**: Run one simulation, cache necessary intermediate results
2. **Backward Pass**: Solve adjoint equation once, use adjoint solution to compute gradients for all parameters
3. **Performance Advantage**: Computational cost is independent of number of parameters, only depends on problem size

This is why `solve_differentiable()` can efficiently compute gradients, even for large-scale optimization problems.

### Relationship Between Differentiable and the C++ Extension

#### Why Does Differentiable Need the C++ Extension?

**Core Reason**: Differentiable functionality needs direct access to C++ Solver objects and low-level methods, which are only available from the compiled `polyfempy` C++ extension (nanobind/pybind11).

##### 1. Need C++ Solver Object

Differentiable does not go through the unified `solve()` entry point (which calls the C++ extension once per solve and does not expose the solver). It uses `pf.Solver()` directly and keeps solver state across forward/backward:

```python
# solve_differentiable() internal implementation
import polyfempy as pf
solver = pf.Solver()  # Directly create C++ Solver object
solver.set_mesh(V, C)
solver.build_basis()
solver.assemble()
solver.solve()
```

**Why?** Because we need:
- Maintain solver state between `forward()` and `backward()`
- Directly call C++ methods (like `solve_adjoint()`)
- Access internal cache and intermediate results

##### 2. Need C++ Adjoint Methods

Backward pass needs to call C++-implemented adjoint methods:

```python
# Must call in backward()
ctx.solver.solve_adjoint(grad_output)  # C++ method
grad = pf.shape_derivative(ctx.solver)  # C++ method
```

These methods only have C++ implementation and must be accessed through the compiled `polyfempy` C++ extension.

##### 3. Need Derivative Computation Functions

Differentiable needs to compute various derivatives:
- `pf.shape_derivative()`: Shape derivative
- `pf.elastic_material_derivative()`: Material parameter derivative
- `pf.initial_velocity_derivative()`: Initial velocity derivative

These all need real stiffness matrix, residual, and other intermediate results, which only the C++ extension provides.

#### C++ Extension Check

`solve_differentiable()` checks that the C++ extension is available (e.g. via `polyfempy.cpp_backend_available()` or a direct import); if not built, it raises a clear error telling the user to build the C++ module first.

#### Architecture Comparison

**Regular `solve()` API**:
```
User Code
    ↓
solve() - Unified interface (solve.py)
    ↓
Direct call to polyfempy C++ extension (Route A)
    ↓
Return Result object
```

**Differentiable API**:
```
User Code
    ↓
solve_differentiable() - High-level wrapper
    ↓
Direct use of pf.Solver() with state kept (C++ API)
    ↓
PolyFEMFunction (PyTorch integration)
    ↓
Return DifferentiableResult object
```

**Key Differences**:
- `solve()`: Each call is a single solve; it calls the C++ extension and returns, without exposing the Solver object.
- `solve_differentiable()`: Needs to keep Solver state across forward/backward, so it uses `pf.Solver()` directly and does not go through the `solve()` wrapper.

#### Why Doesn’t solve_differentiable Go Through solve()?

Exposing the Solver in the `solve()` layer or adding a “differentiable mode” there would:
1. **Increase coupling**: `solve()` is currently a simple “config → solve → result” path; it is not meant to hold state or backward logic.
2. **Keep responsibilities clear**: Differentiable as a separate module using `pf.Solver()` directly is easier to maintain.
3. **Same dependency**: Both rely on the same C++ extension; they just use it differently (one-shot vs stateful).

#### Usage Recommendations

1. **Ensure the C++ extension is built**: `solve_differentiable()` depends on the `polyfempy` C++ extension (either nanobind or pybind11 build). If not built, a clear error is shown.
2. **Error handling**: If the C++ extension is not loaded, an appropriate `ImportError` or module error is raised.

#### Summary

| Aspect | `solve()` | `solve_differentiable()` |
|--------|-----------|-------------------------|
| **Entry** | solve.py unified entry | Direct use of `pf.Solver()` |
| **API path** | solve() → C++ extension | Direct C++ API (keep Solver state) |
| **Solver object** | Not exposed | Direct use of `pf.Solver()` |
| **Purpose** | Regular simulation | Gradient computation, optimization |
| **Dependencies** | Needs C++ extension | Needs C++ extension |

---

## API Design

Differentiable functionality is provided as an independent module, with **cfg forms aligned with solve()** (str path / dict / SimulationConfig):

```python
# Base API (all users)
from polyfempy.api import solve
result = solve(V, C, cfg)

# Differentiable API (users who need gradients)
from polyfempy.differentiable import solve_differentiable
# Style 1: pass V, C, cfg (in-memory mesh)
result = solve_differentiable(V, C, cfg, derivative_type="shape")
# Style 2: pass only cfg; mesh loaded from config's geometry (root_path required)
result = solve_differentiable(cfg="path/to/config.json", derivative_type="shape")
result = solve_differentiable(cfg=SimulationConfig(...), root_path=str(data_dir), ...)
```

**Design Advantages**:
- ✅ Clear responsibilities: Base API stays simple, Differentiable as optional module
- ✅ Dependency management: Only users who need it import PyTorch
- ✅ Backward compatibility: Doesn't affect existing users
- ✅ Easy maintenance: Logic separation, easy to extend

---

## New Implementation vs Old Implementation

### Detailed Comparison

#### Old Implementation (20+ lines, requires manual setup)

```python
import polyfempy as pf
import torch
import json
from legacy_differentiable.diffSimulator import Simulate  # Pre-written class exists

# Usage (requires manual solver configuration)
solver = pf.Solver()
solver.set_settings(json.dumps(cfg))  # ⚠️ Need manual configuration
solver.set_mesh(V, C)                 # ⚠️ Need manual mesh setup
solver.build_basis()                   # ⚠️ Need manual basis construction
solver.assemble()                      # ⚠️ Need manual assembly

vertices = torch.tensor(V, requires_grad=True)
result = Simulate.apply(solver, vertices)  # Use existing Simulate class
loss = torch.norm(result)
loss.backward()
grad = vertices.grad
```

**Problems**:
- ❌ Large code volume (20+ lines), requires manual solver setup
- ❌ Need to manually configure all solver parameters (mesh, materials, boundary conditions, etc.)
- ❌ Need to manually call build_basis(), assemble() and other steps
- ❌ Need to import Simulate class from legacy_differentiable
- ❌ Inconsistent with `solve()` API, different configuration method

#### New Implementation (5 lines, automatic handling)

```python
from polyfempy.differentiable import solve_differentiable
import torch

# Usage (5 lines of code!)
vertices = torch.tensor(V, requires_grad=True)
result = solve_differentiable(vertices, C, cfg)  # ✅ Automatically handles everything
loss = torch.norm(result.u)
loss.backward()
grad = vertices.grad  # ✅ Automatically computes gradient
```

**Advantages**:
- ✅ 75% code reduction (from 20+ lines to 5 lines)
- ✅ Automatic cache setup (won't forget)
- ✅ Automatic adjoint call (won't forget)
- ✅ Automatic solver configuration (no manual work)
- ✅ API completely consistent with `solve()`
- ✅ Supports multiple derivative types
- ✅ Clear error messages

### Feature Comparison Table

| Feature | Old Implementation | New Implementation | Notes |
|---------|-------------------|-------------------|-------|
| **Code Volume** | 20+ lines | 5 lines | 75% reduction |
| **Function Class** | Existing Simulate class | Automatic wrapper | ✅ No import needed |
| **Solver Configuration** | Manual (set_settings, set_mesh, build_basis, assemble) | Automatic | ✅ Improved |
| **Cache Setup** | Handled in Simulate class | Automatic | ✅ Both support |
| **Adjoint Call** | Handled in Simulate class | Automatic | ✅ Both support |
| **Shape Optimization** | ✅ | ✅ | Both support, new implementation simpler |
| **Material Optimization** | ❌ | ✅ | ✅ New feature |
| **Initial Velocity Optimization** | ✅ | ✅ | Both support |
| **API Consistency** | ❌ (uses old API) | ✅ (consistent with solve()) | ✅ New advantage |
| **Error Handling** | Basic | Comprehensive | ✅ Improved |
| **Documentation and Examples** | Limited | Complete | ✅ Improved |

### Why Is New Implementation Better?

1. **Reduce Code Volume**: From 20+ lines to 5 lines (75% reduction), no need to manually configure solver
2. **API Consistency**: Completely consistent with `solve()` API, uses same configuration format
3. **Automatic Handling**: Automatically configures solver, sets cache, calls adjoint, won't forget critical steps
4. **Easy to Learn**: No need to understand legacy_differentiable, directly use unified API
5. **More Features**: Supports more derivative types (material, initial_velocity) and helper tools

---

## Supported Derivative Types

`solve_differentiable()` supports multiple derivative types, specified via `derivative_type` parameter:

### 1. Shape Derivative (Most Common)

Compute derivative of objective function with respect to mesh vertices: `d(loss)/d(vertices)`

```python
result = solve_differentiable(vertices, C, cfg, derivative_type="shape")
loss = torch.norm(result.u)
loss.backward()
grad = vertices.grad  # Shape derivative
```

### 2. Material Parameter Derivative

Compute derivative of objective function with respect to material parameters: `d(loss)/d(material_params)`

```python
result = solve_differentiable(V, C, cfg, derivative_type="material")
# Need to set material parameters as differentiable
```

### 3. Initial Velocity Derivative

Compute derivative of objective function with respect to initial velocity: `d(loss)/d(initial_velocity)`

```python
result = solve_differentiable(V, C, cfg, derivative_type="initial_velocity")
```

---

## Complete Examples

### Shape Optimization Example

Complete shape optimization example, including gradient computation and optimization loop:

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

# Create differentiable vertices
vertices = torch.tensor(V, requires_grad=True)

# Run differentiable simulation
result = solve_differentiable(vertices, C, cfg, derivative_type="shape")

# Define loss function (e.g., minimize total displacement)
loss = torch.norm(result.u)

# Compute gradient
loss.backward()

# Get gradient
grad = vertices.grad
print(f"Gradient shape: {grad.shape}")
print(f"Gradient norm: {torch.norm(grad).item()}")

# Use gradient for optimization (e.g., gradient descent)
learning_rate = 0.01
vertices_new = vertices - learning_rate * grad
```

---

## Advanced Usage

### 1. Custom Function (Advanced Users)

If you need complete control over forward and backward passes, you can inherit from `PolyFEMFunction`:

```python
from polyfempy.differentiable import PolyFEMFunction
import torch

class MyCustomSimulate(PolyFEMFunction):
    """Custom Function with additional processing logic"""
    
    @staticmethod
    def forward(ctx, solver, vertices, derivative_type="shape"):
        # Can add custom logic here
        # e.g., preprocess vertices, record additional information, etc.
        
        # Call parent method
        result = PolyFEMFunction.forward(ctx, solver, vertices, derivative_type)
        
        # Can add post-processing here
        # e.g., compute additional quantities, save intermediate results, etc.
        
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        # Can customize backward pass logic
        # Or directly use parent method
        return PolyFEMFunction.backward(ctx, grad_output)

# Usage
result = MyCustomSimulate.apply(solver, vertices, "shape")
```

### 2. Using Helper Tools

#### Create Shape Optimizer

```python
from polyfempy.differentiable import create_shape_optimizer

# Create optimizer function
optimizer = create_shape_optimizer(V, C, cfg)

# Use
vertices = torch.tensor(V, requires_grad=True)
loss, grad = optimizer(vertices)
```

#### Verify Gradient

```python
from polyfempy.differentiable import gradient_check

def loss_fn(vertices):
    result = solve_differentiable(vertices, C, cfg)
    return torch.norm(result.u)

vertices = torch.tensor(V, requires_grad=True)
result = solve_differentiable(vertices, C, cfg)
loss = loss_fn(vertices)
loss.backward()
grad = vertices.grad

# Verify gradient is correct
is_correct, error = gradient_check(loss_fn, vertices, grad)
print(f"Gradient correct: {is_correct}, relative error: {error}")
```

### 3. Integration with Optimization Libraries

```python
from polyfempy.differentiable import solve_differentiable
import torch
import torch.optim as optim

# Prepare data
vertices = torch.tensor(V, requires_grad=True)

# Create optimizer
optimizer = optim.Adam([vertices], lr=0.01)

# Optimization loop
for iteration in range(100):
    optimizer.zero_grad()
    
    # Run differentiable simulation
    result = solve_differentiable(vertices, C, cfg)
    
    # Compute loss
    loss = torch.norm(result.u)
    
    # Backward pass
    loss.backward()
    
    # Update parameters
    optimizer.step()
    
    print(f"Iteration {iteration}: loss = {loss.item()}")
```

---

## Implementation Details

### Key Technical Details

#### 1. Why Does `backward()` Return a Tuple?

`backward()` must return a tuple of gradients matching the number of inputs to `forward()`:

```python
# forward has 3 inputs: solver, vertices, derivative_type
@staticmethod
def forward(ctx, solver, vertices, derivative_type):
    ...

# backward must return 3 gradients
@staticmethod
def backward(ctx, grad_output):
    return None, grad_tensor, None
    #     ↑      ↑           ↑
    #   solver  vertices  derivative_type
```

#### 2. Why Do We Need `ctx`?

`ctx` is used to pass data between `forward()` and `backward()`:

```python
def forward(ctx, ...):
    # Save to ctx for backward to use
    ctx.solver = solver
    ctx.derivative_type = derivative_type

def backward(ctx, grad_output):
    # Read from ctx
    solver = ctx.solver
    derivative_type = ctx.derivative_type
```

#### 3. Why Do We Need `set_cache_level(CacheLevel.Derivatives)`?

Computing derivatives requires intermediate results (like stiffness matrix, residual, etc.), cache must be enabled:

```python
solver.set_cache_level(pf.CacheLevel.Derivatives)
```

If not set, `backward()` will error.

#### 4. Why Do We Need `solve_adjoint()`?

The core of adjoint method is solving the adjoint equation:

```python
# Must call in backward
ctx.solver.solve_adjoint(grad_output)
```

This solves the adjoint problem, preparing for subsequent derivative computation.

### Performance Considerations

1. **Memory**: Only cache necessary intermediate results, much more memory-efficient than automatic differentiation
2. **Computation**: Only need one adjoint solve, regardless of number of parameters
3. **Parallelization**: Can run multiple independent simulations in parallel

### Limitations and Future Improvements

**Current Limitations**:
- Requires the C++ extension (polyfempy) to be built; otherwise unavailable
- Can only compute one derivative type at a time

**Future Improvements**:
- Support computing multiple derivative types simultaneously
- Support more derivative types
- Better error handling and diagnostics

---

