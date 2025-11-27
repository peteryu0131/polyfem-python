"""Example usage of differentiable simulations.

This module provides example code for common differentiable simulation scenarios.
"""

# Example 1: Basic shape optimization
EXAMPLE_SHAPE_OPTIMIZATION = """
import torch
import numpy as np
from polyfempy.differentiable import solve_differentiable

# Setup
V = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]], dtype=np.float64)
C = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

cfg = {
    "materials": [{"type": "LinearElasticity", "E": 1e6, "nu": 0.3, "rho": 1.0}],
    "boundary_conditions": {
        "dirichlet_boundary": [{"selection": 1, "value": [0.0, 0.0]}],
        "neumann_boundary": [{"selection": 2, "value": [0.0, -1000.0]}],
    },
    "output": {"directory": "out", "paraview": {"file_name": "solution"}}
}

# Make vertices differentiable
vertices = torch.tensor(V, requires_grad=True)

# Run simulation
result = solve_differentiable(vertices, C, cfg, derivative_type="shape")

# Compute loss
loss = torch.norm(result.u)
print(f"Loss: {loss.item()}")

# Compute gradient
loss.backward()
grad = vertices.grad
print(f"Gradient shape: {grad.shape}")
"""

# Example 2: Using helper function
EXAMPLE_WITH_HELPER = """
import torch
import numpy as np
from polyfempy.differentiable import create_shape_optimizer

# Setup (same as above)
V = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]], dtype=np.float64)
C = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
cfg = {...}  # Same config as above

# Create optimizer
optimizer = create_shape_optimizer(V, C, cfg)

# Use it
vertices = torch.tensor(V, requires_grad=True)
loss, grad = optimizer(vertices)
"""

# Example 3: Custom Function (advanced)
EXAMPLE_CUSTOM_FUNCTION = """
import torch
import polyfempy as pf
from polyfempy.differentiable import PolyFEMFunction

class MySimulate(PolyFEMFunction):
    @staticmethod
    def forward(ctx, solver, vertices):
        # Custom forward logic
        return PolyFEMFunction.forward(ctx, solver, vertices, derivative_type="shape")
    
    @staticmethod
    def backward(ctx, grad_output):
        # Custom backward logic (or use default)
        return PolyFEMFunction.backward(ctx, grad_output)

# Use custom function
solver = pf.Solver()
# ... setup solver ...
vertices = torch.tensor(V, requires_grad=True)
result = MySimulate.apply(solver, vertices)
"""

