"""Simple shape optimization example."""

import torch
import numpy as np
from polyfempy.differentiable import solve_differentiable


def shape_optimization():
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
    
    vertices = torch.tensor(V, requires_grad=True)
    result = solve_differentiable(vertices, C, cfg, derivative_type="shape")
    
    loss = torch.norm(result.u)
    print(f"Loss: {loss.item()}")
    
    loss.backward()
    grad = vertices.grad
    print(f"Gradient shape: {grad.shape}")
    print(f"Gradient norm: {torch.norm(grad).item()}")
    
    return loss, grad


if __name__ == "__main__":
    loss, grad = shape_optimization()

