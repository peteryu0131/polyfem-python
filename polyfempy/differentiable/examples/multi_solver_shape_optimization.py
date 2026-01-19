"""Multi-solver shape optimization example.

This example demonstrates shape optimization with multiple solvers,
showing how to combine gradients from multiple simulations.

Based on test/test_diff.py
"""

import torch
import numpy as np
import json
from polyfempy.differentiable import solve_differentiable


def multi_solver_optimization():
    """Shape optimization with multiple solvers (different configurations)."""
    
    # Load configuration
    root = "../data/differentiable/input"
    with open(f"{root}/initial-contact.json", "r") as f:
        config = json.load(f)
    
    config["root_path"] = f"{root}/initial-contact.json"
    
    # Get initial vertices and cells from mesh
    import polyfempy as pf
    solver1 = pf.Solver()
    solver1.set_settings(json.dumps(config), False)
    solver1.load_mesh_from_settings()
    V = solver1.mesh().vertices()
    C = solver1.mesh().cells()  # Get cells from mesh
    
    # Configuration for solver 1
    cfg1 = {
        "materials": [{"type": "LinearElasticity", "E": 1e6, "nu": 0.3}],
        "boundary_conditions": {
            "dirichlet_boundary": [{"selection": 1, "value": [0.0, 0.0]}],
        },
        "output": {"directory": "out"}
    }
    
    # Configuration for solver 2 (different initial velocity)
    config2 = config.copy()
    config2["initial_conditions"]["velocity"][0]["value"] = [3, 0]
    cfg2 = {
        "materials": [{"type": "LinearElasticity", "E": 1e6, "nu": 0.3}],
        "boundary_conditions": {
            "dirichlet_boundary": [{"selection": 1, "value": [0.0, 0.0]}],
        },
        "time": {"t0": 0.0, "tend": 1.0, "dt": 0.01, "integrator": "ImplicitEuler"},
        "output": {"directory": "out"}
    }
    
    # Make vertices differentiable
    vertices = torch.tensor(V, requires_grad=True, dtype=torch.float64)
    
    # Run both simulations
    result1 = solve_differentiable(vertices, C, cfg1, derivative_type="shape")
    result2 = solve_differentiable(vertices, C, cfg2, derivative_type="shape")
    
    # Combined loss (from both simulations)
    def loss_fn(vertices):
        r1 = solve_differentiable(vertices, C, cfg1, derivative_type="shape")
        r2 = solve_differentiable(vertices, C, cfg2, derivative_type="shape")
        
        # Use last time step for transient problems
        if r1.u.ndim == 2:
            loss1 = torch.linalg.norm(r1.u[:, -1])
        else:
            loss1 = torch.linalg.norm(r1.u)
        
        if r2.u.ndim == 2:
            loss2 = torch.linalg.norm(r2.u[:, -1])
        else:
            loss2 = torch.linalg.norm(r2.u)
        
        return loss1 * loss2
    
    # Compute loss and gradient
    loss = loss_fn(vertices)
    loss.backward()
    grad = vertices.grad
    
    print(f"Loss: {loss.item()}")
    print(f"Gradient shape: {grad.shape}")
    print(f"Gradient norm: {torch.norm(grad).item()}")
    
    # Verify gradient using finite differences
    theta = torch.randn_like(vertices)
    t = 1e-6
    with torch.no_grad():
        analytic = torch.tensordot(grad, theta)
        f1 = loss_fn(vertices + theta * t)
        f2 = loss_fn(vertices - theta * t)
        fd = (f1 - f2) / (2 * t)
        relative_error = abs(analytic - fd) / (abs(analytic) + 1e-10)
        print(f"\nGradient verification:")
        print(f"  Analytical: {analytic.item()}")
        print(f"  Finite diff: {fd.item()}")
        print(f"  Relative error: {relative_error.item():.3e}")
        assert relative_error < 1e-4, "Gradient verification failed!"
    
    return loss, grad


if __name__ == "__main__":
    loss, grad = multi_solver_optimization()
    print("Optimization completed successfully!")

