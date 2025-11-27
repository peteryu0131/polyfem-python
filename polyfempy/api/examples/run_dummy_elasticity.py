"""
Dummy backend example: minimal 2D elasticity on unit square.

This example demonstrates the Dummy backend with a simple 2D mesh.
Run with:
    python -m polyfempy.api.examples.run_dummy_elasticity
"""

import numpy as np
from polyfempy.api import solve, SimulationConfig


def main():
    # Unit square mesh: 4 vertices forming 2 triangles
    V = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ], dtype=np.float64)
    
    C = np.array([
        [0, 1, 2],  # Triangle 1
        [0, 2, 3]   # Triangle 2
    ], dtype=np.int32)
    
    # Configuration with default values
    cfg = SimulationConfig()
    
    print("Solving with Dummy backend...")
    print(f"Vertices: {V.shape}")
    print(f"Cells: {C.shape}")
    
    # Solve
    result = solve(V, C, cfg)
    
    # Print summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(result.summary())
    
    print("\n" + "="*50)
    print("META")
    print("="*50)
    print(result.meta)
    
    print("\n" + "="*50)
    print("SUCCESS")
    print("="*50)
    print(f"u shape: {result.u.shape}")
    print(f"u dtype: {result.u.dtype}")
    print(f"Backend: {result.meta['backend']}")
    print(f"Iterations: {result.meta['iters']}")
    print(f"Residual: {result.meta['residual']}")
    print(f"Seed: {result.meta['seed']}")


if __name__ == "__main__":
    main()

