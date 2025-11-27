"""
Example: Monitoring simulation progress.

This example demonstrates how to monitor simulation progress and analyze results.
Note: Callbacks are currently not directly supported in the high-level solve() API,
but you can monitor progress by checking the result metadata and running multiple
simulations with different parameters.

Run with:
    python -m polyfempy.api.examples.with_callbacks
"""

import numpy as np
from polyfempy.api import solve, SimulationConfig


def main():
    # Create a simple 2D mesh (unit square with 2 triangles)
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
    
    # Note: Callbacks are not directly supported in solve() API
    # Instead, we can monitor progress by:
    # 1. Running simulations with different max_iters to see convergence
    # 2. Checking result metadata after solve
    # 3. Running parameter sweeps to study behavior
    
    print("="*60)
    print("MONITORING SIMULATION PROGRESS")
    print("="*60)
    print(f"Mesh: {V.shape[0]} vertices, {C.shape[0]} cells")
    print(f"Max iterations: 20")
    print()
    
    # Create configuration
    cfg = SimulationConfig(
        pde="linear_elasticity",
        materials={"E": 1e6, "nu": 0.3},
        boundary_conditions={
            "dirichlet_boundary": [{"id": 4, "value": [0.0, 0.0]}],
            "rhs": [1.0, 0.0],
        },
        extras={"max_iters": 20, "random_seed": 42}  # More iterations to see progress
    )
    
    # Solve
    print("Running simulation...")
    result = solve(V, C, cfg)
    
    # Analyze results
    print("\n" + "="*60)
    print("SIMULATION COMPLETE")
    print("="*60)
    print(f"Total iterations: {result.meta.get('iters', 'N/A')}")
    print(f"Final residual: {result.meta.get('residual', 'N/A'):.6e}")
    print(f"Backend: {result.meta.get('backend', 'N/A')}")
    print()
    
    # Demonstrate monitoring by running with different max_iters
    print("="*60)
    print("CONVERGENCE STUDY")
    print("="*60)
    print("Running simulations with different max_iters to study convergence...")
    
    max_iters_list = [5, 10, 20]
    convergence_data = []
    
    for max_iter in max_iters_list:
        cfg_iter = SimulationConfig(
            pde="linear_elasticity",
            materials={"E": 1e6, "nu": 0.3},
            boundary_conditions={
                "dirichlet_boundary": [{"id": 4, "value": [0.0, 0.0]}],
                "rhs": [1.0, 0.0],
            },
            extras={"max_iters": max_iter, "random_seed": 42}
        )
        result_iter = solve(V, C, cfg_iter)
        convergence_data.append((max_iter, result_iter.meta.get('residual', 0.0)))
        print(f"  max_iters={max_iter:2d}: residual = {result_iter.meta.get('residual', 0.0):.6e}")
    
    print("\nObservation: More iterations generally lead to lower residual.")
    
    # Plot convergence (if matplotlib is available)
    try:
        import matplotlib.pyplot as plt
        iters, residuals = zip(*convergence_data)
        plt.figure(figsize=(8, 5))
        plt.semilogy(iters, residuals, 'o-')
        plt.xlabel('Max Iterations')
        plt.ylabel('Final Residual')
        plt.title('Convergence vs Max Iterations')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('convergence_study.png', dpi=150)
        print("\nConvergence plot saved to: convergence_study.png")
    except ImportError:
        print("\n(Install matplotlib to generate convergence plot)")
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(result.summary())


if __name__ == "__main__":
    main()

