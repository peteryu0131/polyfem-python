"""
Example: Parameter sweep - run multiple simulations with different parameters.

This example demonstrates how to run multiple simulations with different
material parameters (e.g., different Young's moduli) to study parameter sensitivity.

Run with:
    python -m polyfempy.api.examples.parameter_sweep
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
    
    # Define parameter sweep: different Young's moduli
    E_values = [1e5, 1e6, 1e7]  # Different stiffness values
    nu = 0.3  # Fixed Poisson's ratio
    
    print("="*60)
    print("PARAMETER SWEEP: Young's Modulus")
    print("="*60)
    
    results = []
    
    for i, E in enumerate(E_values):
        print(f"\n[{i+1}/{len(E_values)}] Running with E = {E:.1e} Pa")
        
        # Create configuration with current parameter
        cfg = SimulationConfig(
            pde="linear_elasticity",
            materials={"E": E, "nu": nu},
            boundary_conditions={
                "dirichlet_boundary": [{"id": 4, "value": [0.0, 0.0]}],  # Left side fixed
                "rhs": [1.0, 0.0],  # Horizontal body force
            },
            extras={"max_iters": 10, "random_seed": 42}  # Same seed for reproducibility
        )
        
        # Solve
        result = solve(V, C, cfg)
        results.append((E, result))
        
        # Print summary
        u_norm = np.linalg.norm(result.u) if result.u is not None else 0.0
        print(f"  Solution norm: {u_norm:.6e}")
        # Note: iters and residual may not be available in meta
        if 'iters' in result.meta:
            print(f"  Iterations: {result.meta['iters']}")
        if 'residual' in result.meta:
            print(f"  Residual: {result.meta['residual']:.6e}")
    
    # Compare results
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    print(f"{'E (Pa)':<15} {'||u||':<15} {'Iterations':<12} {'Residual':<15}")
    print("-"*60)
    
    for E, result in results:
        u_norm = np.linalg.norm(result.u) if result.u is not None else 0.0
        iters = result.meta.get('iters', 'N/A')
        residual = result.meta.get('residual', 'N/A')
        print(f"{E:<15.1e} {u_norm:<15.6e} {iters:<12} {residual:<15}")
    
    print("\n" + "="*60)
    print("OBSERVATION")
    print("="*60)
    print("As E increases (stiffer material), the displacement decreases.")
    print("This is expected: stiffer materials deform less under the same load.")


if __name__ == "__main__":
    main()

