"""
Example: Batch processing - run multiple simulations with error isolation.

This example demonstrates how to use batch_solve() to run multiple simulations
efficiently, with automatic error isolation (one failure doesn't stop others).

Run with:
    python -m polyfempy.api.examples.batch_processing
"""

import numpy as np
from polyfempy.api import batch_solve, SimulationConfig


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
    
    # Create multiple jobs with different configurations
    jobs = []
    
    # Job 1: Linear elasticity with low stiffness
    cfg1 = SimulationConfig(
        pde="linear_elasticity",
        materials={"E": 1e5, "nu": 0.3},
        boundary_conditions={
            "dirichlet_boundary": [{"id": 4, "value": [0.0, 0.0]}],
            "rhs": [1.0, 0.0],
        },
        extras={"max_iters": 10, "random_seed": 42}
    )
    jobs.append((V, C, cfg1, None))
    
    # Job 2: Linear elasticity with high stiffness
    cfg2 = SimulationConfig(
        pde="linear_elasticity",
        materials={"E": 1e7, "nu": 0.3},
        boundary_conditions={
            "dirichlet_boundary": [{"id": 4, "value": [0.0, 0.0]}],
            "rhs": [1.0, 0.0],
        },
        extras={"max_iters": 10, "random_seed": 42}
    )
    jobs.append((V, C, cfg2, None))
    
    # Job 3: Different boundary condition
    cfg3 = SimulationConfig(
        pde="linear_elasticity",
        materials={"E": 1e6, "nu": 0.3},
        boundary_conditions={
            "dirichlet_boundary": [{"id": 4, "value": [0.0, 0.0]}],
            "neumann_boundary": [{"id": 2, "value": [0.0, -1000.0]}],
        },
        extras={"max_iters": 10, "random_seed": 42}
    )
    jobs.append((V, C, cfg3, None))
    
    # Job 4: This one might fail (invalid configuration for demonstration)
    # In real scenarios, batch_solve will isolate the error
    cfg4 = SimulationConfig(
        pde="linear_elasticity",
        materials={"E": 1e6, "nu": 0.3},
        extras={"max_iters": 10, "random_seed": 42}
    )
    jobs.append((V, C, cfg4, None))
    
    print("="*60)
    print("BATCH PROCESSING")
    print("="*60)
    print(f"Running {len(jobs)} simulations in batch...")
    print("(Errors are isolated - one failure won't stop others)\n")
    
    # Run batch
    results = batch_solve(jobs)
    
    # Process results
    print("="*60)
    print("RESULTS")
    print("="*60)
    
    success_count = 0
    failure_count = 0
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            failure_count += 1
            print(f"Job {i+1}: FAILED - {type(result).__name__}: {result}")
        else:
            success_count += 1
            u_norm = np.linalg.norm(result.u) if result.u is not None else 0.0
            print(f"Job {i+1}: SUCCESS")
            print(f"  Solution norm: {u_norm:.6e}")
            print(f"  Iterations: {result.meta['iters']}")
            print(f"  Residual: {result.meta['residual']:.6e}")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total jobs: {len(jobs)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {failure_count}")
    print("\nNote: Even if some jobs fail, others continue to run.")
    print("This is useful for parameter sweeps or Monte Carlo simulations.")


if __name__ == "__main__":
    main()

