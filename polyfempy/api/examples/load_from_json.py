"""
Example: Load configuration from JSON file.

This example demonstrates how to load a complete simulation configuration
from a JSON file, including geometry, materials, and boundary conditions.

Run with:
    python -m polyfempy.api.examples.load_from_json
"""

from pathlib import Path
from polyfempy.api import solve, SimulationConfig


def main():
    # Example 1: Load from JSON file (if geometry is in JSON, mesh will be loaded automatically)
    json_file = Path("data/example_config.json")  # Replace with your JSON file
    
    if not json_file.exists():
        print(f"Note: {json_file} does not exist. Creating a minimal example...")
        print("\nTo use this example with a real JSON file:")
        print("1. Create a JSON file with your configuration")
        print("2. Update the 'json_file' path in this script")
        print("3. If your JSON includes 'geometry', the mesh will be loaded automatically")
        print("\nExample JSON structure:")
        print("""
{
    "geometry": [{"mesh": "path/to/mesh.msh"}],
    "materials": [{"type": "LinearElasticity", "E": 1e6, "nu": 0.3}],
    "boundary_conditions": {
        "dirichlet_boundary": [{"id": 1, "value": [0.0, 0.0]}],
        "neumann_boundary": [{"id": 2, "value": [0.0, -1000.0]}]
    },
    "solver": {
        "linear": {"max_iter": 1000},
        "nonlinear": {"max_iter": 50}
    }
}
        """)
        return
    
    # Load configuration from JSON file
    print(f"Loading configuration from: {json_file}")
    cfg = SimulationConfig.from_json_file(str(json_file))
    
    print(f"\nConfiguration loaded:")
    print(f"  PDE: {cfg.pde}")
    print(f"  Materials: {cfg.materials}")
    print(f"  Boundary conditions: {cfg.boundary_conditions}")
    
    # If JSON contains geometry, vertices and cells can be None
    # The mesh will be loaded from the file specified in geometry
    print("\nSolving...")
    result = solve(vertices=None, cells=None, cfg=cfg)
    
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    print(result.summary())
    print(f"\nSolution shape: {result.u.shape if result.u is not None else 'N/A'}")
    print(f"Backend: {result.meta['backend']}")


if __name__ == "__main__":
    main()

