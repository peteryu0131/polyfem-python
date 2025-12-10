"""Example: Using class-based configuration for better IDE support.

This example demonstrates the class-based API for SimulationConfig,
which provides IDE autocomplete support compared to dictionary-based configuration.

For examples of dedicated material classes (NeoHookean, LinearElasticity, etc.),
see material_classes_example.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from api.config import LinearElasticity, SimulationConfig, Material, BoundaryConditions, DirichletBoundary, NeumannBoundary


def example_material_class():
    """Example 1: Using Material class (IDE autocomplete supported)."""
    print("=" * 60)
    print("Example 1: Using Material class")
    print("=" * 60)
    
    # Create Material object - IDE will autocomplete E, nu, rho, type
    material = Material(E=2100, nu=0.3, rho=1.0)
    print(f"Material: E={material.E}, nu={material.nu}, rho={material.rho}")
    # material.E  # IDE will autocomplete
    # material.nu  # IDE will autocomplete
    
    # Use in SimulationConfig
    cfg = SimulationConfig(
        pde="LinearElasticity",
        discr_order=1,
        materials=material
    )
    print(f"Config materials: {cfg.materials}")
    print()


def example_boundary_conditions_class():
    """Example 2: Using BoundaryConditions class (IDE autocomplete supported)."""
    print("=" * 60)
    print("Example 2: Using BoundaryConditions class")
    print("=" * 60)
    
    # Create BoundaryConditions object - IDE will autocomplete methods
    bc = BoundaryConditions()
    bc.add_dirichlet(id=4, value=[0.0, 0.0])  # IDE will autocomplete
    bc.add_neumann(id=2, value=[0.0, -1000.0])  # IDE will autocomplete
    bc.set_rhs([1.0, 0.0])  # IDE will autocomplete
    
    print(f"BoundaryConditions: {bc.to_dict()}")
    print()


def example_convenience_methods():
    """Example 3: Using convenience methods (IDE autocomplete supported)."""
    print("=" * 60)
    print("Example 3: Using convenience methods")
    print("=" * 60)
    
    # Create config and use convenience methods
    cfg = SimulationConfig()
    cfg.set_material(E=2100, nu=0.3)  # IDE will autocomplete all parameters
    cfg.set_dirichlet_boundary(id=4, value=[0.0, 0.0])  # IDE will autocomplete
    cfg.set_neumann_boundary(id=2, value=[0.0, -1000.0])  # IDE will autocomplete
    cfg.set_rhs([1.0, 0.0])  # IDE will autocomplete
    
    print(f"Config: {cfg.to_dict()}")
    print()


def example_backward_compatibility():
    """Example 4: Backward compatibility with dict (still works, but no IDE autocomplete)."""
    print("=" * 60)
    print("Example 4: Backward compatibility (dict still works)")
    print("=" * 60)
    
    # Old way still works (but no IDE autocomplete)
    cfg = SimulationConfig(
        materials={"E": 2100, "nu": 0.3},  # ⚠️ IDE cannot autocomplete keys
        boundary_conditions={
            "dirichlet_boundary": [{"id": 4, "value": [0.0, 0.0]}],  # ⚠️ IDE cannot autocomplete
            "rhs": [1.0, 0.0]
        }
    )
    print(f"Config (dict): {cfg.to_dict()}")
    print()


def example_comparison():
    """Example 5: Side-by-side comparison."""
    print("=" * 60)
    print("Example 5: Comparison - Class vs Dict")
    print("=" * 60)
    
    print("\n❌ Old way (dict) - No IDE autocomplete:")
    print("   cfg.materials['E'] = 2100  # IDE doesn't know 'E' is valid")
    print("   cfg.materials['nu'] = 0.3  # IDE doesn't know 'nu' is valid")
    
    print("\n✅ New way (class) - IDE autocomplete:")
    print("   material = Material(E=2100, nu=0.3)  # IDE autocompletes E, nu, rho, type")
    print("   cfg = SimulationConfig(materials=material)")
    print("   material.E  # IDE knows this is valid")
    
    print("\n✅ New way (convenience methods) - IDE autocomplete:")
    print("   cfg = SimulationConfig()")
    print("   cfg.set_material(E=2100, nu=0.3)  # IDE autocompletes all parameters")
    print("   cfg.set_dirichlet_boundary(id=4, value=[0.0, 0.0])  # IDE autocompletes")
    print()


if __name__ == "__main__":
    example_material_class()
    example_boundary_conditions_class()
    example_convenience_methods()
    example_backward_compatibility()
    example_comparison()
    
    print("=" * 60)
    print("Summary:")
    print("  ✅ Class-based API provides IDE autocomplete")
    print("  ✅ Convenience methods make it easy to set parameters")
    print("  ✅ Backward compatible with dict-based API")
    print("  ✅ Type checking helps catch errors early")
    print("=" * 60)

