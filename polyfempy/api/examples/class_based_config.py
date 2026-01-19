"""Class-based configuration example.

For dedicated material classes (NeoHookean, LinearElasticity, etc.),
see material_classes_example.py
"""

from polyfempy.api.config import LinearElasticity, SimulationConfig, Material, BoundaryConditions, DirichletBoundary, NeumannBoundary


def example_material_class():
    print("=" * 60)
    print("Example 1: Using Material class")
    print("=" * 60)
    
    material = Material(E=2100, nu=0.3, rho=1.0)
    print(f"Material: E={material.E}, nu={material.nu}, rho={material.rho}")
    
    cfg = SimulationConfig(
        pde="LinearElasticity",
        discr_order=1,
        materials=material
    )
    print(f"Config materials: {cfg.materials}")
    print()


def example_boundary_conditions_class():
    print("=" * 60)
    print("Example 2: Using BoundaryConditions class")
    print("=" * 60)
    
    bc = BoundaryConditions()
    bc.add_dirichlet(id=4, value=[0.0, 0.0])
    bc.add_neumann(id=2, value=[0.0, -1000.0])
    bc.set_rhs([1.0, 0.0])
    
    print(f"BoundaryConditions: {bc.to_dict()}")
    print()


def example_convenience_methods():
    print("=" * 60)
    print("Example 3: Using convenience methods")
    print("=" * 60)
    
    cfg = SimulationConfig()
    cfg.set_material(E=2100, nu=0.3)
    cfg.set_dirichlet_boundary(id=4, value=[0.0, 0.0])
    cfg.set_neumann_boundary(id=2, value=[0.0, -1000.0])
    cfg.set_rhs([1.0, 0.0])
    
    print(f"Config: {cfg.to_dict()}")
    print()


def example_backward_compatibility():
    print("=" * 60)
    print("Example 4: Backward compatibility (dict still works)")
    print("=" * 60)
    
    cfg = SimulationConfig(
        materials={"E": 2100, "nu": 0.3},
        boundary_conditions={
            "dirichlet_boundary": [{"id": 4, "value": [0.0, 0.0]}],
            "rhs": [1.0, 0.0]
        }
    )
    print(f"Config (dict): {cfg.to_dict()}")
    print()


if __name__ == "__main__":
    example_material_class()
    example_boundary_conditions_class()
    example_convenience_methods()
    example_backward_compatibility()

