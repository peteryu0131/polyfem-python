"""Example: Using dedicated material classes for better IDE support.

This example demonstrates the new dedicated material classes (NeoHookean, 
LinearElasticity, Stokes, etc.) which provide IDE autocomplete support and
support multiple input modes (E-nu vs lambda-mu, etc.).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from api.config import (
    SimulationConfig,
    # Elastic materials
    NeoHookean,
    IsochoricNeoHookean,
    LinearElasticity,
    HookeLinearElasticity,
    SaintVenant,
    IncompressibleLinearElasticity,
    # Hyperelastic materials
    MooneyRivlin,
    MooneyRivlin3Param,
    MooneyRivlin3ParamSymbolic,
    UnconstrainedOgden,
    IncompressibleOgden,
    # Fluid materials
    Stokes,
    NavierStokes,
    OperatorSplitting,
    # Other materials
    Electrostatics,
)


def example_neo_hookean():
    """Example 1: NeoHookean material with E-nu and lambda-mu inputs."""
    print("=" * 70)
    print("Example 1: NeoHookean Material")
    print("=" * 70)
    
    # Mode 1: E-nu input (Young's modulus and Poisson's ratio)
    material1 = NeoHookean(
        E=2100,      # Young's modulus
        nu=0.3,      # Poisson's ratio
        rho=1.0,     # Density
        id=0
    )
    print("NeoHookean (E-nu input):")
    print(f"  E={material1.E}, nu={material1.nu}, rho={material1.rho}")
    print(f"  to_dict(): {material1.to_dict()}")
    
    # Mode 2: lambda-mu input (Lamé parameters)
    material2 = NeoHookean(
        lambda_=1000,  # First Lamé parameter
        mu=800,        # Shear modulus
        rho=1.0
    )
    print("\nNeoHookean (lambda-mu input):")
    print(f"  lambda_={material2.lambda_}, mu={material2.mu}, rho={material2.rho}")
    print(f"  to_dict(): {material2.to_dict()}")
    
    # Use in SimulationConfig
    cfg = SimulationConfig(materials=material1)
    print(f"\nSimulationConfig materials: {cfg.to_dict()['materials']}")
    print()


def example_linear_elasticity():
    """Example 2: LinearElasticity material with multiple input modes."""
    print("=" * 70)
    print("Example 2: LinearElasticity Material")
    print("=" * 70)
    
    # E-nu input
    material1 = LinearElasticity(E=2100, nu=0.3, rho=1.0)
    print("LinearElasticity (E-nu input):")
    print(f"  {material1.to_dict()}")
    
    # lambda-mu input
    material2 = LinearElasticity(lambda_=1000, mu=800, rho=1.0)
    print("\nLinearElasticity (lambda-mu input):")
    print(f"  {material2.to_dict()}")
    
    # Use in SimulationConfig
    cfg = SimulationConfig(materials=material1)
    print(f"\nSimulationConfig: {cfg.to_dict()['materials']}")
    print()


def example_hooke_saint_venant():
    """Example 3: HookeLinearElasticity and SaintVenant with elasticity_tensor."""
    print("=" * 70)
    print("Example 3: HookeLinearElasticity and SaintVenant")
    print("=" * 70)
    
    # HookeLinearElasticity with E-nu
    hooke1 = HookeLinearElasticity(
        E=2100,
        nu=0.3,
        fiber_direction=[1, 0, 0]
    )
    print("HookeLinearElasticity (E-nu input):")
    print(f"  {hooke1.to_dict()}")
    
    # HookeLinearElasticity with elasticity_tensor
    hooke2 = HookeLinearElasticity(
        elasticity_tensor=[100, 50, 50, 0, 0, 0, 50, 100, 50, 0, 0, 0, 50, 50, 100, 0, 0, 0,
                           0, 0, 0, 25, 0, 0, 0, 0, 0, 0, 25, 0, 0, 0, 0, 0, 0, 25],
        fiber_direction=[1, 0, 0]
    )
    print("\nHookeLinearElasticity (elasticity_tensor input):")
    print(f"  type: {hooke2.type}")
    print(f"  has elasticity_tensor: {'elasticity_tensor' in hooke2.to_dict()}")
    
    # SaintVenant with E-nu
    saint1 = SaintVenant(
        E=2100,
        nu=0.3,
        phi=0,
        psi=0,
        fiber_direction=[0, 1, 0]
    )
    print("\nSaintVenant (E-nu input):")
    print(f"  {saint1.to_dict()}")
    print()


def example_mooney_rivlin():
    """Example 4: MooneyRivlin materials."""
    print("=" * 70)
    print("Example 4: MooneyRivlin Materials")
    print("=" * 70)
    
    # MooneyRivlin
    mr = MooneyRivlin(c1=0.5, c2=0.1, k=1000, rho=1.0)
    print("MooneyRivlin:")
    print(f"  c1={mr.c1}, c2={mr.c2}, k={mr.k}, rho={mr.rho}")
    print(f"  to_dict(): {mr.to_dict()}")
    
    # MooneyRivlin3Param
    mr3 = MooneyRivlin3Param(c1=0.5, c2=0.1, c3=0.05, d1=1000, rho=1.0)
    print("\nMooneyRivlin3Param:")
    print(f"  c1={mr3.c1}, c2={mr3.c2}, c3={mr3.c3}, d1={mr3.d1}")
    print(f"  to_dict(): {mr3.to_dict()}")
    
    # MooneyRivlin3ParamSymbolic
    mr3s = MooneyRivlin3ParamSymbolic(c1=0.5, c2=0.1, c3=0.05, d1=1000, rho=1.0)
    print("\nMooneyRivlin3ParamSymbolic:")
    print(f"  to_dict(): {mr3s.to_dict()}")
    print()


def example_ogden():
    """Example 5: Ogden materials."""
    print("=" * 70)
    print("Example 5: Ogden Materials")
    print("=" * 70)
    
    # UnconstrainedOgden
    uo = UnconstrainedOgden(
        alphas=2.0,
        mus=[1.0, 0.5],
        Ds=[0.1, 0.2],
        rho=1.0
    )
    print("UnconstrainedOgden:")
    print(f"  alphas={uo.alphas}, mus={uo.mus}, Ds={uo.Ds}")
    print(f"  to_dict(): {uo.to_dict()}")
    
    # IncompressibleOgden
    io = IncompressibleOgden(c=1.0, m=2.0, k=1000, rho=1.0)
    print("\nIncompressibleOgden:")
    print(f"  c={io.c}, m={io.m}, k={io.k}")
    print(f"  to_dict(): {io.to_dict()}")
    print()


def example_fluid_materials():
    """Example 6: Fluid materials (Stokes, NavierStokes, OperatorSplitting)."""
    print("=" * 70)
    print("Example 6: Fluid Materials")
    print("=" * 70)
    
    # Stokes
    stokes = Stokes(viscosity=0.1, rho=1.0)
    print("Stokes:")
    print(f"  viscosity={stokes.viscosity}, rho={stokes.rho}")
    print(f"  to_dict(): {stokes.to_dict()}")
    
    # NavierStokes
    ns = NavierStokes(viscosity=0.1, rho=1.0)
    print("\nNavierStokes:")
    print(f"  viscosity={ns.viscosity}, rho={ns.rho}")
    print(f"  to_dict(): {ns.to_dict()}")
    
    # OperatorSplitting
    os = OperatorSplitting(viscosity=0.1, rho=1.0)
    print("\nOperatorSplitting:")
    print(f"  viscosity={os.viscosity}, rho={os.rho}")
    print(f"  to_dict(): {os.to_dict()}")
    
    # Use in SimulationConfig
    cfg = SimulationConfig(materials=stokes)
    print(f"\nSimulationConfig with Stokes: {cfg.to_dict()['materials']}")
    print()


def example_electrostatics():
    """Example 7: Electrostatics material."""
    print("=" * 70)
    print("Example 7: Electrostatics Material")
    print("=" * 70)
    
    es = Electrostatics(epsilon=8.85e-12, rho=1.0)
    print("Electrostatics:")
    print(f"  epsilon={es.epsilon}, rho={es.rho}")
    print(f"  to_dict(): {es.to_dict()}")
    
    cfg = SimulationConfig(materials=es)
    print(f"\nSimulationConfig: {cfg.to_dict()['materials']}")
    print()


def example_validation():
    """Example 8: Parameter validation in material classes."""
    print("=" * 70)
    print("Example 8: Parameter Validation")
    print("=" * 70)
    
    print("Testing validation for NeoHookean...")
    
    # Test 1: Missing required parameters
    try:
        n = NeoHookean(E=2100)  # Missing nu
        print("  ERROR: Should have raised ValueError")
    except ValueError as e:
        print(f"  [OK] Correctly caught missing parameter: {str(e)}")
    
    # Test 2: Both input modes provided (conflict)
    try:
        n = NeoHookean(E=2100, nu=0.3, lambda_=1000, mu=800)
        print("  ERROR: Should have raised ValueError")
    except ValueError as e:
        print(f"  [OK] Correctly caught conflict: {str(e)}")
    
    # Test 3: Valid input
    try:
        n = NeoHookean(E=2100, nu=0.3)
        print(f"  [OK] Valid input works: {n.to_dict()}")
    except ValueError as e:
        print(f"  ERROR: Should not raise ValueError: {str(e)}")
    
    print()


def example_comparison():
    """Example 9: Comparison with old Material class."""
    print("=" * 70)
    print("Example 9: Comparison - Generic Material vs Dedicated Classes")
    print("=" * 70)
    
    print("\n[OLD] Old way (generic Material class):")
    print("   from api.config import Material")
    print("   material = Material(E=2100, nu=0.3, type='NeoHookean')")
    print("   # IDE doesn't know NeoHookean-specific parameters")
    print("   # No validation for input modes")
    
    print("\n[NEW] New way (dedicated classes):")
    print("   from api.config import NeoHookean")
    print("   material = NeoHookean(E=2100, nu=0.3)  # IDE autocompletes all parameters")
    print("   material = NeoHookean(lambda_=1000, mu=800)  # Alternative input mode")
    print("   # IDE knows all available parameters")
    print("   # Automatic validation of input modes")
    print()


def example_integration():
    """Example 10: Integration with SimulationConfig."""
    print("=" * 70)
    print("Example 10: Integration with SimulationConfig")
    print("=" * 70)
    
    # Different materials in SimulationConfig
    configs = [
        ("NeoHookean (E-nu)", SimulationConfig(materials=NeoHookean(E=2100, nu=0.3))),
        ("NeoHookean (lambda-mu)", SimulationConfig(materials=NeoHookean(lambda_=1000, mu=800))),
        ("LinearElasticity", SimulationConfig(materials=LinearElasticity(E=2100, nu=0.3))),
        ("Stokes", SimulationConfig(materials=Stokes(viscosity=0.1))),
        ("MooneyRivlin", SimulationConfig(materials=MooneyRivlin(c1=0.5, c2=0.1, k=1000))),
    ]
    
    for name, cfg in configs:
        print(f"{name}:")
        print(f"  {cfg.to_dict()['materials']}")
    
    print()


if __name__ == "__main__":
    example_neo_hookean()
    example_linear_elasticity()
    example_hooke_saint_venant()
    example_mooney_rivlin()
    example_ogden()
    example_fluid_materials()
    example_electrostatics()
    example_validation()
    example_comparison()
    example_integration()
    
    print("=" * 70)
    print("Summary:")
    print("  [OK] Dedicated material classes provide IDE autocomplete")
    print("  [OK] Support multiple input modes (E-nu, lambda-mu, elasticity_tensor)")
    print("  [OK] Automatic parameter validation")
    print("  [OK] Type-safe and error-resistant")
    print("  [OK] Clean integration with SimulationConfig")
    print("=" * 70)

