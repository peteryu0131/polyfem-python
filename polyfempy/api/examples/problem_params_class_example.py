"""Example: Using ProblemParams classes for better IDE support.

This example demonstrates the new class-based API for problem_params,
which provides IDE autocomplete support compared to dictionary-based configuration.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from api.config import (
    SimulationConfig,
    GravityParams,
    TorsionParams,
    FlowParams,
    FlowWithObstacleParams
)


def example_gravity_params():
    """Example 1: Using GravityParams class (IDE autocomplete supported)."""
    print("=" * 70)
    print("Example 1: Using GravityParams class")
    print("=" * 70)
    
    # Create GravityParams object - IDE will autocomplete 'force'
    params = GravityParams(force=0.1)
    print(f"GravityParams: force={params.force}")
    # params.force  # IDE will autocomplete
    
    # Use in SimulationConfig
    cfg = SimulationConfig(
        problem_type="Gravity",
        problem_params=params
    )
    print(f"Config problem_params: {cfg.problem_params}")
    print(f"  Type: {type(cfg.problem_params).__name__}")
    print()


def example_torsion_params():
    """Example 2: Using TorsionParams class (IDE autocomplete supported)."""
    print("=" * 70)
    print("Example 2: Using TorsionParams class")
    print("=" * 70)
    
    # Create TorsionParams object - IDE will autocomplete all parameters
    params = TorsionParams(
        axis_coordinate=2,  # IDE will autocomplete
        n_turns=0.5,       # IDE will autocomplete
        fixed_boundary=5,  # IDE will autocomplete
        turning_boundary=6 # IDE will autocomplete
    )
    print(f"TorsionParams:")
    print(f"  axis_coordinate: {params.axis_coordinate}")
    print(f"  n_turns: {params.n_turns}")
    print(f"  fixed_boundary: {params.fixed_boundary}")
    print(f"  turning_boundary: {params.turning_boundary}")
    
    # Use in SimulationConfig
    cfg = SimulationConfig(
        problem_type="TorsionElastic",
        problem_params=params
    )
    print(f"Config problem_params type: {type(cfg.problem_params).__name__}")
    print()


def example_flow_params():
    """Example 3: Using FlowParams class (IDE autocomplete supported)."""
    print("=" * 70)
    print("Example 3: Using FlowParams class")
    print("=" * 70)
    
    # Create FlowParams object - IDE will autocomplete all parameters
    # Note: Class uses correct spelling (inflow_amount), not the typo (inflow_amout)
    params = FlowParams(
        inflow=1,           # IDE will autocomplete
        outflow=3,          # IDE will autocomplete
        inflow_amount=0.25, # IDE will autocomplete (correct spelling!)
        outflow_amount=0.25, # IDE will autocomplete (correct spelling!)
        direction=0,        # IDE will autocomplete
        obstacle=[7]        # IDE will autocomplete
    )
    print(f"FlowParams:")
    print(f"  inflow: {params.inflow}")
    print(f"  outflow: {params.outflow}")
    print(f"  inflow_amount: {params.inflow_amount}  # Correct spelling in class!")
    print(f"  outflow_amount: {params.outflow_amount}  # Correct spelling in class!")
    print(f"  direction: {params.direction}")
    print(f"  obstacle: {params.obstacle}")
    
    # Use in SimulationConfig
    cfg = SimulationConfig(
        problem_type="Flow",
        problem_params=params
    )
    print(f"Config problem_params type: {type(cfg.problem_params).__name__}")
    
    # to_dict() handles backward compatibility (converts to legacy typo format)
    print(f"to_dict() (for backend): {cfg.problem_params.to_dict()}")
    print()


def example_flow_with_obstacle_params():
    """Example 4: Using FlowWithObstacleParams class."""
    print("=" * 70)
    print("Example 4: Using FlowWithObstacleParams class")
    print("=" * 70)
    
    # Create FlowWithObstacleParams object
    params = FlowWithObstacleParams(U=1.5, time_dependent=True)
    print(f"FlowWithObstacleParams: U={params.U}, time_dependent={params.time_dependent}")
    
    # Use in SimulationConfig
    cfg = SimulationConfig(
        problem_type="FlowWithObstacle",
        problem_params=params
    )
    print(f"Config problem_params type: {type(cfg.problem_params).__name__}")
    print()


def example_convenience_factories():
    """Example 5: Convenience factories automatically use classes."""
    print("=" * 70)
    print("Example 5: Convenience factories use classes automatically")
    print("=" * 70)
    
    # Gravity factory
    cfg1 = SimulationConfig.gravity(force=0.1, E=1e6, nu=0.3)
    print(f"SimulationConfig.gravity() -> problem_params type: {type(cfg1.problem_params).__name__}")
    print(f"  force: {cfg1.problem_params.force}")
    
    # Torsion factory
    cfg2 = SimulationConfig.torsion(axis_coordinate=2, n_turns=0.5)
    print(f"SimulationConfig.torsion() -> problem_params type: {type(cfg2.problem_params).__name__}")
    print(f"  axis_coordinate: {cfg2.problem_params.axis_coordinate}")
    print(f"  n_turns: {cfg2.problem_params.n_turns}")
    
    # Flow factory
    cfg3 = SimulationConfig.flow(inflow=1, outflow=3, inflow_amount=0.25)
    print(f"SimulationConfig.flow() -> problem_params type: {type(cfg3.problem_params).__name__}")
    print(f"  inflow: {cfg3.problem_params.inflow}")
    print(f"  inflow_amount: {cfg3.problem_params.inflow_amount}  # Correct spelling!")
    
    # FlowWithObstacle factory
    cfg4 = SimulationConfig.flow_with_obstacle(U=1.5, time_dependent=True)
    print(f"SimulationConfig.flow_with_obstacle() -> problem_params type: {type(cfg4.problem_params).__name__}")
    print(f"  U: {cfg4.problem_params.U}")
    print()


def example_backward_compatibility():
    """Example 6: Backward compatibility with dict (still works)."""
    print("=" * 70)
    print("Example 6: Backward compatibility (dict still works)")
    print("=" * 70)
    
    # Old way still works (but no IDE autocomplete)
    cfg = SimulationConfig(
        problem_type="Gravity",
        problem_params={"force": 0.1}  # ⚠️ IDE cannot autocomplete keys
    )
    print(f"Config with dict: {cfg.problem_params}")
    print(f"  Type: {type(cfg.problem_params).__name__}")
    print("  Note: Dict input is still supported for backward compatibility")
    print()


def example_comparison():
    """Example 7: Side-by-side comparison."""
    print("=" * 70)
    print("Example 7: Comparison - Dict vs Class")
    print("=" * 70)
    
    print("\n❌ Old way (dict) - No IDE autocomplete:")
    print("   cfg = SimulationConfig(problem_type='Gravity')")
    print("   cfg.problem_params['force'] = 0.1  # IDE doesn't know 'force' is valid")
    print("   cfg.problem_params['forse'] = 0.1  # Typo, but IDE won't catch it")
    
    print("\n✅ New way (class) - IDE autocomplete:")
    print("   params = GravityParams(force=0.1)  # IDE autocompletes 'force'")
    print("   cfg = SimulationConfig(problem_type='Gravity', problem_params=params)")
    print("   params.force  # IDE knows this is valid")
    print("   params.forse  # IDE will show error (typo detected!)")
    print()


if __name__ == "__main__":
    example_gravity_params()
    example_torsion_params()
    example_flow_params()
    example_flow_with_obstacle_params()
    example_convenience_factories()
    example_backward_compatibility()
    example_comparison()
    
    print("=" * 70)
    print("Summary:")
    print("  ✅ ProblemParams classes provide IDE autocomplete")
    print("  ✅ Correct spelling in classes (inflow_amount, not inflow_amout)")
    print("  ✅ Backward compatible with dict-based API")
    print("  ✅ Type checking helps catch errors early")
    print("  ✅ Convenience factories automatically use classes")
    print("=" * 70)

