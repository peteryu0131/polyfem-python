"""ProblemParams classes example."""

from polyfempy.api.config import (
    SimulationConfig,
    GravityParams,
    TorsionParams,
    FlowParams,
    FlowWithObstacleParams
)


def example_gravity_params():
    print("=" * 70)
    print("Example 1: Using GravityParams class")
    print("=" * 70)
    
    params = GravityParams(force=0.1)
    print(f"GravityParams: force={params.force}")
    
    cfg = SimulationConfig(problem_type="Gravity", problem_params=params)
    print(f"Config problem_params: {cfg.problem_params}")
    print(f"  Type: {type(cfg.problem_params).__name__}")
    print()


def example_torsion_params():
    print("=" * 70)
    print("Example 2: Using TorsionParams class")
    print("=" * 70)
    
    params = TorsionParams(axis_coordinate=2, n_turns=0.5, fixed_boundary=5, turning_boundary=6)
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
    print("=" * 70)
    print("Example 3: Using FlowParams class")
    print("=" * 70)
    
    params = FlowParams(
        inflow=1, outflow=3, inflow_amount=0.25, outflow_amount=0.25,
        direction=0, obstacle=[7]
    )
    print(f"FlowParams:")
    print(f"  inflow: {params.inflow}")
    print(f"  outflow: {params.outflow}")
    print(f"  inflow_amount: {params.inflow_amount}")
    print(f"  outflow_amount: {params.outflow_amount}")
    print(f"  direction: {params.direction}")
    print(f"  obstacle: {params.obstacle}")
    
    cfg = SimulationConfig(problem_type="Flow", problem_params=params)
    print(f"Config problem_params type: {type(cfg.problem_params).__name__}")
    print(f"to_dict() (for backend): {cfg.problem_params.to_dict()}")
    print()


def example_flow_with_obstacle_params():
    print("=" * 70)
    print("Example 4: Using FlowWithObstacleParams class")
    print("=" * 70)
    
    params = FlowWithObstacleParams(U=1.5, time_dependent=True)
    print(f"FlowWithObstacleParams: U={params.U}, time_dependent={params.time_dependent}")
    
    cfg = SimulationConfig(problem_type="FlowWithObstacle", problem_params=params)
    print(f"Config problem_params type: {type(cfg.problem_params).__name__}")
    print()


def example_convenience_factories():
    print("=" * 70)
    print("Example 5: Convenience factories")
    print("=" * 70)
    
    cfg1 = SimulationConfig.gravity(force=0.1, E=1e6, nu=0.3)
    print(f"SimulationConfig.gravity() -> problem_params type: {type(cfg1.problem_params).__name__}")
    print(f"  force: {cfg1.problem_params.force}")
    
    cfg2 = SimulationConfig.torsion(axis_coordinate=2, n_turns=0.5)
    print(f"SimulationConfig.torsion() -> problem_params type: {type(cfg2.problem_params).__name__}")
    print(f"  axis_coordinate: {cfg2.problem_params.axis_coordinate}")
    print(f"  n_turns: {cfg2.problem_params.n_turns}")
    
    cfg3 = SimulationConfig.flow(inflow=1, outflow=3, inflow_amount=0.25)
    print(f"SimulationConfig.flow() -> problem_params type: {type(cfg3.problem_params).__name__}")
    print(f"  inflow: {cfg3.problem_params.inflow}")
    print(f"  inflow_amount: {cfg3.problem_params.inflow_amount}")
    
    cfg4 = SimulationConfig.flow_with_obstacle(U=1.5, time_dependent=True)
    print(f"SimulationConfig.flow_with_obstacle() -> problem_params type: {type(cfg4.problem_params).__name__}")
    print(f"  U: {cfg4.problem_params.U}")
    print()


def example_backward_compatibility():
    print("=" * 70)
    print("Example 6: Backward compatibility (dict still works)")
    print("=" * 70)
    
    cfg = SimulationConfig(problem_type="Gravity", problem_params={"force": 0.1})
    print(f"Config with dict: {cfg.problem_params}")
    print(f"  Type: {type(cfg.problem_params).__name__}")
    print()


def example_comparison():
    print("=" * 70)
    print("Example 7: Comparison - Dict vs Class")
    print("=" * 70)
    
    print("\n[OLD] Old way (dict):")
    print("   cfg = SimulationConfig(problem_type='Gravity')")
    print("   cfg.problem_params['force'] = 0.1")
    
    print("\n[NEW] New way (class):")
    print("   params = GravityParams(force=0.1)")
    print("   cfg = SimulationConfig(problem_type='Gravity', problem_params=params)")
    print()


if __name__ == "__main__":
    example_gravity_params()
    example_torsion_params()
    example_flow_params()
    example_flow_with_obstacle_params()
    example_convenience_factories()
    example_backward_compatibility()
    example_comparison()
    

