def test_solver_config_blocks_live_in_config_solver_module():
    from polyfempy.api import config
    from polyfempy.api import config_solver

    names = [
        "LineSearch",
        "AugmentedLagrangian",
        "SolverContactOptions",
        "RayleighDamping",
        "SolverAdvanced",
        "LinearSolver",
        "NonlinearSolver",
        "Solver",
    ]

    for name in names:
        assert getattr(config, name) is getattr(config_solver, name)
