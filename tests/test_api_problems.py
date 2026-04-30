from polyfempy.api import SimulationConfig
from polyfempy.api.problems import (
    Flow,
    Problem,
    Torsion,
    available_problem_names,
    get_problem_class,
)


def test_predefined_problem_registry_includes_torsionelastic_alias():
    assert "TorsionElastic" in available_problem_names()
    assert get_problem_class("TorsionElastic") is Torsion


def test_problem_payload_matches_backend_shape():
    problem = Problem()
    problem.set_displacement(3, [0.0, 0.0], [True, True])
    problem.set_force(4, [1.0, 0.0])

    assert problem.params()["dirichlet_boundary"] == [
        {"id": 3, "value": [0.0, 0.0], "dimension": [True, True]}
    ]
    assert problem.params()["neumann_boundary"] == [
        {"id": 4, "value": [1.0, 0.0]}
    ]
    assert "initial_velocity" not in problem.params()


def test_flow_accepts_correct_and_legacy_amount_spellings():
    correct = Flow(inflow_amount=0.4, outflow_amount=0.6)
    legacy = Flow(inflow_amout=0.4, outflow_amout=0.6)

    assert correct.params()["inflow_amout"] == 0.4
    assert correct.params()["outflow_amout"] == 0.6
    assert legacy.params() == correct.params()


def test_simulation_config_predefined_problem_factories_build_settings():
    configs = [
        SimulationConfig.gravity(),
        SimulationConfig.franke(),
        SimulationConfig.torsion(),
        SimulationConfig.flow(),
        SimulationConfig.driven_cavity(),
        SimulationConfig.flow_with_obstacle(),
    ]

    for cfg in configs:
        settings = cfg.to_settings()
        assert hasattr(settings, "set_problem") or hasattr(settings, "_problem")
