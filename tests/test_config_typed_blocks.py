"""Tests for the newer typed config blocks added on top of ``config.py``.

These tests deliberately stay in pure Python land: they validate config object
construction, JSON round-tripping, and default-preserving ``to_dict()``
behavior without importing the C++ backend.
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
CONFIG_PATH = _REPO / "polyfempy" / "api" / "config.py"

SPEC = importlib.util.spec_from_file_location(
    "polyfempy_api_config_for_typed_block_tests", CONFIG_PATH
)
CONFIG_MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(CONFIG_MODULE)

SimulationConfig = CONFIG_MODULE.SimulationConfig
NeoHookean = CONFIG_MODULE.NeoHookean
LinearElasticity = CONFIG_MODULE.LinearElasticity
HookeLinearElasticity = CONFIG_MODULE.HookeLinearElasticity
SaintVenant = CONFIG_MODULE.SaintVenant
BoundaryConditions = CONFIG_MODULE.BoundaryConditions
Geometry = CONFIG_MODULE.Geometry
GeometryMesh = CONFIG_MODULE.GeometryMesh
GeometryPlane = CONFIG_MODULE.GeometryPlane
GeometryGround = CONFIG_MODULE.GeometryGround
GeometryMeshSequence = CONFIG_MODULE.GeometryMeshSequence
Output = CONFIG_MODULE.Output
OutputLog = CONFIG_MODULE.OutputLog
OutputData = CONFIG_MODULE.OutputData
OutputAdvanced = CONFIG_MODULE.OutputAdvanced
OutputReference = CONFIG_MODULE.OutputReference
OutputParaviewOptions = CONFIG_MODULE.OutputParaviewOptions
ParaviewOutput = CONFIG_MODULE.ParaviewOutput
Contact = CONFIG_MODULE.Contact
CollisionMesh = CONFIG_MODULE.CollisionMesh
Adhesion = CONFIG_MODULE.Adhesion
InitialConditions = CONFIG_MODULE.InitialConditions
SurfaceSelection = CONFIG_MODULE.SurfaceSelection
Constraints = CONFIG_MODULE.Constraints
SoftConstraint = CONFIG_MODULE.SoftConstraint
Space = CONFIG_MODULE.Space
ConfigTests = CONFIG_MODULE.Tests
Input = CONFIG_MODULE.Input
Time = CONFIG_MODULE.Time
AugmentedLagrangian = CONFIG_MODULE.AugmentedLagrangian
Solver = CONFIG_MODULE.Solver
LinearSolver = CONFIG_MODULE.LinearSolver
NonlinearSolver = CONFIG_MODULE.NonlinearSolver
SolverContactOptions = CONFIG_MODULE.SolverContactOptions


class MaterialNamedConstructorTests(unittest.TestCase):
    def test_neohookean_young_poisson_builds_unit_wrapped_values(self):
        material = NeoHookean.young_poisson(
            id=2,
            E=200.0,
            E_unit="GPa",
            nu=0.45,
            rho=7850.0,
            rho_unit="kg/m^3",
        )

        self.assertEqual(material.id, 2)
        self.assertEqual(material.E.to_dict(), {"value": 200.0, "unit": "GPa"})
        self.assertEqual(material.nu, 0.45)
        self.assertEqual(material.rho.to_dict(), {"value": 7850.0, "unit": "kg/m^3"})

    def test_linear_elasticity_young_poisson_accepts_plain_and_unit_wrapped(self):
        unit_wrapped = LinearElasticity.young_poisson(E=20.0, E_unit="MPa", nu=0.3)
        plain = LinearElasticity.young_poisson(E=21.0, nu=0.31)

        self.assertEqual(unit_wrapped.E.to_dict(), {"value": 20.0, "unit": "MPa"})
        self.assertEqual(plain.E, 21.0)
        self.assertEqual(plain.nu, 0.31)

    def test_tensor_material_named_constructors_exist(self):
        hooke = HookeLinearElasticity.young_poisson(E=10.0, nu=0.2)
        saint_venant = SaintVenant.young_poisson(E=11.0, nu=0.25)

        self.assertEqual(hooke.E, 10.0)
        self.assertEqual(hooke.nu, 0.2)
        self.assertEqual(saint_venant.E, 11.0)
        self.assertEqual(saint_venant.nu, 0.25)

    def test_tensor_mode_named_constructors_exist(self):
        hooke = HookeLinearElasticity.tensor(
            elasticity_tensor=[1.0, 2.0, 3.0],
            fiber_direction=[1.0, 0.0, 0.0],
        )
        saint_venant = SaintVenant.tensor(
            elasticity_tensor=[4.0, 5.0, 6.0],
            phi=0.1,
            psi=0.2,
        )

        self.assertEqual(hooke.elasticity_tensor, [1.0, 2.0, 3.0])
        self.assertEqual(hooke.fiber_direction, [1.0, 0.0, 0.0])
        self.assertEqual(saint_venant.elasticity_tensor, [4.0, 5.0, 6.0])
        self.assertEqual(saint_venant.phi, 0.1)
        self.assertEqual(saint_venant.psi, 0.2)

    def test_lame_named_constructors_exist_for_materials_with_lambda_mu_mode(self):
        neo = NeoHookean.lame(
            id=2,
            lambda_=1.2e11,
            lambda_unit="Pa",
            mu=8.0e10,
            mu_unit="Pa",
            rho=7850.0,
            rho_unit="kg/m^3",
        )
        linear = LinearElasticity.lame(lambda_=3.0, mu=4.0)

        self.assertEqual(neo.lambda_.to_dict(), {"value": 120000000000.0, "unit": "Pa"})
        self.assertEqual(neo.mu.to_dict(), {"value": 80000000000.0, "unit": "Pa"})
        self.assertEqual(neo.rho.to_dict(), {"value": 7850.0, "unit": "kg/m^3"})
        self.assertEqual(linear.lambda_, 3.0)
        self.assertEqual(linear.mu, 4.0)


class GeometryTypedEntryTests(unittest.TestCase):
    def test_geometry_items_serializes_multiple_entry_types(self):
        geom = Geometry(
            items=[
                GeometryMesh(mesh="beam.msh", volume_selection=1),
                GeometryPlane(point=[0.0, 0.0], normal=[0.0, 1.0]),
                GeometryGround(height=0.25),
                GeometryMeshSequence(mesh_sequence=["a.obj", "b.obj"], fps=24),
            ]
        )

        payload = geom.to_dict()

        self.assertEqual(payload[0]["mesh"], "beam.msh")
        self.assertEqual(payload[1]["point"], [0.0, 0.0])
        self.assertEqual(payload[2]["height"], 0.25)
        self.assertEqual(payload[3]["mesh_sequence"], ["a.obj", "b.obj"])
        self.assertEqual(payload[3]["fps"], 24)

    def test_geometry_mesh_from_file_named_constructor(self):
        geom = GeometryMesh.from_file(
            "beam.msh",
            volume_selection=1,
            surface_selection=[{"id": 3}],
            is_obstacle=True,
        )

        payload = geom.to_dict()

        self.assertEqual(payload["mesh"], "beam.msh")
        self.assertEqual(payload["volume_selection"], 1)
        self.assertEqual(payload["surface_selection"], [{"id": 3}])
        self.assertTrue(payload["is_obstacle"])

    def test_other_geometry_named_constructors(self):
        plane = GeometryPlane.obstacle(point=[0.0, 0.0], normal=[0.0, 1.0])
        ground = GeometryGround.obstacle(height=0.25)
        sequence = GeometryMeshSequence.from_files(["a.obj", "b.obj"], fps=24, is_obstacle=True)

        self.assertTrue(plane.to_dict()["is_obstacle"])
        self.assertEqual(ground.to_dict()["height"], 0.25)
        self.assertTrue(ground.to_dict()["is_obstacle"])
        self.assertEqual(sequence.to_dict()["mesh_sequence"], ["a.obj", "b.obj"])
        self.assertEqual(sequence.to_dict()["fps"], 24)
        self.assertTrue(sequence.to_dict()["is_obstacle"])


class OutputTypedBlockTests(unittest.TestCase):
    def test_output_typed_blocks_round_trip(self):
        output = Output(
            directory="out",
            restart_json="restart.json",
            log=OutputLog(path="polyfem.log", quiet=True),
            data=OutputData(solution="u.txt"),
            advanced=OutputAdvanced(save_time_sequence=True, timestep_prefix="impact_step_"),
            reference=OutputReference(solution=["u"]),
            stats=True,
            paraview=ParaviewOutput(
                file_name="impact.pvd",
                options=OutputParaviewOptions(material=True, body_ids=True),
            ),
        )

        payload = output.to_dict()
        round_tripped = Output.from_dict(payload).to_dict()

        self.assertEqual(payload["log"]["path"], "polyfem.log")
        self.assertTrue(payload["log"]["quiet"])
        self.assertEqual(payload["data"]["solution"], "u.txt")
        self.assertEqual(payload["advanced"]["timestep_prefix"], "impact_step_")
        self.assertNotIn("save_time_sequence", payload["advanced"])
        self.assertEqual(payload["reference"]["solution"], ["u"])
        self.assertTrue(payload["paraview"]["options"]["material"])
        self.assertTrue(payload["paraview"]["options"]["body_ids"])
        self.assertEqual(round_tripped["restart_json"], "restart.json")
        self.assertTrue(round_tripped["stats"])

    def test_named_output_constructors_build_history_run_defaults(self):
        paraview = ParaviewOutput.time_sequence(
            file_name="impact.pvd",
            material=True,
            body_ids=True,
            velocity=True,
        )
        payload = paraview.to_dict()

        self.assertEqual(payload["file_name"], "impact.pvd")
        self.assertTrue(payload["options"]["material"])
        self.assertTrue(payload["options"]["body_ids"])
        self.assertTrue(payload["options"]["velocity"])

        output = Output.history_run(
            directory="out",
            json="impact_stats.json",
            log_path="polyfem.log",
            pvd="impact.pvd",
            timestep_prefix="impact_step_",
            requested_fields=["u", "stress"],
            save_vtu=True,
            material=True,
            body_ids=True,
        )

        output_payload = output.to_dict()
        runtime = output.runtime_options()

        self.assertEqual(output_payload["advanced"]["timestep_prefix"], "impact_step_")
        self.assertEqual(output_payload["paraview"]["file_name"], "impact.pvd")
        self.assertTrue(output_payload["paraview"]["options"]["material"])
        self.assertTrue(output_payload["paraview"]["options"]["body_ids"])
        self.assertEqual(runtime["result"]["fields"], ["u", "stress"])

    def test_output_history_progressive_helpers_build_same_shape_more_readably(self):
        output = Output.history(
            directory="out",
            json="impact_stats.json",
            pvd="impact.pvd",
            timestep_prefix="impact_step_",
            vismesh_rel_area=10000000,
            save_vtu=True,
        )
        output.set_log(path="polyfem.log", level="debug", file_level="debug")
        output.enable_paraview_fields(
            material=True,
            body_ids=True,
            velocity=True,
            scalar_values=True,
            tensor_values=True,
        )
        output.request_results(["u", "stress"])

        payload = output.to_dict()
        runtime = output.runtime_options()

        self.assertEqual(payload["directory"], "out")
        self.assertEqual(payload["json"], "impact_stats.json")
        self.assertEqual(payload["log"]["path"], "polyfem.log")
        self.assertEqual(payload["paraview"]["file_name"], "impact.pvd")
        self.assertEqual(payload["advanced"]["timestep_prefix"], "impact_step_")
        self.assertEqual(payload["paraview"]["vismesh_rel_area"], 10000000)
        self.assertTrue(payload["paraview"]["options"]["material"])
        self.assertTrue(payload["paraview"]["options"]["body_ids"])
        self.assertTrue(payload["paraview"]["options"]["velocity"])
        self.assertNotIn("scalar_values", payload["paraview"]["options"])
        self.assertNotIn("tensor_values", payload["paraview"]["options"])
        self.assertEqual(runtime["result"]["fields"], ["u", "stress"])


class ContactTypedBlockTests(unittest.TestCase):
    def test_contact_nested_blocks_round_trip(self):
        contact = Contact(
            enabled=True,
            friction_coefficient=0.2,
            collision_mesh=CollisionMesh(enabled=False),
            adhesion=Adhesion(adhesion_enabled=True, adhesion_strength=0.5),
        )

        payload = contact.to_dict()
        round_tripped = Contact.from_dict(payload)

        self.assertTrue(payload["enabled"])
        self.assertEqual(payload["friction_coefficient"], 0.2)
        self.assertEqual(payload["collision_mesh"]["enabled"], False)
        self.assertTrue(payload["adhesion"]["adhesion_enabled"])
        self.assertAlmostEqual(round_tripped.friction_coefficient, 0.2)
        self.assertIsInstance(round_tripped.collision_mesh, CollisionMesh)
        self.assertIsInstance(round_tripped.adhesion, Adhesion)

    def test_contact_named_constructors(self):
        frictionless = Contact.frictionless(dhat=0.012)
        coulomb = Contact.coulomb(mu=0.3, dhat=0.02)
        adhesive = Contact.adhesive(adhesion_strength=0.5, mu=0.1)

        self.assertTrue(frictionless.enabled)
        self.assertEqual(frictionless.dhat, 0.012)
        self.assertEqual(frictionless.friction_coefficient, 0.0)
        self.assertNotIn("friction_coefficient", frictionless.to_dict())

        self.assertTrue(coulomb.enabled)
        self.assertEqual(coulomb.dhat, 0.02)
        self.assertEqual(coulomb.friction_coefficient, 0.3)
        self.assertEqual(coulomb.mu, 0.3)
        self.assertEqual(coulomb.to_dict()["friction_coefficient"], 0.3)
        self.assertTrue(adhesive.adhesion.adhesion_enabled)
        self.assertEqual(adhesive.adhesion.adhesion_strength, 0.5)
        self.assertEqual(adhesive.friction_coefficient, 0.1)
        self.assertTrue(adhesive.to_dict()["adhesion"]["adhesion_enabled"])


class SimulationConfigTypedBlockRoundTripTests(unittest.TestCase):
    def test_from_json_dict_parses_new_typed_blocks(self):
        full = {
            "pde": "LinearElasticity",
            "discr_order": 1,
            "materials": [{"type": "LinearElasticity", "E": 20, "nu": 0.3}],
            "boundary_conditions": {},
            "geometry": [{"mesh": "beam.msh"}],
            "initial_conditions": {
                "velocity": [{"id": 2, "value": [0.0, 1.0]}],
            },
            "constraints": {
                "soft": [{"weight": 1.5, "data": "soft.h5"}],
            },
            "space": {"discr_order": 1, "pressure_discr_order": 2},
            "tests": {"margin": 1e-4, "time_steps": 2},
            "input": {"data": "state.h5"},
            "solver": {
                "augmented_lagrangian": {"initial_weight": 1e4},
                "contact": {"barrier_stiffness": "adaptive"},
            },
            "contact": {
                "enabled": True,
                "collision_mesh": {"enabled": False},
                "adhesion": {"adhesion_enabled": True},
            },
            "output": {
                "directory": "out",
                "log": {"path": "polyfem.log"},
                "data": {"solution": "u.txt"},
                "advanced": {"save_time_sequence": True, "timestep_prefix": "impact_step_"},
                "reference": {"solution": ["u"]},
                "paraview": {
                    "file_name": "impact.pvd",
                    "options": {"material": True, "body_ids": True},
                },
                "stats": True,
            },
            "time": {"dt": 0.01, "tend": 0.1},
        }

        cfg = SimulationConfig.from_json_dict(full)
        round_tripped = cfg.to_full_json_dict()

        self.assertIsInstance(cfg.initial_conditions, InitialConditions)
        self.assertIsInstance(cfg.constraints, Constraints)
        self.assertIsInstance(cfg.space, Space)
        self.assertIsInstance(cfg.tests, ConfigTests)
        self.assertIsInstance(cfg.input, Input)
        self.assertIsInstance(cfg.output, Output)
        self.assertIsInstance(cfg.output.log, OutputLog)
        self.assertIsInstance(cfg.contact, Contact)
        self.assertIsInstance(cfg.contact.collision_mesh, CollisionMesh)
        self.assertIsInstance(cfg.contact.adhesion, Adhesion)
        self.assertIsInstance(cfg.solver, Solver)
        self.assertIsInstance(cfg.solver.augmented_lagrangian, AugmentedLagrangian)
        self.assertIsInstance(cfg.solver.contact, SolverContactOptions)

        self.assertEqual(round_tripped["initial_conditions"]["velocity"][0]["id"], 2)
        self.assertEqual(round_tripped["constraints"]["soft"][0]["data"], "soft.h5")
        self.assertEqual(round_tripped["space"]["pressure_discr_order"], 2)
        self.assertEqual(round_tripped["tests"]["time_steps"], 2)
        self.assertEqual(round_tripped["input"]["data"], "state.h5")
        self.assertEqual(round_tripped["output"]["data"]["solution"], "u.txt")
        self.assertTrue(round_tripped["contact"]["adhesion"]["adhesion_enabled"])
        self.assertEqual(
            round_tripped["solver"]["contact"]["barrier_stiffness"],
            "adaptive",
        )


class SolverNamedConstructorTests(unittest.TestCase):
    def test_linear_and_nonlinear_named_constructors(self):
        linear = LinearSolver.pardiso_ldlt()
        nonlinear = NonlinearSolver.newton(
            max_iterations=800,
            grad_norm=0.002,
            residual_tolerance=100,
        )

        self.assertEqual(linear.solver_type, "Eigen::PardisoLDLT")
        self.assertEqual(nonlinear.solver_type, "Newton")
        self.assertEqual(nonlinear.max_iterations, 800)
        self.assertEqual(nonlinear.grad_norm, 0.002)
        self.assertEqual(
            nonlinear.to_dict()["Newton"]["residual_tolerance"],
            100,
        )

    def test_solver_newton_contact_constructor(self):
        solver = Solver.newton_contact(
            linear=LinearSolver.pardiso_ldlt(),
            max_iterations=800,
            grad_norm=0.002,
            residual_tolerance=100,
            barrier_stiffness="adaptive",
        )
        payload = solver.to_dict()

        self.assertEqual(payload["linear"]["solver"], "Eigen::PardisoLDLT")
        self.assertEqual(payload["nonlinear"]["solver"], "Newton")
        self.assertEqual(payload["nonlinear"]["max_iterations"], 800)
        self.assertEqual(payload["nonlinear"]["grad_norm"], 0.002)
        self.assertEqual(payload["nonlinear"]["Newton"]["residual_tolerance"], 100)
        self.assertEqual(payload["contact"]["barrier_stiffness"], "adaptive")


class BoundaryAndInitialConditionNamedConstructorTests(unittest.TestCase):
    def test_boundary_conditions_dirichlet_rhs_constructor(self):
        bc = BoundaryConditions.dirichlet_rhs(id=3, value=[0.0, 0.0], rhs=[0.0, 980.0])
        payload = bc.to_dict()

        self.assertEqual(payload["dirichlet_boundary"][0]["id"], 3)
        self.assertEqual(payload["dirichlet_boundary"][0]["value"], [0.0, 0.0])
        self.assertEqual(payload["rhs"], [0.0, 980.0])

    def test_boundary_conditions_neumann_and_periodic_constructors(self):
        neumann = BoundaryConditions.neumann_rhs(id=4, value=[0.0, -1.0], rhs=[0.0, 980.0])
        periodic = BoundaryConditions.periodic(
            tolerance=1e-4,
            correspondence=[{"surface_1": 1, "surface_2": 2}],
            force_zero_mean=True,
        )

        neumann_payload = neumann.to_dict()
        periodic_payload = periodic.to_dict()

        self.assertEqual(neumann_payload["neumann_boundary"][0]["id"], 4)
        self.assertEqual(neumann_payload["neumann_boundary"][0]["value"], [0.0, -1.0])
        self.assertEqual(neumann_payload["rhs"], [0.0, 980.0])
        self.assertTrue(periodic_payload["periodic_boundary"]["enabled"])
        self.assertEqual(periodic_payload["periodic_boundary"]["tolerance"], 1e-4)
        self.assertTrue(periodic_payload["periodic_boundary"]["force_zero_mean"])

    def test_initial_conditions_velocity_only_constructor(self):
        ic = InitialConditions.velocity_only(id=2, value=[0.0, 0.0])
        payload = ic.to_dict()

        self.assertEqual(payload["velocity"][0]["id"], 2)
        self.assertEqual(payload["velocity"][0]["value"], [0.0, 0.0])

    def test_initial_conditions_solution_and_acceleration_constructors(self):
        solution = InitialConditions.solution_only(id=1, value=[1.0, 2.0])
        acceleration = InitialConditions.acceleration_only(id=3, value=[0.0, -9.8])

        self.assertEqual(solution.to_dict()["solution"][0]["id"], 1)
        self.assertEqual(solution.to_dict()["solution"][0]["value"], [1.0, 2.0])
        self.assertEqual(acceleration.to_dict()["acceleration"][0]["id"], 3)
        self.assertEqual(acceleration.to_dict()["acceleration"][0]["value"], [0.0, -9.8])


class BodyAndSelectionApiTests(unittest.TestCase):
    def test_add_body_auto_assigns_and_aligns_ids(self):
        cfg = SimulationConfig()

        lattice = cfg.add_body(
            name="lattice",
            geometry=GeometryMesh.from_file("lattice.msh"),
            material=NeoHookean.young_poisson(E=20.0, E_unit="MPa", nu=0.45, rho=1100.0, rho_unit="kg/m^3"),
        )
        block = cfg.add_body(
            name="block",
            geometry=GeometryMesh.from_file("block.msh"),
            material=NeoHookean.young_poisson(E=200.0, E_unit="GPa", nu=0.45, rho=7850.0, rho_unit="kg/m^3"),
        )

        self.assertEqual(lattice.volume_id, 1)
        self.assertEqual(block.volume_id, 2)
        self.assertEqual(cfg.materials[0].id, 1)
        self.assertEqual(cfg.materials[1].id, 2)
        self.assertEqual(cfg.geometry.items[0].volume_selection, 1)
        self.assertEqual(cfg.geometry.items[1].volume_selection, 2)

    def test_body_surface_and_initial_condition_helpers_update_config(self):
        cfg = SimulationConfig()
        body = cfg.add_body(
            name="lattice",
            geometry=GeometryMesh.from_file("lattice.msh"),
            material=NeoHookean.young_poisson(E=20.0, E_unit="MPa", nu=0.45),
        )

        body.fix_surface(SurfaceSelection.position(axis=-2, position=1e-4), value=[0.0, 0.0])
        body.set_initial_velocity([1.0, 2.0])

        payload = cfg.to_full_json_dict()

        self.assertEqual(payload["geometry"][0]["surface_selection"][0]["id"], 1)
        self.assertEqual(payload["geometry"][0]["surface_selection"][0]["axis"], -2)
        self.assertEqual(payload["boundary_conditions"]["dirichlet_boundary"][0]["id"], 1)
        self.assertEqual(payload["boundary_conditions"]["dirichlet_boundary"][0]["value"], [0.0, 0.0])
        self.assertEqual(payload["initial_conditions"]["velocity"][0]["id"], 1)
        self.assertEqual(payload["initial_conditions"]["velocity"][0]["value"], [1.0, 2.0])


class UnitsAndTimeNamedConstructorTests(unittest.TestCase):
    def test_units_named_constructors(self):
        explicit = CONFIG_MODULE.Units.set_units(length="cm", mass="g", time="s")
        mapping = CONFIG_MODULE.Units.set_units({"length": "cm", "mass": "g", "time": "s"})
        si = CONFIG_MODULE.Units.si()
        cgs = CONFIG_MODULE.Units.cgs()

        self.assertEqual(explicit.to_dict(), {"length": "cm", "mass": "g", "time": "s"})
        self.assertEqual(mapping.to_dict(), {"length": "cm", "mass": "g", "time": "s"})
        self.assertEqual(si.to_dict(), {"length": "m", "mass": "kg", "time": "s"})
        self.assertEqual(cgs.to_dict(), {"length": "cm", "mass": "g", "time": "s"})

    def test_time_named_constructors(self):
        transient = Time.transient(tend=0.02, dt=0.01)
        bdf = Time.bdf(tend=0.5, dt=0.1, steps=2)
        newmark = Time.implicit_newmark(tend=1.0, dt=0.05, gamma=0.6, beta=0.3)

        self.assertEqual(transient.t0, 0.0)
        self.assertEqual(transient.tend, 0.02)
        self.assertEqual(transient.dt, 0.01)
        self.assertEqual(transient.integrator, "ImplicitEuler")

        self.assertEqual(bdf.integrator.to_dict(), {"type": "BDF", "steps": 2})
        self.assertEqual(newmark.integrator.to_dict(), {"type": "ImplicitNewmark", "gamma": 0.6, "beta": 0.3})


class TimeValidationTests(unittest.TestCase):
    def test_validate_rejects_incomplete_time_block(self):
        cfg = SimulationConfig.linear_elasticity(2100, 0.3)
        cfg.time = Time()
        with self.assertRaisesRegex(ValueError, "time requires at least two"):
            cfg.validate()

    def test_validate_accepts_dt_plus_time_steps(self):
        cfg = SimulationConfig.linear_elasticity(2100, 0.3)
        cfg.time = Time(dt=0.01, time_steps=10)
        cfg.validate()


if __name__ == "__main__":
    unittest.main()
