"""PolyFEM handle, ID and JSON export behavior of the generated builder."""
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

from polyfempy._selection_refs import SelectionPool
from polyfempy._model_builder import SelectionHandle
from polyfempy.generated_api import generated_api
from polyfempy.generated_api.generated_api import ModelBuilder

ROOT = Path(__file__).resolve().parents[1]
RULES = json.loads((ROOT / "generator-config/builder.json").read_text())["builder"]
GENERATED_GROUPS = json.loads(Path(generated_api.__file__).with_name("builder_api_manifest.json").read_text())["namespaces"]


def manifest_tools():
    from unittest.mock import patch
    with patch.object(sys, "path", [str(ROOT / "python-from-jse"), *sys.path]):
        return importlib.import_module("generator.model_builder")


def make_builder(api, **kwargs):
    return ModelBuilder(api, namespaces=GENERATED_GROUPS, **kwargs)


def _generator_helpers():
    import importlib.util
    helper = ROOT / "python-from-jse/tests/generator_test_helpers.py"
    spec = importlib.util.spec_from_file_location("polyfem_generator_test_helpers", helper)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class ProjectApiGenerationTests(unittest.TestCase):
    def test_model_geometry_methods_follow_new_spec_types(self):
        generator = _generator_helpers().import_generator("json_to_tree_new_geometry")
        root = generator.build_tree([
            {"pointer": "/", "type": "object", "required": ["geometry"]},
            {"pointer": "/geometry", "type": "list"},
            {
                "pointer": "/geometry/*", "type": "object", "type_name": "VoxelMesh",
                "required": ["type", "mesh"], "optional": ["is_obstacle"],
            },
            {"pointer": "/geometry/*/type", "type": "string", "options": ["VoxelMesh"]},
            {"pointer": "/geometry/*/mesh", "type": "string"},
            {"pointer": "/geometry/*/is_obstacle", "type": "bool"},
        ])
        generated = _generator_helpers().module_from_generated_text(generator.generated_class_text(root))
        api = _generator_helpers().module_from_generated_api_text(generator.generated_api_text(root, {"builder": RULES}), generated.Root)
        model = api.model(relationships=[], selection_helper_rules={}, namespaces=GENERATED_GROUPS)
        model.voxel_mesh(mesh="grid.mesh", is_obstacle=True)
        self.assertEqual(
            [api.voxel_mesh(mesh="grid.mesh", is_obstacle=True).as_dict()],
            model.config().as_dict()["geometry"],
        )

    def test_generated_api_exposes_model_builder_factory(self):
        generator = _generator_helpers().import_generator("json_to_tree_generated_api_model_builder")
        root = generator.build_tree([
            {
                "pointer": "/",
                "type": "object",
                "required": ["geometry", "materials"],
            },
            {
                "pointer": "/geometry",
                "type": "list",
            },
            {
                "pointer": "/geometry/*",
                "type": "object",
                "type_name": "Mesh",
                "required": ["type", "mesh"],
                "optional": ["volume_selection"],
            },
            {
                "pointer": "/geometry/*/type",
                "type": "string",
                "options": ["mesh"],
            },
            {
                "pointer": "/geometry/*/mesh",
                "type": "string",
            },
            {
                "pointer": "/geometry/*/volume_selection",
                "type": "int",
            },
            {
                "pointer": "/materials",
                "type": "list",
            },
            {
                "pointer": "/materials/*",
                "type": "object",
                "type_name": "NeoHookean",
                "required": ["type", "E", "nu"],
                "optional": ["id"],
            },
            {
                "pointer": "/materials/*/type",
                "type": "string",
                "options": ["NeoHookean"],
            },
            {
                "pointer": "/materials/*/E",
                "type": "float",
            },
            {
                "pointer": "/materials/*/nu",
                "type": "float",
            },
            {
                "pointer": "/materials/*/id",
                "type": "int",
            },
        ])
        generated_module = _generator_helpers().module_from_generated_text(generator.generated_class_text(root))
        api_module = _generator_helpers().module_from_generated_api_text(
            generator.generated_api_text(root, {"builder": RULES}),
            generated_module.Root,
        )

        relationships = [
            {
                "namespace": "volume",
                "producer": "geometry[*].volume_selection",
                "consumer": "materials[*].id",
                "builder_api": "body.material",
                "status": "required",
            }
        ]
        model = api_module.model(relationships=relationships, namespaces=GENERATED_GROUPS)
        body = model.mesh(mesh="beam.msh")
        body.material(api_module.neo_hookean(E=20.0, nu=0.45))

        explicit_cfg = api_module.config(
            geometry=model.geometry(),
            materials=model.materials(),
        )
        cfg = api_module.config(model=model)

        self.assertIs(model.__class__, api_module.ModelBuilder)
        self.assertEqual(explicit_cfg.as_dict(), cfg.as_dict())
        self.assertEqual(1, cfg.as_dict()["geometry"][0]["volume_selection"])
        self.assertEqual(1, cfg.as_dict()["materials"][0]["id"])
        self.assertNotIn("uuid", repr(cfg.as_dict()))

        new_model = api_module.model(relationships=relationships, namespaces=GENERATED_GROUPS)
        new_body = api_module.body(model=new_model)
        mesh_descriptor = api_module.mesh(mesh="beam.msh")
        new_body.mesh(mesh_descriptor)
        new_body.material(api_module.neo_hookean(E=20.0, nu=0.45))
        new_cfg = api_module.config(model=new_model)

        self.assertEqual(cfg.as_dict(), new_cfg.as_dict())
        self.assertNotIn("volume_selection", mesh_descriptor.as_dict())


class FakeGeneratedApi:
    __file__: str

    def config(self, **kwargs):
        return dict(kwargs)

    def mesh(self, **kwargs):
        return dict(kwargs)

    def neo_hookean(self, **kwargs):
        result = {"type": "NeoHookean"}
        result.update(kwargs)
        return result


class FakeGeneratedObject:
    def __init__(self, payload):
        self._payload = dict(payload)

    def as_dict(self):
        return dict(self._payload)


POLYFEM_RELATIONSHIPS = [
    {
        "namespace": "volume",
        "producer": "geometry[*].volume_selection",
        "consumer": "materials[*].id",
        "builder_api": "body.material",
        "status": "required",
    },
    {
        "namespace": "volume",
        "producer": "geometry[*].volume_selection",
        "consumer": "initial_conditions.velocity[*].id",
        "builder_api": "body.velocity",
        "status": "required",
    },
    {
        "namespace": "volume",
        "producer": "geometry[*].volume_selection",
        "consumer": "initial_conditions.solution[*].id",
        "builder_api": "body.solution",
        "status": "required",
    },
    {
        "namespace": "volume",
        "producer": "geometry[*].volume_selection",
        "consumer": "initial_conditions.acceleration[*].id",
        "builder_api": "body.acceleration",
        "status": "required",
    },
    {
        "namespace": "volume",
        "producer": "geometry[*].volume_selection",
        "consumer": "space.discr_order[*].id",
        "builder_api": "body.discr_order",
        "status": "required",
    },
    {
        "namespace": "surface",
        "producer": "geometry[*].surface_selection",
        "consumer": "boundary_conditions.dirichlet_boundary[*].id",
        "builder_api": "surface.dirichlet",
        "status": "required",
    },
    {
        "namespace": "surface",
        "producer": "geometry[*].surface_selection",
        "consumer": "boundary_conditions.neumann_boundary[*].id",
        "builder_api": "surface.neumann",
        "status": "required",
    },
    {
        "namespace": "surface",
        "producer": "geometry[*].surface_selection",
        "consumer": "boundary_conditions.normal_aligned_neumann_boundary[*].id",
        "builder_api": "surface.normal_aligned_neumann",
        "status": "required",
    },
    {
        "namespace": "surface",
        "producer": "geometry[*].surface_selection",
        "consumer": "boundary_conditions.pressure_boundary[*].id",
        "builder_api": "surface.pressure",
        "status": "required",
    },
    {
        "namespace": "surface",
        "producer": "geometry[*].surface_selection",
        "consumer": "boundary_conditions.pressure_cavity[*].id",
        "builder_api": "surface.pressure_cavity",
        "status": "required",
    },
    {
        "namespace": "surface",
        "producer": "geometry[*].surface_selection",
        "consumer": "boundary_conditions.obstacle_displacements[*].id",
        "builder_api": "surface.obstacle_displacement",
        "status": "required",
    },
    {
        "namespace": "point",
        "producer": "geometry[*].point_selection",
        "consumer": "boundary_conditions.neumann_boundary[*].id",
        "builder_api": "point.neumann",
        "status": "required",
    },
]
POLYFEM_SELECTION_HELPER_RULES = {
    "surface_axis": {
        "namespace": "surface",
        "field": "surface_selection",
        "allowed_fields": ("id", "axis", "position"),
    },
    "point_box": {
        "namespace": "point",
        "field": "point_selection",
        "allowed_fields": ("id", "box"),
    },
}


class SelectionPoolTests(unittest.TestCase):
    def test_selection_pool_allocates_uuid_and_namespace_backend_ids(self):
        pool = SelectionPool(GENERATED_GROUPS)

        volume = pool.allocate("volume", name="body")
        surface = pool.allocate("surface", name="fixed")
        second_volume = pool.allocate("volume", name="other")

        self.assertEqual("volume", volume.namespace)
        self.assertEqual(1, volume.backend_id)
        self.assertEqual(1, surface.backend_id)
        self.assertEqual(2, second_volume.backend_id)
        self.assertNotEqual(volume.uuid, surface.uuid)


class ModelBuilderTests(unittest.TestCase):
    def test_existing_geometry_rule_ids_are_reserved(self):
        model = make_builder(
            FakeGeneratedApi(), relationships=POLYFEM_RELATIONSHIPS,
            selection_helper_rules=POLYFEM_SELECTION_HELPER_RULES,
        )
        model.mesh(
            mesh="tagged.msh",
            volume_selection=[{"id": 4, "axis": 1, "position": 0}],
            point_selection=[{"id": 8, "box": [[0, 0], [1, 1]]}],
        )
        body = model.mesh(mesh="auto.msh")
        body.material({"type": "NeoHookean", "E": 1000, "nu": 0.3})
        body.surface_all().dirichlet(value=[0, 0])
        self.assertGreater(model.geometry()[1]["volume_selection"], 4)
        self.assertGreater(model.geometry()[1]["surface_selection"], 8)

    def test_repeated_selection_does_not_duplicate_geometry_rules(self):
        model = make_builder(
            FakeGeneratedApi(), relationships=POLYFEM_RELATIONSHIPS,
            selection_helper_rules=POLYFEM_SELECTION_HELPER_RULES,
        )
        body = model.mesh(mesh="part.msh")
        first = body.surface_axis(axis=1, position=0)
        second = body.surface_axis(axis=1, position=0)
        assert isinstance(first, SelectionHandle)
        assert isinstance(second, SelectionHandle)
        self.assertEqual(first.selection_ref.backend_id, second.selection_ref.backend_id)
        self.assertEqual(1, len(model.geometry()[0]["surface_selection"]))
        whole_body = model.mesh(mesh="whole.msh")
        first = whole_body.dirichlet(value=[0, 0])
        second = whole_body.surface_all()
        assert isinstance(first, SelectionHandle)
        self.assertEqual(first.selection_ref.backend_id, second.selection_ref.backend_id)

    def test_bind_rejects_foreign_refs_with_a_shared_pool(self):
        pool = SelectionPool(GENERATED_GROUPS)
        models = [
            make_builder(
                FakeGeneratedApi(), relationships=POLYFEM_RELATIONSHIPS,
                selection_helper_rules={}, selection_pool=pool,
            )
            for _ in range(2)
        ]
        foreign = models[0].mesh(mesh="foreign.msh").surface_all()
        local = models[1].mesh(mesh="local.msh").surface_all()
        before = models[1].config()
        with self.assertRaisesRegex(ValueError, "this model"):
            models[1].bind("surface.dirichlet", foreign.selection_ref, {"value": [1, 0]})
        self.assertEqual(before, models[1].config())
        models[1].bind("surface.dirichlet", local.selection_ref, {"value": [0, 0]})
        self.assertEqual(
            local.selection_ref.backend_id,
            models[1].boundary_conditions()["dirichlet_boundary"][0]["id"],
        )

    def test_boundary_settings_persist_and_keep_the_dictionary_api(self):
        model = make_builder(
            FakeGeneratedApi(), relationships=POLYFEM_RELATIONSHIPS,
            selection_helper_rules={},
        )
        model.mesh(mesh="part.msh").surface_all().dirichlet(value=[0, 0])
        boundary = model.boundary_conditions(rhs=[0, -1])
        self.assertIsInstance(boundary, dict)
        self.assertEqual(boundary, model.config()["boundary_conditions"])
        self.assertEqual(model.config(), model.config(boundary_conditions=boundary))
        self.assertEqual(1, len(model.config()["boundary_conditions"]["dirichlet_boundary"]))
        model.boundary_conditions(rhs=[0, -2])
        self.assertEqual([0, -2], model.config()["boundary_conditions"]["rhs"])

    def test_config_does_not_save_one_time_boundary_settings(self):
        model = make_builder(
            FakeGeneratedApi(), relationships=POLYFEM_RELATIONSHIPS,
            selection_helper_rules={},
        )
        before = model.config()
        self.assertEqual([0, -1], model.config(rhs=[0, -1])["boundary_conditions"]["rhs"])
        self.assertEqual(before, model.config())
        self.assertEqual({}, model.boundary_conditions())

    def test_boundary_conflict_does_not_replace_previous_settings(self):
        relationships = [{
            "namespace": "volume", "producer": "geometry[*].volume_selection",
            "consumer": "boundary_conditions.rhs[*].id", "builder_api": "body.rhs",
            "status": "required",
        }]
        model = make_builder(
            FakeGeneratedApi(), relationships=relationships, selection_helper_rules={},
        )
        model.mesh(mesh="part.msh").rhs(value=[0, -1])
        before = model.config()
        with self.assertRaises(TypeError):
            model.boundary_conditions(rhs=[0, -2])
        self.assertEqual(before, model.config())

    def test_model_builder_loads_relationships_from_api_adjacent_manifest(self):

        with tempfile.TemporaryDirectory() as tmpdir:
            api = FakeGeneratedApi()
            api.__file__ = str(Path(tmpdir) / "generated_api.py")
            (Path(tmpdir) / "builder_api_manifest.json").write_text(
                json.dumps({
                    "version": 1,
                    "model_entry": "polyfem.model",
                    "relationships": [
                        {
                            "namespace": "volume",
                            "producer": "geometry[*].volume_selection",
                            "consumer": "materials[*].id",
                            "builder_api": "body.material",
                        }
                    ],
                }),
                encoding="utf-8",
            )

            model = make_builder(api, selection_helper_rules={})
            body = model.mesh(mesh="beam.msh")
            body.material(api.neo_hookean(E=20.0, nu=0.45))

        self.assertEqual(
            [{"type": "NeoHookean", "E": 20.0, "nu": 0.45, "id": 1}],
            model.materials(),
        )

    def test_model_builder_loads_selection_helpers_from_api_adjacent_manifest(self):

        with tempfile.TemporaryDirectory() as tmpdir:
            api = FakeGeneratedApi()
            api.__file__ = str(Path(tmpdir) / "generated_api.py")
            generated_dir = Path(tmpdir)
            (generated_dir / "builder_api_manifest.json").write_text(
                json.dumps({
                    "version": 1,
                    "model_entry": "polyfem.model",
                    "relationships": [
                        {
                            "namespace": "surface",
                            "producer": "geometry[*].surface_selection",
                            "consumer": "boundary_conditions.dirichlet_boundary[*].id",
                            "builder_api": "surface.dirichlet",
                        }
                    ],
                }),
                encoding="utf-8",
            )
            (generated_dir / "class_tree_manifest.json").write_text(
                json.dumps([
                    {
                        "class_path": "Root.Geometry.Mesh.Surface_selection",
                        "params": ["axis"],
                    },
                    {
                        "class_path": "Root.Geometry.Mesh.Surface_selection.Axis",
                        "schema_path": "geometry[*].surface_selection",
                        "variant": "axis",
                        "params": ["id", "axis", "position"],
                    },
                ]),
                encoding="utf-8",
            )

            # Runtime reads the validated rules shipped in the builder manifest.
            manifest_file = generated_dir / "builder_api_manifest.json"
            manifest = json.loads(manifest_file.read_text())
            class_tree = json.loads((generated_dir / "class_tree_manifest.json").read_text())
            manifest["selection_rules"] = manifest_tools().selection_helper_rules_from_manifest(
                class_tree, manifest["relationships"],
            )
            manifest_file.write_text(json.dumps(manifest))
            model = make_builder(api)
            body = model.mesh(mesh="beam.msh")
            body.surface_axis(axis=1, position=0.5).dirichlet(value=[0, 0])

        self.assertEqual(
            {"dirichlet_boundary": [{"value": [0, 0], "id": 1}]},
            model.boundary_conditions(),
        )

    def test_body_material_uses_relationship_map_and_exports_backend_ids_only(self):
        api = FakeGeneratedApi()
        model = make_builder(
            api,
            relationships=POLYFEM_RELATIONSHIPS,
            selection_helper_rules=POLYFEM_SELECTION_HELPER_RULES,
        )

        body = model.mesh(mesh="beam.msh")
        material = api.neo_hookean(E=20.0, nu=0.45)
        body.material(material)

        self.assertEqual(
            [{"mesh": "beam.msh", "volume_selection": 1}],
            model.geometry(),
        )
        self.assertEqual(
            [{"type": "NeoHookean", "E": 20.0, "nu": 0.45, "id": 1}],
            model.materials(),
        )
        self.assertNotIn("uuid", repr(model.geometry()))
        self.assertNotIn("uuid", repr(model.materials()))
        self.assertNotIn("name", repr(model.geometry()))
        self.assertNotIn("name", repr(model.materials()))

    def test_body_methods_are_provided_from_relationship_map(self):
        relationships = [
            {
                "namespace": "volume",
                "producer": "geometry[*].volume_selection",
                "consumer": "materials[*].id",
                "builder_api": "body.alt_material",
                "status": "required",
            }
        ]
        model = make_builder(FakeGeneratedApi(), relationships=relationships)
        body = model.mesh(mesh="beam.msh")

        body.alt_material({"type": "NeoHookean", "E": 20.0, "nu": 0.45})

        self.assertEqual(
            [{"type": "NeoHookean", "E": 20.0, "nu": 0.45, "id": 1}],
            model.materials(),
        )

    def test_material_binding_does_not_mutate_user_material_object(self):
        model = make_builder(
            FakeGeneratedApi(),
            relationships=POLYFEM_RELATIONSHIPS,
            selection_helper_rules=POLYFEM_SELECTION_HELPER_RULES,
        )
        body = model.mesh(mesh="beam.msh")
        material = {"type": "NeoHookean", "E": 20.0, "nu": 0.45}

        body.material(material)

        self.assertEqual({"type": "NeoHookean", "E": 20.0, "nu": 0.45}, material)
        self.assertEqual(1, model.materials()[0]["id"])

    def test_material_binding_accepts_generated_object_payload(self):
        model = make_builder(
            FakeGeneratedApi(),
            relationships=POLYFEM_RELATIONSHIPS,
            selection_helper_rules=POLYFEM_SELECTION_HELPER_RULES,
        )
        body = model.mesh(mesh="beam.msh")
        material = FakeGeneratedObject({"type": "NeoHookean", "E": 20.0, "nu": 0.45})

        body.material(material)

        self.assertEqual(
            [{"type": "NeoHookean", "E": 20.0, "nu": 0.45, "id": 1}],
            model.materials(),
        )

    def test_body_velocity_uses_volume_relationship_and_exports_initial_conditions(self):
        relationships = [
            {
                "namespace": "volume",
                "producer": "geometry[*].volume_selection",
                "consumer": "initial_conditions.velocity[*].id",
                "builder_api": "body.velocity",
                "status": "required",
            }
        ]
        model = make_builder(FakeGeneratedApi(), relationships=relationships)
        body = model.mesh(mesh="beam.msh")

        body.velocity({"value": [67.0, 0.0]})

        self.assertEqual(
            {"velocity": [{"value": [67.0, 0.0], "id": 1}]},
            model.initial_conditions(),
        )
        self.assertNotIn("uuid", repr(model.initial_conditions()))
        self.assertNotIn("name", repr(model.initial_conditions()))

    def test_default_body_initial_condition_methods_export_volume_ids(self):
        model = make_builder(
            FakeGeneratedApi(),
            relationships=POLYFEM_RELATIONSHIPS,
            selection_helper_rules=POLYFEM_SELECTION_HELPER_RULES,
        )
        body = model.mesh(mesh="beam.msh")

        body.solution({"value": [0.0, 0.0]})
        body.velocity({"value": [67.0, 0.0]})
        body.acceleration({"value": [0.0, 0.0]})

        self.assertEqual(
            {
                "solution": [{"value": [0.0, 0.0], "id": 1}],
                "velocity": [{"value": [67.0, 0.0], "id": 1}],
                "acceleration": [{"value": [0.0, 0.0], "id": 1}],
            },
            model.initial_conditions(),
        )

    def test_default_body_discr_order_exports_space_volume_id(self):
        model = make_builder(
            FakeGeneratedApi(),
            relationships=POLYFEM_RELATIONSHIPS,
            selection_helper_rules=POLYFEM_SELECTION_HELPER_RULES,
        )
        model.mesh(mesh="sphere.msh")
        wall = model.mesh(mesh="cube.msh")

        wall.discr_order(order=2)

        self.assertEqual(
            {"discr_order": [{"order": 2, "id": 2}]},
            model.space(),
        )
        self.assertEqual(
            [
                {"mesh": "sphere.msh"},
                {"mesh": "cube.msh", "volume_selection": 2},
            ],
            model.geometry(),
        )

    def test_explicit_volume_selection_is_reused_for_material_id(self):
        model = make_builder(
            FakeGeneratedApi(),
            relationships=POLYFEM_RELATIONSHIPS,
            selection_helper_rules=POLYFEM_SELECTION_HELPER_RULES,
        )

        body = model.mesh(mesh="beam.msh", volume_selection=7)
        body.material({"type": "NeoHookean", "E": 20.0, "nu": 0.45})
        next_body = model.mesh(mesh="other.msh")

        self.assertEqual(7, model.geometry()[0]["volume_selection"])
        self.assertEqual(7, model.materials()[0]["id"])
        assert next_body.volume_ref is not None
        self.assertEqual(8, next_body.volume_ref.backend_id)

    def test_surface_axis_selection_binds_boundary_conditions(self):
        model = make_builder(
            FakeGeneratedApi(),
            relationships=POLYFEM_RELATIONSHIPS,
            selection_helper_rules=POLYFEM_SELECTION_HELPER_RULES,
        )
        wall = model.mesh(mesh="wall.obj")

        fixed = wall.surface_axis(axis=1, position=0.1025)
        loaded = wall.surface_axis(axis=-1, position=-0.1025)
        fixed.dirichlet(value=[0, 0])
        loaded.neumann(value=[0, -10])
        loaded.normal_aligned_neumann(value=5)
        loaded.pressure(value=100)
        loaded.pressure_cavity(value=200)
        loaded.obstacle_displacement(value=[0, 0])

        self.assertEqual(
            [
                {
                    "mesh": "wall.obj",
                    "surface_selection": [
                        {"id": 1, "axis": 1, "position": 0.1025},
                        {"id": 2, "axis": -1, "position": -0.1025},
                    ],
                }
            ],
            model.geometry(),
        )
        self.assertEqual(
            {
                "dirichlet_boundary": [{"value": [0, 0], "id": 1}],
                "neumann_boundary": [{"value": [0, -10], "id": 2}],
                "normal_aligned_neumann_boundary": [{"value": 5, "id": 2}],
                "pressure_boundary": [{"value": 100, "id": 2}],
                "pressure_cavity": [{"value": 200, "id": 2}],
                "obstacle_displacements": [{"value": [0, 0], "id": 2}],
            },
            model.boundary_conditions(),
        )

    def test_unbound_geometry_mesh_can_create_surface_selection_without_volume_id(self):
        model = make_builder(
            FakeGeneratedApi(),
            relationships=POLYFEM_RELATIONSHIPS,
            selection_helper_rules=POLYFEM_SELECTION_HELPER_RULES,
        )
        wall = model.mesh(mesh="wall.obj")

        fixed = wall.surface_all(id=3)
        fixed.dirichlet(value=[0, 0])

        self.assertEqual(
            [
                {
                    "mesh": "wall.obj",
                    "surface_selection": 3,
                }
            ],
            model.geometry(),
        )
        self.assertEqual(
            {
                "dirichlet_boundary": [{"value": [0, 0], "id": 3}],
            },
            model.boundary_conditions(),
        )

    def test_selection_helper_can_preserve_single_object_shape(self):
        model = make_builder(
            FakeGeneratedApi(),
            relationships=POLYFEM_RELATIONSHIPS,
            selection_helper_rules=POLYFEM_SELECTION_HELPER_RULES,
        )
        wall = model.mesh(mesh="wall.obj")

        fixed = wall.surface_axis(
            id=3,
            axis="-y",
            position=1e-05,
            append=False,
        )
        fixed.dirichlet(value=[0, 0])

        self.assertEqual(
            [
                {
                    "mesh": "wall.obj",
                    "surface_selection": {
                        "id": 3,
                        "axis": "-y",
                        "position": 1e-05,
                    },
                }
            ],
            model.geometry(),
        )
        self.assertEqual(
            {
                "dirichlet_boundary": [{"value": [0, 0], "id": 3}],
            },
            model.boundary_conditions(),
        )

    def test_matching_surface_helpers_on_different_meshes_bind_independently(self):
        model = make_builder(
            FakeGeneratedApi(),
            relationships=POLYFEM_RELATIONSHIPS,
            selection_helper_rules=POLYFEM_SELECTION_HELPER_RULES,
        )
        ball = model.mesh(mesh="ball.obj")
        wall = model.mesh(mesh="wall.obj")

        ball_fixed = ball.surface_axis(axis=1, position=0.1025)
        wall_fixed = wall.surface_axis(axis=1, position=0.1025)
        ball_fixed.dirichlet(value=[0, 0])
        wall_fixed.dirichlet(value=[1, 0])

        self.assertEqual(
            [
                {
                    "mesh": "ball.obj",
                    "surface_selection": [
                        {"id": 1, "axis": 1, "position": 0.1025},
                    ],
                },
                {
                    "mesh": "wall.obj",
                    "surface_selection": [
                        {"id": 2, "axis": 1, "position": 0.1025},
                    ],
                },
            ],
            model.geometry(),
        )
        self.assertEqual(
            {
                "dirichlet_boundary": [{"value": [0, 0], "id": 1}, {"value": [1, 0], "id": 2}],
            },
            model.boundary_conditions(),
        )

    def test_point_selection_uses_same_boundary_id_pool_as_surface_selection(self):
        model = make_builder(
            FakeGeneratedApi(),
            relationships=POLYFEM_RELATIONSHIPS,
            selection_helper_rules=POLYFEM_SELECTION_HELPER_RULES,
        )
        body = model.mesh(mesh="body.obj")

        side = body.surface_axis(axis=1, position=0.5)
        pin = body.point_box(box=[[0.1, 2, -10], [1, 10, 10]])
        side.dirichlet(value=[0, 0])
        pin.neumann(value=[0, -10])

        self.assertEqual(
            [
                {
                    "mesh": "body.obj",
                    "surface_selection": [
                        {"id": 1, "axis": 1, "position": 0.5},
                    ],
                    "point_selection": [
                        {"id": 2, "box": [[0.1, 2, -10], [1, 10, 10]]},
                    ],
                }
            ],
            model.geometry(),
        )
        self.assertEqual(
            {
                "dirichlet_boundary": [{"value": [0, 0], "id": 1}],
                "neumann_boundary": [{"value": [0, -10], "id": 2}],
            },
            model.boundary_conditions(),
        )

    def test_selection_helpers_are_loaded_from_class_tree_manifest(self):
        manifest = [
            {
                "class_path": "Root.Geometry.Mesh.Surface_selection",
                "params": ["ring"],
            },
            {
                "class_path": "Root.Geometry.Mesh.Surface_selection.Ring",
                "schema_path": "geometry[*].surface_selection",
                "variant": "ring",
                "params": ["id", "radius", "relative"],
            },
            {
                "class_path": "Root.Geometry.Mesh.Point_selection",
                "params": ["list"],
            },
            {
                "class_path": "Root.Geometry.Mesh.Point_selection.List",
                "params": ["ring"],
            },
            {
                "class_path": "Root.Geometry.Mesh.Point_selection.List.Ring",
                "schema_path": "geometry[*].point_selection[*]",
                "variant": "ring",
                "params": ["id", "radius", "relative"],
            },
        ]
        selection_helper_rules = manifest_tools().selection_helper_rules_from_manifest(
            manifest, POLYFEM_RELATIONSHIPS
        )
        self.assertEqual(
            "Root.Geometry.Mesh.Surface_selection.Ring",
            selection_helper_rules["surface_ring"]["class_path"],
        )
        self.assertEqual(
            "Root.Geometry.Mesh.Point_selection.List.Ring",
            selection_helper_rules["point_ring"]["class_path"],
        )
        model = make_builder(
            FakeGeneratedApi(),
            relationships=POLYFEM_RELATIONSHIPS,
            selection_helper_rules=selection_helper_rules,
        )
        body = model.mesh(mesh="body.obj")

        surface = body.surface_ring(radius=0.01)
        point = body.point_ring(radius=0.02)
        surface.dirichlet(value=[0, 0])
        point.neumann(value=[0, -10])

        self.assertEqual(
            [
                {
                    "mesh": "body.obj",
                    "surface_selection": [
                        {"id": 1, "radius": 0.01},
                    ],
                    "point_selection": [
                        {"id": 2, "radius": 0.02},
                    ],
                }
            ],
            model.geometry(),
        )
        self.assertEqual(
            {
                "dirichlet_boundary": [{"value": [0, 0], "id": 1}],
                "neumann_boundary": [{"value": [0, -10], "id": 2}],
            },
            model.boundary_conditions(),
        )

    def test_builder_api_manifest_is_generated_from_relationships_and_class_tree(self):
        relationships = [
            {
                "namespace": "volume",
                "producer": "geometry[*].volume_selection",
                "consumer": "materials[*].id",
                "builder_api": "body.material",
                "status": "required",
            },
            {
                "namespace": "surface",
                "producer": "geometry[*].surface_selection",
                "consumer": "boundary_conditions.dirichlet_boundary[*].id",
                "builder_api": "surface.dirichlet",
                "status": "required",
            },
            {
                "namespace": "point",
                "producer": "geometry[*].point_selection",
                "consumer": "boundary_conditions.neumann_boundary[*].id",
                "builder_api": "point.neumann",
                "status": "required",
            },
        ]
        class_tree_manifest = [
            {
                "class_path": "Root.Geometry.Mesh.Surface_selection",
                "params": ["axis"],
            },
            {
                "class_path": "Root.Geometry.Mesh.Surface_selection.Axis",
                "schema_path": "geometry[*].surface_selection",
                "variant": "axis",
                "params": ["id", "axis", "position"],
            },
            {
                "class_path": "Root.Geometry.Mesh.Point_selection",
                "params": ["list"],
            },
            {
                "class_path": "Root.Geometry.Mesh.Point_selection.List",
                "params": ["axis"],
            },
            {
                "class_path": "Root.Geometry.Mesh.Point_selection.List.Axis",
                "schema_path": "geometry[*].point_selection[*]",
                "variant": "axis",
                "params": ["id", "axis", "position"],
            },
        ]

        manifest = manifest_tools().builder_api_manifest_from_inputs(
            relationships,
            class_tree_manifest, rules=RULES,
        )

        self.assertEqual("generated_api.model", manifest["model_entry"])
        self.assertEqual(["material"], manifest["body_methods"])
        self.assertEqual(["dirichlet"], manifest["selection_methods"]["surface"])
        self.assertEqual(["neumann"], manifest["selection_methods"]["point"])
        self.assertEqual(["surface_axis"], manifest["selection_helpers"]["surface"])
        self.assertEqual(["point_axis"], manifest["selection_helpers"]["point"])
        self.assertEqual(
            ["boundary_conditions", "materials"],
            manifest["config_auto_sections"],
        )

    def test_builder_api_manifest_accepts_project_model_entry_override(self):

        manifest = manifest_tools().builder_api_manifest_from_inputs(
            [],
            [],
            model_entry="polyfem.model",
        )

        self.assertEqual("polyfem.model", manifest["model_entry"])

    def test_model_config_collects_builder_sections_and_rhs(self):
        api = FakeGeneratedApi()
        model = make_builder(
            api,
            relationships=POLYFEM_RELATIONSHIPS,
            selection_helper_rules=POLYFEM_SELECTION_HELPER_RULES,
        )
        body = model.mesh(mesh="beam.msh")
        body.material(api.neo_hookean(E=20.0, nu=0.45))
        body.velocity({"value": [67.0, 0.0]})
        body.surface_axis(axis=1, position=0.5).dirichlet(value=[0, 0])

        cfg = model.config(
            rhs=[0, 0],
            time_tend=0.004,
            time_dt=2e-05,
        )

        self.assertEqual(
            [
                {
                    "mesh": "beam.msh",
                    "volume_selection": 1,
                    "surface_selection": [
                        {"id": 1, "axis": 1, "position": 0.5},
                    ],
                }
            ],
            cfg["geometry"],
        )
        self.assertEqual(
            [{"type": "NeoHookean", "E": 20.0, "nu": 0.45, "id": 1}],
            cfg["materials"],
        )
        self.assertEqual(
            {
                "rhs": [0, 0],
                "dirichlet_boundary": [{"value": [0, 0], "id": 1}],
            },
            cfg["boundary_conditions"],
        )
        self.assertEqual(
            {"velocity": [{"value": [67.0, 0.0], "id": 1}]},
            cfg["initial_conditions"],
        )
        self.assertEqual(0.004, cfg["time_tend"])
        self.assertEqual(2e-05, cfg["time_dt"])
        self.assertNotIn("output", cfg)
        self.assertNotIn("solver", cfg)

    def test_body_mesh_matches_legacy_model_mesh_without_mutating_descriptor(self):
        api = FakeGeneratedApi()

        legacy_model = make_builder(
            api,
            relationships=POLYFEM_RELATIONSHIPS,
            selection_helper_rules=POLYFEM_SELECTION_HELPER_RULES,
        )
        legacy_body = legacy_model.mesh(mesh="beam.msh")
        legacy_body.material(api.neo_hookean(E=20.0, nu=0.45))
        legacy_body.surface_axis(axis=1, position=0.5).dirichlet(value=[0, 0])
        legacy_cfg = legacy_model.config(rhs=[0, 0])

        new_model = make_builder(
            api,
            relationships=POLYFEM_RELATIONSHIPS,
            selection_helper_rules=POLYFEM_SELECTION_HELPER_RULES,
        )
        mesh_descriptor = api.mesh(mesh="beam.msh")
        body = new_model.body()
        body.mesh(mesh_descriptor)
        body.material(api.neo_hookean(E=20.0, nu=0.45))
        body.surface_axis(axis=1, position=0.5).dirichlet(value=[0, 0])
        new_model.rhs([0, 0])
        new_cfg = new_model.config()

        self.assertEqual(legacy_cfg, new_cfg)
        self.assertEqual({"mesh": "beam.msh"}, mesh_descriptor)

    def test_body_requires_mesh_before_body_or_surface_bindings(self):
        model = make_builder(
            FakeGeneratedApi(),
            relationships=POLYFEM_RELATIONSHIPS,
            selection_helper_rules=POLYFEM_SELECTION_HELPER_RULES,
        )
        body = model.body()

        with self.assertRaisesRegex(ValueError, "body.mesh"):
            body.material({"type": "NeoHookean", "E": 20.0, "nu": 0.45})
        with self.assertRaisesRegex(ValueError, "body.mesh"):
            body.surface_all(id=1)

    def test_model_config_preserves_explicit_solver(self):
        explicit_solver = {"linear": {"solver": "custom"}}
        model = make_builder(
            FakeGeneratedApi(),
            relationships=POLYFEM_RELATIONSHIPS,
            selection_helper_rules=POLYFEM_SELECTION_HELPER_RULES,
        )

        cfg = model.config(solver=explicit_solver)

        self.assertIs(explicit_solver, cfg["solver"])

    def test_model_config_preserves_explicit_output(self):
        explicit_output = {"json": "custom.json"}
        model = make_builder(
            FakeGeneratedApi(),
            relationships=POLYFEM_RELATIONSHIPS,
            selection_helper_rules=POLYFEM_SELECTION_HELPER_RULES,
        )

        cfg = model.config(output=explicit_output)

        self.assertIs(explicit_output, cfg["output"])

    def test_model_config_collects_sections_from_relationship_consumers(self):
        relationships = [
            {
                "namespace": "volume",
                "producer": "geometry[*].volume_selection",
                "consumer": "constraints.some_constraint[*].id",
                "builder_api": "body.constraint",
                "status": "required",
            }
        ]
        model = make_builder(
            FakeGeneratedApi(),
            relationships=relationships,
        )
        body = model.mesh(mesh="beam.msh")

        body.constraint({"value": "fixed"})
        cfg = model.config()

        self.assertEqual(
            {"some_constraint": [{"value": "fixed", "id": 1}]},
            cfg["constraints"],
        )

    def test_missing_body_material_relationship_reports_clear_error(self):
        relationships = [
            {
                "namespace": "surface",
                "producer": "geometry[*].surface_selection",
                "consumer": "boundary_conditions.dirichlet_boundary[*].id",
                "builder_api": "surface.dirichlet",
                "status": "required",
            }
        ]
        model = make_builder(FakeGeneratedApi(), relationships=relationships)
        body = model.mesh(mesh="beam.msh")

        with self.assertRaisesRegex(AttributeError, "body.material"):
            body.material({"type": "NeoHookean", "E": 20.0, "nu": 0.45})


if __name__ == "__main__":
    unittest.main()
