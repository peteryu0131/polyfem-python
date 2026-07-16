"""Smoke test for the generated-api example.

The example should use generated_api factories for authoring while still
returning a generated Root-compatible object for the solve contract.
"""

from __future__ import annotations

import importlib.util
import json
import py_compile
import sys
import tempfile
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from polyfempy.api._solve_contract import prepare_canonical_solve_input  # noqa: E402


EXAMPLE_02_PATH = (
    _REPO / "polyfempy" / "new_api" / "example" / "02_forward_solve_generated_api.py"
)
EXAMPLE_03_PATH = (
    _REPO
    / "polyfempy"
    / "new_api"
    / "example"
    / "03_forward_solve_generated_api_shortcuts.py"
)
CLASSIC_CONTACT_GOLF_BALL_PATH = (
    _REPO / "examples" / "classic_example" / "2D" / "contact_2d_golf_ball_generated_api.py"
)
NEW_CLASSIC_CONTACT_GOLF_BALL_DEFORMABLE_WALL_PATH = (
    _REPO
    / "examples"
    / "classic_example"
    / "2D"
    / "new_contact_2d_golf_ball_deformable_wall_generated_api.py"
)
BETTER_CLASSIC_CONTACT_GOLF_BALL_DEFORMABLE_WALL_PATH = (
    _REPO
    / "examples"
    / "classic_example"
    / "2D"
    / "better_contact_2d_golf_ball_deformable_wall_generated_api.py"
)
NEW_BETTER_CLASSIC_CONTACT_GOLF_BALL_DEFORMABLE_WALL_PATH = (
    _REPO
    / "examples"
    / "classic_example"
    / "2D"
    / "new_better_contact_2d_golf_ball_deformable_wall_generated_api.py"
)
CLASSIC_EXAMPLE_ROOT = _REPO / "examples" / "classic_example"
CONTACT_EXAMPLE_ROOT = _REPO / "data" / "contact" / "examples"
BUILDER_STYLE_CLASSIC_EXAMPLES = {
    "better_contact_2d_golf_ball_deformable_wall_generated_api.py",
    "new_better_contact_2d_golf_ball_deformable_wall_generated_api.py",
    "new_contact_2d_golf_ball_deformable_wall_generated_api.py",
}
CLASSIC_2D_SOURCE_BY_EXAMPLE = {
    "contact_2d_5_squares_generated_api.py": "2D/unit-tests/5-squares.json",
    "contact_2d_arch_generated_api.py": "2D/friction/arch.json",
    "contact_2d_card_house_generated_api.py": "2D/friction/card-house.json",
    "contact_2d_circle_mat_generated_api.py": "2D/large-ratios/circle-mat.json",
    "contact_2d_circle_rollers_generated_api.py": "2D/friction/circle-rollers.json",
    "contact_2d_cliff_edges_generated_api.py": "2D/unit-tests/erleben/cliff-edges.json",
    "contact_2d_disk_codim_points_generated_api.py": "2D/codimensional/disk-codim-points.json",
    "contact_2d_edge_edge_generated_api.py": "2D/unit-tests/edge-edge.json",
    "contact_2d_edge_vertex_generated_api.py": "2D/unit-tests/edge-vertex.json",
    "contact_2d_friction_slope_generated_api.py": "2D/static/friction-slope.json",
    "contact_2d_golf_ball_deformable_wall_generated_api.py": (
        "2D/golf-ball-doformable-wall.json"
    ),
    "contact_2d_golf_ball_generated_api.py": "2D/golf-ball.json",
    "contact_2d_high_school_physics_slopetest_mu_0_49_generated_api.py": (
        "2D/friction/high-school-physics-slopetest-mu=0.49.json"
    ),
    "contact_2d_high_school_physics_slopetest_mu_0_50_generated_api.py": (
        "2D/friction/high-school-physics-slopetest-mu=0.50.json"
    ),
    "contact_2d_initial_angular_velocity_generated_api.py": "2D/initial_angular_velocity.json",
    "contact_2d_internal_edges_generated_api.py": "2D/unit-tests/erleben/internal-edges.json",
    "contact_2d_large_mass_ratio_generated_api.py": "2D/large-ratios/large-mass-ratio.json",
    "contact_2d_large_stiffness_ratio_generated_api.py": (
        "2D/large-ratios/large-stiffness-ratio.json"
    ),
    "contact_2d_moving_ground_generated_api.py": "2D/friction/moving-ground.json",
    "contact_2d_pin_cushion_ball_generated_api.py": "2D/codimensional/pin-cushion-ball.json",
    "contact_2d_rotating_slope_generated_api.py": "2D/friction/rotating-slope.json",
    "contact_2d_sliding_spike_convergence_generated_api.py": "2D/convergence/sliding-spike.json",
    "contact_2d_sliding_spike_erleben_generated_api.py": (
        "2D/unit-tests/erleben/sliding-spike.json"
    ),
    "contact_2d_spike_in_crack_generated_api.py": "2D/unit-tests/erleben/spike-in-crack.json",
    "contact_2d_spikes_generated_api.py": "2D/unit-tests/erleben/spikes.json",
    "contact_2d_triangle_corner_generated_api.py": "2D/unit-tests/triangle-corner.json",
    "contact_2d_vertex_edge_generated_api.py": "2D/unit-tests/vertex-edge.json",
    "contact_2d_vertex_vertex_generated_api.py": "2D/unit-tests/vertex-vertex.json",
    "new_contact_2d_golf_ball_deformable_wall_generated_api.py": (
        "2D/golf-ball-doformable-wall.json"
    ),
    "new_better_contact_2d_golf_ball_deformable_wall_generated_api.py": (
        "2D/golf-ball-doformable-wall.json"
    ),
}
EXPECTED_CLASSIC_2D_EXAMPLES = {
    "better_contact_2d_golf_ball_deformable_wall_generated_api.py",
    "contact_2d_5_squares_generated_api.py",
    "contact_2d_arch_generated_api.py",
    "contact_2d_card_house_generated_api.py",
    "contact_2d_circle_mat_generated_api.py",
    "contact_2d_circle_rollers_generated_api.py",
    "contact_2d_cliff_edges_generated_api.py",
    "contact_2d_disk_codim_points_generated_api.py",
    "contact_2d_edge_edge_generated_api.py",
    "contact_2d_edge_vertex_generated_api.py",
    "contact_2d_friction_slope_generated_api.py",
    "contact_2d_golf_ball_deformable_wall_generated_api.py",
    "contact_2d_golf_ball_generated_api.py",
    "contact_2d_high_school_physics_slopetest_mu_0_49_generated_api.py",
    "contact_2d_high_school_physics_slopetest_mu_0_50_generated_api.py",
    "contact_2d_initial_angular_velocity_generated_api.py",
    "contact_2d_internal_edges_generated_api.py",
    "contact_2d_large_mass_ratio_generated_api.py",
    "contact_2d_large_stiffness_ratio_generated_api.py",
    "contact_2d_moving_ground_generated_api.py",
    "contact_2d_pin_cushion_ball_generated_api.py",
    "contact_2d_rotating_slope_generated_api.py",
    "contact_2d_sliding_spike_convergence_generated_api.py",
    "contact_2d_sliding_spike_erleben_generated_api.py",
    "contact_2d_spike_in_crack_generated_api.py",
    "contact_2d_spikes_generated_api.py",
    "contact_2d_triangle_corner_generated_api.py",
    "contact_2d_vertex_edge_generated_api.py",
    "contact_2d_vertex_vertex_generated_api.py",
    "new_better_contact_2d_golf_ball_deformable_wall_generated_api.py",
    "new_contact_2d_golf_ball_deformable_wall_generated_api.py",
}


def slug_example_part(value):
    chars = [ch.lower() if ch.isalnum() else "_" for ch in value]
    slug = "".join(chars).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug


def classic_3d_example_name_for_source(source_rel):
    rel_path = Path(source_rel)
    stem_parts = rel_path.with_suffix("").parts[1:]
    slug = "_".join(slug_example_part(part) for part in stem_parts)
    return f"contact_3d_{slug}_generated_api.py"


def classic_3d_source_paths():
    source_root = CONTACT_EXAMPLE_ROOT / "3D"
    return [
        path.relative_to(CONTACT_EXAMPLE_ROOT).as_posix()
        for path in sorted(source_root.rglob("*.json"))
        if path.name != "common.json"
    ]


CLASSIC_3D_SOURCE_BY_EXAMPLE = {
    classic_3d_example_name_for_source(source_rel): source_rel
    for source_rel in classic_3d_source_paths()
}
EXPECTED_CLASSIC_3D_EXAMPLES = set(CLASSIC_3D_SOURCE_BY_EXAMPLE)


def import_example(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def collect_mesh_paths(value):
    if isinstance(value, dict):
        paths = []
        for key, child in value.items():
            if key in {"mesh", "mesh_sequence"} and isinstance(child, str):
                paths.append(child)
            else:
                paths.extend(collect_mesh_paths(child))
        return paths
    if isinstance(value, list):
        paths = []
        for child in value:
            paths.extend(collect_mesh_paths(child))
        return paths
    return []


def classic_example_paths():
    paths = []
    for folder in ("2D", "3D"):
        root = CLASSIC_EXAMPLE_ROOT / folder
        if root.exists():
            paths.extend(path for path in root.rglob("*.py") if not path.name.startswith("_"))
    return sorted(paths)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def deep_merge(base, override):
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def source_config_for(
    source_rel,
    *,
    vectorize_transform_scalars=True,
    generated_solver_names=True,
):
    source_path = CONTACT_EXAMPLE_ROOT / source_rel
    expected = resolved_source_config(source_path)
    expected = normalize_expected_source(
        expected,
        source_path.parent,
        vectorize_transform_scalars=vectorize_transform_scalars,
        generated_solver_names=generated_solver_names,
    )
    if isinstance(expected.get("materials"), dict):
        expected["materials"] = [expected["materials"]]
    normalize_generated_list_fields(expected)
    return expected


def normalize_generated_list_fields(expected):
    boundary_conditions = expected.get("boundary_conditions")
    if isinstance(boundary_conditions, dict):
        for key in (
            "dirichlet_boundary",
            "neumann_boundary",
            "pressure_boundary",
        ):
            if isinstance(boundary_conditions.get(key), dict):
                boundary_conditions[key] = [boundary_conditions[key]]


def resolved_source_config(source_path):
    data = load_json(source_path)
    common_rel = data.get("common")
    expected = {}
    if common_rel:
        expected = resolved_source_config((source_path.parent / common_rel).resolve())

    data_without_meta = {
        key: value
        for key, value in data.items()
        if key not in {"common", "tests", "patch", "default_params"}
    }
    expected = deep_merge(expected, data_without_meta)
    for patch in data.get("patch", []):
        apply_json_patch(expected, patch)
    return expected


def apply_json_patch(document, patch):
    if patch.get("op") != "replace":
        raise ValueError("unsupported JSON patch operation: %r" % patch)

    parts = [part for part in patch["path"].split("/") if part]
    if not parts:
        raise ValueError("unsupported empty JSON patch path")

    target = document
    for part in parts[:-1]:
        if isinstance(target, list):
            target = target[int(part)]
        else:
            target = target[part]

    last = parts[-1]
    if isinstance(target, list):
        target[int(last)] = patch["value"]
    else:
        target[last] = patch["value"]


def normalize_expected_source(
    value,
    source_dir,
    *,
    vectorize_transform_scalars=True,
    generated_solver_names=True,
):
    if isinstance(value, list):
        return [
            normalize_expected_source(
                item,
                source_dir,
                vectorize_transform_scalars=vectorize_transform_scalars,
                generated_solver_names=generated_solver_names,
            )
            for item in value
        ]
    if not isinstance(value, dict):
        return value

    normalized = {
        key: normalize_expected_source(
            child,
            source_dir,
            vectorize_transform_scalars=vectorize_transform_scalars,
            generated_solver_names=generated_solver_names,
        )
        for key, child in value.items()
    }
    for key in ("mesh", "mesh_sequence"):
        if key in normalized and isinstance(normalized[key], str):
            mesh_path = Path(normalized[key])
            if not mesh_path.is_absolute():
                mesh_path = source_dir / mesh_path
            normalized[key] = str(mesh_path.resolve())

    if vectorize_transform_scalars:
        for vectorized_scalar in ("scale", "rotation"):
            if vectorized_scalar in normalized and isinstance(normalized[vectorized_scalar], (int, float)):
                normalized[vectorized_scalar] = [normalized[vectorized_scalar]]
    if "solver" in normalized and isinstance(normalized["solver"], list):
        normalized["solver"] = normalized["solver"][0]
    if generated_solver_names and "x_delta" in normalized:
        normalized["x_delta_tol"] = normalized.pop("x_delta")
    if generated_solver_names and "grad_norm" in normalized:
        normalized["grad_norm_tol"] = normalized.pop("grad_norm")
    if generated_solver_names and "f_delta" in normalized:
        advanced = normalized.setdefault("advanced", {})
        advanced["f_delta_tol"] = normalized.pop("f_delta")
    if generated_solver_names and "use_grad_norm" in normalized:
        line_search = normalized.setdefault("line_search", {})
        line_search["use_grad_norm_tol"] = normalized.pop("use_grad_norm")
    return normalized


def assert_subset(testcase, expected, actual, path="payload"):
    if isinstance(expected, dict):
        testcase.assertIsInstance(actual, dict, path)
        for key, value in expected.items():
            testcase.assertIn(key, actual, path)
            assert_subset(testcase, value, actual[key], f"{path}.{key}")
        return
    if isinstance(expected, list):
        testcase.assertIsInstance(actual, list, path)
        testcase.assertEqual(len(expected), len(actual), path)
        for index, value in enumerate(expected):
            assert_subset(testcase, value, actual[index], f"{path}[{index}]")
        return
    testcase.assertEqual(expected, actual, path)


class GeneratedApiExampleTests(unittest.TestCase):
    def test_source_config_for_resolves_recursive_common_and_patch(self):
        expected = source_config_for("3D/higher-order/ball-bounce/P2.json")

        self.assertTrue(expected["geometry"][0]["mesh"].endswith("P2.msh"))
        self.assertEqual(2, expected["space"]["discr_order"])
        self.assertEqual([0, 9.81, 0], expected["boundary_conditions"]["rhs"])

    def test_source_config_for_maps_legacy_use_grad_norm(self):
        expected = source_config_for("3D/static/two-cubes.json")

        self.assertEqual(
            1e-5,
            expected["solver"]["nonlinear"]["line_search"]["use_grad_norm_tol"],
        )
        self.assertNotIn("use_grad_norm", expected["solver"]["nonlinear"])

    def test_example_builds_generated_config_for_generated_solve_path(self):
        example = import_example("forward_solve_generated_api", EXAMPLE_02_PATH)

        cfg = example.config_for_workspace(Path("runs/generated_api_smoke"))
        payload = cfg.as_dict()
        canonical = prepare_canonical_solve_input(
            vertices=None,
            cells=None,
            cfg=cfg,
            dtype=None,
        )

        self.assertEqual(canonical.metadata["config_source"], "generated")
        self.assertEqual(canonical.metadata["mesh_source"], "json")
        self.assertEqual(payload["root_path"], str(example.CONFIG_PATH))
        self.assertEqual(payload["materials"][0]["type"], "NeoHookean")
        self.assertEqual(payload["time"]["tend"], 0.02)
        self.assertEqual(payload["output"]["json"], "impact_stats.json")
        self.assertEqual(payload["solver"]["nonlinear"]["grad_norm_tol"], 0.002)
        self.assertEqual(payload["space"]["basis_type"], "Lagrange")
        self.assertNotIn("result", payload["output"])
        self.assertNotIn("fallback", payload["output"])

    def test_example_authors_config_in_python_without_loading_json(self):
        source = EXAMPLE_02_PATH.read_text(encoding="utf-8")

        self.assertNotIn("import json", source)
        self.assertNotIn("json.load", source)
        self.assertNotIn("def _build_generated_api_config", source)
        self.assertNotIn("SIMULATION_TEMPLATE", source)
        self.assertIn("polyfem_config = G.config(", source)
        self.assertLess(
            source.index("polyfem_config = G.config("),
            source.index("def main()"),
        )

    def test_shortcut_example_uses_generated_convenience_api(self):
        example = import_example("forward_solve_generated_api_shortcuts", EXAMPLE_03_PATH)

        workspace = Path("runs/generated_api_shortcut_smoke")
        cfg = example.config_for_workspace(workspace)
        payload = cfg.as_dict()
        canonical = prepare_canonical_solve_input(
            vertices=None,
            cells=None,
            cfg=cfg,
            dtype=None,
        )

        self.assertEqual(canonical.metadata["config_source"], "generated")
        self.assertEqual(canonical.metadata["mesh_source"], "json")
        self.assertEqual(payload["root_path"], str(example.CONFIG_PATH))
        self.assertEqual(1, payload["boundary_conditions"]["dirichlet_boundary"][0]["id"])
        self.assertEqual(
            [0.0, 0.0],
            payload["boundary_conditions"]["dirichlet_boundary"][0]["value"],
        )
        self.assertEqual(
            {"id": 1, "axis": -2, "position": 0.0001, "relative": False},
            payload["geometry"][0]["surface_selection"],
        )

        self.assertEqual("hash_grid", payload["solver"]["contact"]["CCD"]["broad_phase"])
        self.assertEqual(1e-6, payload["solver"]["contact"]["CCD"]["tolerance"])
        self.assertEqual(1_000_000, payload["solver"]["contact"]["CCD"]["max_iterations"])
        self.assertEqual(0.0, payload["time"]["t0"])
        self.assertEqual("Lagrange", payload["space"]["basis_type"])
        self.assertNotIn("advanced", payload["space"])

        source = EXAMPLE_03_PATH.read_text(encoding="utf-8")
        self.assertIn("G.boundary_conditions(", source)
        self.assertIn("dirichlet=[", source)
        self.assertIn("G.dirichlet(", source)
        self.assertIn("G.surface_axis(", source)
        self.assertIn("G.output_paraview(", source)
        self.assertNotIn("G.dirichlet_boundary(", source)
        self.assertIn('surface_selection=G.surface_axis(id=1, axis=-2, position=0.0001),', source)
        self.assertIn('G.mesh(mesh="../assets/impact/falling_weight_block.msh", volume_selection=2),', source)
        self.assertIn('units = G.units(length="cm", mass="g", time="s")', source)
        self.assertIn('contact=G.solver_contact(CCD=G.ccd(), barrier_stiffness="adaptive"),', source)
        self.assertIn('time_cfg = G.tend_dt(tend=0.02, dt=0.01, integrator="ImplicitEuler")', source)
        self.assertIn('options=G.options(material=True, body_ids=True, velocity=True),', source)
        self.assertIn('log=G.output_log(level="debug", file_level="debug", path="polyfem.log", quiet=True),', source)
        self.assertIn('advanced=G.output_advanced(timestep_prefix="impact_step_"),', source)
        self.assertIn("contact = G.contact(enabled=True, dhat=0.012)", source)
        self.assertIn("space = G.space(discr_order=1)", source)
        self.assertNotIn("characteristic_length=1.0", source)
        self.assertNotIn("t0=0.0", source)
        self.assertNotIn("broad_phase=", source)
        self.assertNotIn("tolerance=1e-6", source)
        self.assertNotIn("max_iterations=1_000_000", source)
        self.assertNotIn("volume=True", source)
        self.assertNotIn("surface=False", source)
        self.assertNotIn("wireframe=False", source)
        self.assertNotIn("points=False", source)
        self.assertNotIn("high_order_mesh=True", source)
        self.assertNotIn('file_name=""', source)
        self.assertNotIn("skip_frame=1", source)
        self.assertNotIn("scalar_values=True", source)
        self.assertNotIn("tensor_values=True", source)
        self.assertNotIn("save_time_sequence=True", source)
        self.assertNotIn("pressure_discr_order=1", source)
        self.assertNotIn("use_p_ref=False", source)
        self.assertNotIn('basis_type="Lagrange"', source)
        self.assertNotIn('"bc_method": "sample"', source)

    def test_classic_contact_example_is_self_contained_generated_api(self):
        example = import_example(
            "classic_contact_2d_golf_ball_generated_api",
            CLASSIC_CONTACT_GOLF_BALL_PATH,
        )

        workspace = Path("runs/classic_contact_golf_ball_smoke")
        cfg = example.config_for_workspace(workspace)
        payload = cfg.as_dict()
        canonical = prepare_canonical_solve_input(
            vertices=None,
            cells=None,
            cfg=cfg,
            dtype=None,
        )

        self.assertEqual(canonical.metadata["config_source"], "generated")
        self.assertEqual(canonical.metadata["mesh_source"], "json")
        self.assertEqual("", payload["root_path"])
        self.assertNotIn("common", payload)
        self.assertTrue(Path(payload["geometry"][0]["mesh"]).is_absolute())
        self.assertTrue(Path(payload["geometry"][1]["mesh"]).is_absolute())
        self.assertEqual([0.04], payload["geometry"][0]["transformation"]["scale"])
        self.assertEqual([0.105, 0.0], payload["geometry"][1]["transformation"]["translation"])
        self.assertEqual([0.01, 0.4], payload["geometry"][1]["transformation"]["dimensions"])
        self.assertEqual(6.92820323e-5, payload["contact"]["dhat"])
        self.assertEqual(0.004, payload["time"]["tend"])
        self.assertEqual(2e-5, payload["time"]["dt"])
        self.assertEqual([67.0, 0.0], payload["initial_conditions"]["velocity"][0]["value"])
        self.assertEqual([0.0, 0.0], payload["boundary_conditions"]["dirichlet_boundary"][0]["value"])
        self.assertEqual("NeoHookean", payload["materials"][0]["type"])
        self.assertEqual(1150.0, payload["materials"][0]["rho"])
        self.assertEqual(str(workspace), payload["output"]["directory"])
        self.assertEqual(str(workspace / "polyfem.log"), payload["output"]["log"]["path"])

        source = CLASSIC_CONTACT_GOLF_BALL_PATH.read_text(encoding="utf-8")
        self.assertNotIn("import json", source)
        self.assertNotIn("json.load", source)
        self.assertNotIn("root_path=", source)
        self.assertNotIn("CONFIG_PATH", source)
        self.assertIn("G.config(", source)
        self.assertIn("G.mesh(", source)
        self.assertIn("G.neo_hookean(", source)
        self.assertIn("G.dirichlet(", source)

    def test_new_deformable_wall_example_uses_model_builder_and_matches_manual_payload(self):
        manual = import_example(
            "classic_contact_2d_golf_ball_deformable_wall_generated_api",
            CLASSIC_EXAMPLE_ROOT
            / "2D"
            / "contact_2d_golf_ball_deformable_wall_generated_api.py",
        )
        builder = import_example(
            "new_contact_2d_golf_ball_deformable_wall_generated_api",
            NEW_CLASSIC_CONTACT_GOLF_BALL_DEFORMABLE_WALL_PATH,
        )

        workspace = Path("runs/new_contact_2d_golf_ball_deformable_wall")
        manual_payload = manual.config_for_workspace(workspace).as_dict()
        builder_payload = builder.config_for_workspace(workspace).as_dict()

        self.assertEqual(manual_payload["geometry"], builder_payload["geometry"])
        self.assertEqual(manual_payload["materials"], builder_payload["materials"])
        self.assertEqual(
            manual_payload["initial_conditions"],
            builder_payload["initial_conditions"],
        )
        self.assertEqual(
            manual_payload["boundary_conditions"],
            builder_payload["boundary_conditions"],
        )

        source = NEW_CLASSIC_CONTACT_GOLF_BALL_DEFORMABLE_WALL_PATH.read_text(
            encoding="utf-8"
        )
        self.assertIn("model = G.model()", source)
        self.assertIn(".material(", source)
        self.assertIn(".velocity(", source)
        self.assertIn("geometry=model.geometry()", source)
        self.assertIn("materials=model.materials()", source)
        self.assertIn("initial_conditions=model.initial_conditions()", source)

    def test_better_deformable_wall_example_uses_boundary_selection_handles(self):
        better = import_example(
            "better_contact_2d_golf_ball_deformable_wall_generated_api",
            BETTER_CLASSIC_CONTACT_GOLF_BALL_DEFORMABLE_WALL_PATH,
        )

        workspace = Path("runs/better_contact_2d_golf_ball_deformable_wall")
        payload = better.config_for_workspace(workspace).as_dict()

        self.assertEqual(
            [{"id": 1, "axis": 1, "position": 0.1025}],
            payload["geometry"][0]["surface_selection"],
        )
        self.assertEqual(
            [{"id": 2, "axis": 1, "position": 0.1025}],
            payload["geometry"][1]["surface_selection"],
        )
        dirichlet_boundary = payload["boundary_conditions"]["dirichlet_boundary"]
        self.assertEqual([1, 2], [item["id"] for item in dirichlet_boundary])
        self.assertEqual(
            [[0.0, 0.0], [0.0, 0.0]],
            [item["value"] for item in dirichlet_boundary],
        )
        self.assertEqual(1, payload["materials"][0]["id"])
        self.assertEqual(2, payload["materials"][1]["id"])
        self.assertEqual(1, payload["initial_conditions"]["velocity"][0]["id"])

        source = BETTER_CLASSIC_CONTACT_GOLF_BALL_DEFORMABLE_WALL_PATH.read_text(
            encoding="utf-8"
        )
        self.assertIn(".surface_axis(", source)
        self.assertIn(".dirichlet(", source)
        self.assertIn("boundary_conditions=model.boundary_conditions(", source)
        self.assertNotIn("surface_selection=[", source)
        self.assertNotIn("G.dirichlet(", source)
        self.assertNotIn("attach_surface", source)

    def test_new_better_deformable_wall_example_uses_flat_config_shortcuts(self):
        better = import_example(
            "new_better_contact_2d_golf_ball_deformable_wall_generated_api",
            NEW_BETTER_CLASSIC_CONTACT_GOLF_BALL_DEFORMABLE_WALL_PATH,
        )

        workspace = Path("runs/new_better_contact_2d_golf_ball_deformable_wall")
        payload = better.config_for_workspace(workspace).as_dict()

        self.assertEqual(0.004, payload["time"]["tend"])
        self.assertEqual(2e-05, payload["time"]["dt"])
        self.assertTrue(payload["contact"]["enabled"])
        self.assertEqual(6.92820323e-05, payload["contact"]["dhat"])
        self.assertEqual("sim.pvd", payload["output"]["paraview"]["file_name"])
        self.assertEqual("sim.json", payload["output"]["json"])
        self.assertEqual("Eigen::PardisoLDLT", payload["solver"]["linear"]["solver"])
        self.assertEqual("Newton", payload["solver"]["nonlinear"]["solver"])
        self.assertEqual(1e-12, payload["solver"]["nonlinear"]["x_delta_tol"])
        self.assertEqual(1e-05, payload["solver"]["nonlinear"]["grad_norm_tol"])
        self.assertEqual("RobustArmijo", payload["solver"]["nonlinear"]["line_search"]["method"])
        self.assertTrue(payload["solver"]["advanced"]["lump_mass_matrix"])
        self.assertEqual(
            [1, 1],
            [
                item["id"]
                for item in (
                    payload["geometry"][0]["surface_selection"]
                    + payload["geometry"][1]["surface_selection"]
                )
            ],
        )
        self.assertEqual([1], [
            item["id"]
            for item in payload["boundary_conditions"]["dirichlet_boundary"]
        ])
        self.assertEqual(1, payload["materials"][0]["id"])
        self.assertEqual(2, payload["materials"][1]["id"])
        self.assertEqual(1, payload["initial_conditions"]["velocity"][0]["id"])

        source = NEW_BETTER_CLASSIC_CONTACT_GOLF_BALL_DEFORMABLE_WALL_PATH.read_text(
            encoding="utf-8"
        )
        self.assertIn("time_tend=0.004", source)
        self.assertIn("time_dt=2e-05", source)
        self.assertIn("contact_enabled=True", source)
        self.assertIn("contact_dhat=6.92820323e-05", source)
        self.assertIn("polyfem_config = model.config(", source)
        self.assertIn("rhs=[0, 0]", source)
        self.assertNotIn("geometry=model.geometry()", source)
        self.assertNotIn("materials=model.materials()", source)
        self.assertNotIn("boundary_conditions=model.boundary_conditions(", source)
        self.assertNotIn("initial_conditions=model.initial_conditions()", source)
        self.assertIn("solver = polyfem.solver(", source)
        self.assertIn("solver=solver", source)
        self.assertIn("output = polyfem.output(", source)
        self.assertIn("output=output", source)
        self.assertNotIn("time_cfg = G.tend_dt(", source)
        self.assertNotIn("contact = G.contact(", source)
        self.assertNotIn("solve=solver", source)

    def test_new_better_deformable_wall_matches_source_authored_settings(self):
        better = import_example(
            "new_better_contact_2d_golf_ball_deformable_wall_source_parity",
            NEW_BETTER_CLASSIC_CONTACT_GOLF_BALL_DEFORMABLE_WALL_PATH,
        )

        workspace = Path("runs/new_better_contact_2d_golf_ball_source_parity")
        payload = better.config_for_workspace(workspace).as_dict()
        expected = source_config_for("2D/golf-ball-doformable-wall.json")

        assert_subset(self, expected, payload)

    def test_classic_examples_are_grouped_and_smoke_without_backend(self):
        root_python_files = sorted(CLASSIC_EXAMPLE_ROOT.glob("*.py"))
        self.assertEqual([], root_python_files)

        classic_2d_files = {
            path.name
            for path in (CLASSIC_EXAMPLE_ROOT / "2D").glob("*.py")
            if not path.name.startswith("_")
        }
        self.assertEqual(EXPECTED_CLASSIC_2D_EXAMPLES, classic_2d_files)

        classic_3d_files = {
            path.name
            for path in (CLASSIC_EXAMPLE_ROOT / "3D").glob("*.py")
            if not path.name.startswith("_")
        }
        self.assertEqual(EXPECTED_CLASSIC_3D_EXAMPLES, classic_3d_files)

        paths = classic_example_paths()
        self.assertGreaterEqual(len(paths), 1)
        source_by_example = {
            **CLASSIC_2D_SOURCE_BY_EXAMPLE,
            **CLASSIC_3D_SOURCE_BY_EXAMPLE,
        }

        helper_source = (CLASSIC_EXAMPLE_ROOT / "2D" / "_contact_2d_common.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("COMMON_SPEC", helper_source)
        self.assertNotIn("def build_config", helper_source)
        self.assertNotIn("_deep_merge", helper_source)

        for path in paths:
            with self.subTest(path=path.relative_to(_REPO)):
                py_compile.compile(str(path), doraise=True)
                module_name = "classic_example_" + "_".join(path.relative_to(CLASSIC_EXAMPLE_ROOT).with_suffix("").parts)
                example = import_example(module_name, path)

                workspace = Path("runs") / path.stem
                cfg = example.config_for_workspace(workspace)
                payload = cfg.as_dict()
                canonical = prepare_canonical_solve_input(
                    vertices=None,
                    cells=None,
                    cfg=cfg,
                    dtype=None,
                )

                self.assertEqual(canonical.metadata["config_source"], "generated")
                self.assertEqual(canonical.metadata["mesh_source"], "json")
                self.assertNotIn("common", payload)

                mesh_paths = collect_mesh_paths(payload.get("geometry", []))
                self.assertGreaterEqual(len(mesh_paths), 1)
                for mesh_path in mesh_paths:
                    mesh_file = Path(mesh_path)
                    self.assertTrue(mesh_file.is_absolute(), mesh_path)
                    self.assertTrue(mesh_file.exists(), mesh_path)

                source = path.read_text(encoding="utf-8")
                self.assertNotIn("import json", source)
                self.assertNotIn("json.load", source)
                self.assertNotIn("root_path=", source)
                self.assertNotIn("CONFIG_PATH", source)
                self.assertNotIn("EXAMPLE_SPEC", source)
                self.assertNotIn("build_config", source)
                source_rel = source_by_example.get(path.name)
                expected = source_config_for(source_rel) if source_rel else {}
                if path.name in BUILDER_STYLE_CLASSIC_EXAMPLES:
                    if path.name == "new_better_contact_2d_golf_ball_deformable_wall_generated_api.py":
                        self.assertIn("model = polyfem.model()", source)
                    else:
                        self.assertIn("model = G.model()", source)
                    if "polyfem_config = model.config(" not in source:
                        self.assertIn("model.geometry()", source)
                        self.assertIn("model.materials()", source)
                else:
                    if "geometry" in expected:
                        self.assertIn("geometry = [", source)
                    if "materials" in expected:
                        self.assertIn("materials = ", source)
                if "polyfem_config = model.config(" in source:
                    self.assertNotIn("polyfem_config = G.config(", source)
                else:
                    if "contact" in expected:
                        self.assertIn("contact = G.contact(", source)
                    if "solver" in expected:
                        self.assertIn("solver = G.solver(", source)
                    if "output" in expected:
                        self.assertIn("output = G.output(", source)
                    self.assertIn("polyfem_config = G.config(", source)

    def test_classic_2d_examples_preserve_source_json_settings(self):
        for example_name, source_rel in sorted(CLASSIC_2D_SOURCE_BY_EXAMPLE.items()):
            with self.subTest(example=example_name):
                example_path = CLASSIC_EXAMPLE_ROOT / "2D" / example_name
                module_name = "classic_2d_source_match_" + example_path.stem
                example = import_example(module_name, example_path)

                cfg = example.config_for_workspace(Path("runs") / example_path.stem)
                payload = cfg.as_dict()
                expected = source_config_for(source_rel)

                assert_subset(self, expected, payload)

    def test_classic_3d_examples_preserve_source_json_settings(self):
        for example_name, source_rel in sorted(CLASSIC_3D_SOURCE_BY_EXAMPLE.items()):
            with self.subTest(example=example_name):
                example_path = CLASSIC_EXAMPLE_ROOT / "3D" / example_name
                module_name = "classic_3d_source_match_" + example_path.stem
                example = import_example(module_name, example_path)

                cfg = example.config_for_workspace(Path("runs") / example_path.stem)
                payload = cfg.as_dict()
                expected = source_config_for(source_rel)

                assert_subset(self, expected, payload)

    def test_classic_2d_examples_do_not_emit_unrequested_inversion_checks(self):
        for example_name, source_rel in sorted(CLASSIC_2D_SOURCE_BY_EXAMPLE.items()):
            with self.subTest(example=example_name):
                expected = source_config_for(source_rel)
                expected_advanced = expected.get("solver", {}).get("advanced", {})
                if "check_inversion" in expected_advanced:
                    continue

                example_path = CLASSIC_EXAMPLE_ROOT / "2D" / example_name
                module_name = "classic_2d_inversion_check_" + example_path.stem
                example = import_example(module_name, example_path)
                cfg = example.config_for_workspace(Path("runs") / example_path.stem)
                canonical = prepare_canonical_solve_input(
                    vertices=None,
                    cells=None,
                    cfg=cfg,
                    dtype=None,
                )
                advanced = canonical.backend_settings.get("solver", {}).get("advanced", {})

                self.assertNotIn("check_inversion", advanced)

    def test_classic_golf_ball_backend_payload_preserves_source_semantics(self):
        example = import_example(
            "classic_contact_2d_golf_ball_backend_payload",
            CLASSIC_CONTACT_GOLF_BALL_PATH,
        )
        workspace = Path("runs/classic_contact_golf_ball_backend_payload")
        cfg = example.config_for_workspace(workspace)
        canonical = prepare_canonical_solve_input(
            vertices=None,
            cells=None,
            cfg=cfg,
            dtype=None,
        )

        expected = source_config_for(
            "2D/golf-ball.json",
            vectorize_transform_scalars=False,
            generated_solver_names=False,
        )
        backend = canonical.backend_settings

        self.assertEqual(
            expected["geometry"][0]["transformation"]["scale"],
            backend["geometry"][0]["transformation"]["scale"],
        )
        self.assertIsInstance(backend["geometry"][0]["transformation"]["scale"], float)
        self.assertEqual(
            expected["solver"]["advanced"],
            backend["solver"]["advanced"],
        )
        self.assertEqual(
            expected["solver"]["nonlinear"]["x_delta"],
            backend["solver"]["nonlinear"]["x_delta"],
        )
        self.assertEqual(
            expected["solver"]["nonlinear"]["grad_norm"],
            backend["solver"]["nonlinear"]["grad_norm"],
        )
        self.assertNotIn("x_delta_tol", backend["solver"]["nonlinear"])
        self.assertNotIn("grad_norm_tol", backend["solver"]["nonlinear"])
        self.assertNotIn("rel_grad_norm_tol", backend["solver"]["nonlinear"])
        self.assertNotIn("newton_decrement_tol", backend["solver"]["nonlinear"])
        self.assertNotIn("rel_x_delta_tol", backend["solver"]["nonlinear"])
        self.assertNotIn("norm_type", backend["solver"]["nonlinear"])
        for unrequested_default in (
            "cache_size",
            "lagged_regularization_weight",
            "lagged_regularization_iterations",
            "check_inversion",
            "jacobian_threshold",
            "characteristic_length",
            "characteristic_force_density",
        ):
            self.assertNotIn(unrequested_default, backend["solver"]["advanced"])

    def test_classic_golf_ball_backend_build_basis_smoke(self):
        try:
            import polyfempy as pf
        except Exception as exc:  # pragma: no cover - defensive import guard
            self.skipTest(f"polyfempy import failed: {exc}")

        if not getattr(pf, "cpp_backend_available", lambda: False)():
            self.skipTest(f"C++ backend unavailable: {pf.cpp_backend_error()}")

        from polyfempy.polyfempy import Solver

        example = import_example(
            "classic_contact_2d_golf_ball_backend_smoke",
            CLASSIC_CONTACT_GOLF_BALL_PATH,
        )
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "classic_contact_golf_ball_backend_smoke"
            cfg = example.config_for_workspace(workspace)
            canonical = prepare_canonical_solve_input(
                vertices=None,
                cells=None,
                cfg=cfg,
                dtype=None,
            )

            solver = Solver()
            solver.set_settings(json.dumps(canonical.backend_settings), strict_validation=False)
            solver.set_log_level(5)
            solver.load_mesh_from_settings()
            solver.build_basis()


if __name__ == "__main__":
    unittest.main()
