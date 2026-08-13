"""Smoke and source-parity tests for classic generated API examples."""

from __future__ import annotations

import copy
import importlib.util
import json
import py_compile
import sys
from pathlib import Path
from typing import Any

import pytest

from polyfempy.runtime._solve_contract import prepare_canonical_solve_input
from polyfempy.generated_api import generated_api as polyfem


ROOT = Path(__file__).resolve().parents[1]
CLASSIC_EXAMPLES = ROOT / "examples" / "classic_example"
POLYFEM_DATA_CONTACT = ROOT / "polyfem-data" / "contact"
POLYFEM_DATA_EXAMPLES = POLYFEM_DATA_CONTACT / "examples"

EXAMPLE_CASES = [
    (
        CLASSIC_EXAMPLES / "2D" / "contact_2d_golf_ball_generated_api.py",
        POLYFEM_DATA_EXAMPLES / "2D" / "golf-ball.json",
    ),
    (
        CLASSIC_EXAMPLES
        / "2D"
        / "contact_2d_golf_ball_deformable_wall_generated_api.py",
        POLYFEM_DATA_EXAMPLES / "2D" / "golf-ball-doformable-wall.json",
    ),
    (
        CLASSIC_EXAMPLES / "2D" / "contact_2d_friction_circle_rollers_generated_api.py",
        POLYFEM_DATA_EXAMPLES / "2D" / "friction" / "circle-rollers.json",
    ),
    (
        CLASSIC_EXAMPLES / "3D" / "contact_3d_golf_ball_generated_api.py",
        POLYFEM_DATA_EXAMPLES / "3D" / "golf-ball.json",
    ),
    (
        CLASSIC_EXAMPLES
        / "3D"
        / "contact_3d_friction_high_school_slopetest_generated_api.py",
        POLYFEM_DATA_EXAMPLES
        / "3D"
        / "friction"
        / "high-school-physics-slopetest-mu=0.50.json",
    ),
    (
        CLASSIC_EXAMPLES / "3D" / "contact_3d_large_ratios_sphere_mat_generated_api.py",
        POLYFEM_DATA_EXAMPLES / "3D" / "large-ratios" / "sphere-mat.json",
    ),
]

DOLPHIN_MATERIAL_SHAPE_CASES = [
    (
        CLASSIC_EXAMPLES
        / "3D"
        / "contact_3d_stress_tests_dolphin_funnel_generated_api.py",
        POLYFEM_DATA_EXAMPLES / "3D" / "stress-tests" / "dolphin-funnel.json",
    ),
    (
        CLASSIC_EXAMPLES
        / "3D"
        / "contact_3d_stress_tests_dolphin_funnel_linf_generated_api.py",
        POLYFEM_DATA_EXAMPLES / "3D" / "stress-tests" / "dolphin-funnel-Linf.json",
    ),
]

EXPECTED_GENERATED_API_FALLBACK_SOURCES = {
    "3D/static/two-cubes.json",
}


def _ordinary_contact_source_paths() -> set[Path]:
    sources: set[Path] = set()
    for dim in ("2D", "3D"):
        for path in (POLYFEM_DATA_EXAMPLES / dim).rglob("*.json"):
            if _is_auxiliary_contact_json(path):
                continue
            sources.add(path.resolve())
    return sources


def _is_auxiliary_contact_json(path: Path) -> bool:
    return (
        path.name == "common.json"
        or path.name.endswith("-common.json")
        or path.name == "params.json"
    )


def _generated_example_files() -> list[Path]:
    return sorted(CLASSIC_EXAMPLES.glob("[23]D/contact_*_generated_api.py"))


def _declared_source_paths() -> set[Path]:
    return {
        Path(_import_example(path).SOURCE_JSON).resolve()
        for path in _generated_example_files()
    }


def _all_example_cases() -> list[tuple[Path, Path]]:
    cases = []
    for example_path in _generated_example_files():
        example = _import_example(example_path)
        cases.append((example_path, Path(example.SOURCE_JSON).resolve()))
    return cases


def _builder_safe_source_paths() -> set[Path]:
    safe = set()
    for source_path in _ordinary_contact_source_paths():
        config = _load_source_config(source_path)
        try:
            polyfem.config(**copy.deepcopy(config))
        except TypeError:
            continue
        if _is_builder_safe_config(config):
            safe.add(source_path)
    return safe


def _is_builder_safe_config(config: dict[str, Any]) -> bool:
    geometry = config.get("geometry")
    if not isinstance(geometry, list):
        return False
    if any(
        not isinstance(item, dict)
        or item.get("type") not in ("mesh", "mesh_array", "mesh_sequence")
        for item in geometry
    ):
        return False

    normal_geometry = [item for item in geometry if not item.get("is_obstacle")]
    if any(
        "volume_selection" in item
        and (
            not isinstance(item.get("volume_selection"), int)
            or item.get("volume_selection") <= 0
        )
        for item in normal_geometry
    ):
        return False
    volume_geometry = [
        item for item in normal_geometry
        if isinstance(item.get("volume_selection"), int)
        and item.get("volume_selection") > 0
    ]
    obstacle_surface_ids = {
        item.get("surface_selection")
        for item in geometry
        if item.get("is_obstacle")
        and isinstance(item.get("surface_selection"), int)
        and item.get("surface_selection") > 0
    }
    if any(
        item.get("is_obstacle")
        and item.get("point_selection") not in (None, 0)
        for item in geometry
    ):
        return False

    volume_ids = {
        item.get("volume_selection")
        for item in volume_geometry
        if isinstance(item.get("volume_selection"), int)
        and item.get("volume_selection") > 0
    }
    if not all(item.get("volume_selection") in volume_ids for item in volume_geometry):
        return False

    materials = _optional_items(config.get("materials"))
    if any(not isinstance(item, dict) for item in materials):
        return False
    id_materials = [item for item in materials if "id" in item]
    if any(item.get("id") not in volume_ids for item in id_materials):
        return False

    selection_ids = _builder_selection_ids(geometry)
    if selection_ids is None:
        return False
    if not _builder_safe_boundary_conditions(
        config.get("boundary_conditions"),
        selection_ids,
        obstacle_surface_ids,
    ):
        return False
    return _builder_safe_initial_conditions(
        config.get("initial_conditions"),
        volume_ids,
    )


def _builder_selection_ids(geometry: list[dict[str, Any]]) -> set[int] | None:
    selection_ids = set()
    for item in geometry:
        for field in ("surface_selection", "point_selection"):
            value = item.get(field)
            if value is None:
                continue
            if isinstance(value, int):
                if value > 0:
                    selection_ids.add(value)
                continue
            if isinstance(value, dict):
                items = [value]
            elif isinstance(value, list):
                items = value
            else:
                return None
            for selection in items:
                if not isinstance(selection, dict):
                    return None
                selection_id = selection.get("id")
                if not isinstance(selection_id, int) or selection_id <= 0:
                    return None
                selection_ids.add(selection_id)
    return selection_ids


def _builder_safe_boundary_conditions(
    boundary_conditions: Any,
    selection_ids: set[int],
    obstacle_surface_ids: set[int],
) -> bool:
    if boundary_conditions is None:
        return True
    if not isinstance(boundary_conditions, dict):
        return False

    bindable = {
        "dirichlet_boundary",
        "neumann_boundary",
        "normal_aligned_neumann_boundary",
        "pressure_boundary",
        "pressure_cavity",
        "obstacle_displacements",
    }
    if any(key not in {"rhs", *bindable} for key in boundary_conditions):
        return False

    for section in bindable:
        for item in _optional_items(boundary_conditions.get(section)):
            if not isinstance(item, dict):
                return False
            expected_ids = (
                obstacle_surface_ids
                if section == "obstacle_displacements"
                else selection_ids
            )
            if item.get("id") not in expected_ids:
                return False
    return True


def _builder_safe_initial_conditions(
    initial_conditions: Any,
    volume_ids: set[int],
) -> bool:
    if initial_conditions is None:
        return True
    if not isinstance(initial_conditions, dict):
        return False

    bindable = {"velocity", "solution", "acceleration"}
    if any(key not in bindable for key in initial_conditions):
        return False

    for section in bindable:
        for item in _optional_items(initial_conditions.get(section)):
            if not isinstance(item, dict) or item.get("id") not in volume_ids:
                return False
    return True


def _optional_items(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _case_id(case: tuple[Path, Path]) -> str:
    example_path, _source_path = case
    return example_path.stem


def _require_examples_submodule() -> None:
    if not CLASSIC_EXAMPLES.exists():
        pytest.skip(
            "examples submodule is not initialized; run "
            "`git submodule update --init examples`"
        )


def _require_polyfem_data_submodule() -> None:
    if not POLYFEM_DATA_EXAMPLES.exists():
        pytest.skip(
            "polyfem-data submodule is not initialized; run "
            "`git submodule update --init polyfem-data`"
        )


@pytest.fixture(autouse=True)
def _skip_without_required_submodules():
    _require_examples_submodule()
    _require_polyfem_data_submodule()


def test_classic_generated_examples_cover_all_contact_source_jsons():
    assert _declared_source_paths() == _ordinary_contact_source_paths()


def test_classic_generated_examples_show_api_configuration():
    offenders = []
    fallback_sources = {
        (POLYFEM_DATA_EXAMPLES / source).resolve()
        for source in EXPECTED_GENERATED_API_FALLBACK_SOURCES
    }
    for example_path, source_path in _all_example_cases():
        text = example_path.read_text(encoding="utf-8")
        has_constructor_config = (
            "polyfem.config(" in text or "model.config(" in text
        )
        has_known_fallback = (
            source_path in fallback_sources and "SourcePayloadConfig(" in text
        )
        if "load_polyfem_config(SOURCE_JSON)" in text:
            offenders.append(example_path.relative_to(ROOT).as_posix())
        elif not has_constructor_config and not has_known_fallback:
            offenders.append(example_path.relative_to(ROOT).as_posix())

    assert offenders == []


def test_builder_safe_classic_examples_use_model_builder():
    offenders = []
    builder_safe_sources = _builder_safe_source_paths()
    for example_path, source_path in _all_example_cases():
        if source_path not in builder_safe_sources:
            continue
        text = example_path.read_text(encoding="utf-8")
        if "model = polyfem.model()" not in text or "polyfem_config = model.config(" not in text:
            offenders.append(example_path.relative_to(ROOT).as_posix())

    assert offenders == []


def test_obstacle_mesh_classic_example_uses_model_builder():
    example_path = (
        CLASSIC_EXAMPLES
        / "2D"
        / "contact_2d_unit_tests_erleben_spikes_generated_api.py"
    )
    source_path = (
        POLYFEM_DATA_EXAMPLES
        / "2D"
        / "unit-tests"
        / "erleben"
        / "spikes.json"
    )

    assert source_path.resolve() in _builder_safe_source_paths()

    text = example_path.read_text(encoding="utf-8")
    assert "model = polyfem.model()" in text
    assert "polyfem_config = model.config(" in text


def test_model_builder_obstacle_mesh_does_not_create_volume_selection():
    model = polyfem.model()

    obstacle = model.obstacle_mesh(
        mesh="obstacle.obj",
        surface_selection=0,
    )
    config = model.config()

    assert obstacle is None
    assert config.as_dict()["geometry"] == [
        {
            "mesh": "obstacle.obj",
            "surface_selection": 0,
            "is_obstacle": True,
            "type": "mesh",
        }
    ]


def test_global_material_geometry_example_uses_model_builder():
    example_path = (
        CLASSIC_EXAMPLES
        / "2D"
        / "contact_2d_static_friction_slope_generated_api.py"
    )
    source_path = POLYFEM_DATA_EXAMPLES / "2D" / "static" / "friction-slope.json"

    assert source_path.resolve() in _builder_safe_source_paths()

    text = example_path.read_text(encoding="utf-8")
    assert "model = polyfem.model()" in text
    assert "model.geometry_mesh(" in text
    assert "polyfem_config = model.config(" in text


def test_model_builder_geometry_mesh_does_not_create_volume_selection():
    model = polyfem.model()

    geometry = model.geometry_mesh(
        mesh="static-wall.obj",
    )
    surface = geometry.surface_all(id=1)
    surface.dirichlet(value=[0, 0])
    config = model.config()

    assert config.as_dict()["geometry"] == [
        {
            "mesh": "static-wall.obj",
            "surface_selection": 1,
            "type": "mesh",
        }
    ]
    assert config.as_dict()["boundary_conditions"]["dirichlet_boundary"] == [
        {
            "id": 1,
            "value": [0, 0],
        }
    ]


def test_single_dict_selection_example_uses_model_builder():
    example_path = (
        CLASSIC_EXAMPLES
        / "3D"
        / "contact_3d_unit_tests_2_cubes_generated_api.py"
    )
    source_path = POLYFEM_DATA_EXAMPLES / "3D" / "unit-tests" / "2-cubes.json"

    assert source_path.resolve() in _builder_safe_source_paths()

    text = example_path.read_text(encoding="utf-8")
    assert "model = polyfem.model()" in text
    assert ".surface_axis(" in text
    assert "polyfem_config = model.config(" in text


def test_mesh_sequence_example_uses_model_builder():
    example_path = (
        CLASSIC_EXAMPLES
        / "3D"
        / "contact_3d_mesh_sequence_kick_generated_api.py"
    )
    source_path = POLYFEM_DATA_EXAMPLES / "3D" / "mesh-sequence" / "kick.json"

    assert source_path.resolve() in _builder_safe_source_paths()

    text = example_path.read_text(encoding="utf-8")
    assert "model = polyfem.model()" in text
    assert "model.obstacle_mesh_sequence(" in text
    assert "polyfem_config = model.config(" in text


def test_model_builder_mesh_sequence_preserves_payload():
    model = polyfem.model()

    obstacle = model.obstacle_mesh_sequence(
        mesh_sequence="kick-sequence/",
        fps=24,
    )
    config = model.config()

    assert obstacle is None
    assert config.as_dict()["geometry"] == [
        {
            "mesh_sequence": "kick-sequence/",
            "fps": 24,
            "is_obstacle": True,
            "type": "mesh_sequence",
        }
    ]


def test_mesh_array_pile_example_uses_model_builder():
    example_path = (
        CLASSIC_EXAMPLES
        / "3D"
        / "contact_3d_pile_cubes_generated_api.py"
    )
    source_path = POLYFEM_DATA_EXAMPLES / "3D" / "pile" / "cubes.json"

    assert source_path.resolve() in _builder_safe_source_paths()

    text = example_path.read_text(encoding="utf-8")
    assert "model = polyfem.model()" in text
    assert "model.mesh_array(" in text
    assert "polyfem_config = model.config(" in text


def test_model_builder_mesh_array_without_volume_selection_preserves_payload():
    model = polyfem.model()

    mesh_array = model.mesh_array(
        mesh="cube.msh",
        array=polyfem.array(
            size=[2, 3, 4],
            offset=1.25,
        ),
    )
    config = model.config()

    assert mesh_array is None
    assert config.as_dict()["geometry"] == [
        {
            "mesh": "cube.msh",
            "array": {
                "size": [2, 3, 4],
                "offset": 1.25,
            },
            "type": "mesh_array",
        }
    ]


def test_obstacle_displacement_example_uses_model_builder():
    example_path = (
        CLASSIC_EXAMPLES
        / "3D"
        / "contact_3d_friction_ball_rollers_generated_api.py"
    )
    source_path = (
        POLYFEM_DATA_EXAMPLES
        / "3D"
        / "friction"
        / "ball-rollers.json"
    )

    assert source_path.resolve() in _builder_safe_source_paths()

    text = example_path.read_text(encoding="utf-8")
    assert "model = polyfem.model()" in text
    assert ".obstacle_displacement(" in text


def test_model_builder_can_bind_obstacle_surface_displacement():
    model = polyfem.model()

    surface = model.obstacle_mesh(
        mesh="roller.obj",
        surface_selection=1000,
    )
    surface.obstacle_displacement(value=[1, 0, 0])
    config = model.config()

    assert config.as_dict()["geometry"] == [
        {
            "mesh": "roller.obj",
            "surface_selection": 1000,
            "is_obstacle": True,
            "type": "mesh",
        }
    ]
    assert config.as_dict()["boundary_conditions"]["obstacle_displacements"] == [
        {
            "id": 1000,
            "value": [1, 0, 0],
        }
    ]


def _import_example(example_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(example_path.stem, example_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _import_module(module_path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    return module


def _deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        merged = copy.deepcopy(base)
        for key, value in override.items():
            merged[key] = (
                _deep_merge(merged[key], value)
                if key in merged
                else copy.deepcopy(value)
            )
        return merged
    return copy.deepcopy(override)


def _load_source_config(source_path: Path) -> dict[str, Any]:
    source_path = source_path.resolve()
    with source_path.open(encoding="utf-8") as f:
        source = json.load(f)

    common_ref = source.pop("common", None)
    source.pop("tests", None)
    source.pop("default_params", None)
    patch = source.pop("patch", None)

    if common_ref is None:
        merged = source
    else:
        common_path = (source_path.parent / common_ref).resolve()
        merged = _deep_merge(_load_source_config(common_path), source)

    if patch:
        merged = _apply_json_patch(merged, patch)

    _resolve_mesh_paths(merged, source_path.parent)
    _normalize_solver_fallbacks(merged)
    _normalize_generated_mesh_payloads(merged)
    return merged


def _expected_config_from_source(source_path: Path) -> dict[str, Any]:
    config = _load_source_config(source_path)
    try:
        return polyfem.config(**copy.deepcopy(config)).as_dict()
    except TypeError:
        return config


def _apply_json_patch(config: dict[str, Any], patch: list[dict[str, Any]]) -> dict[str, Any]:
    patched = copy.deepcopy(config)
    for item in patch:
        assert item.get("op") == "replace"
        _replace_json_pointer(patched, item["path"], item.get("value"))
    return patched


def _replace_json_pointer(config: Any, pointer: str, value: Any) -> None:
    target = config
    tokens = _json_pointer_tokens(pointer)
    for token in tokens[:-1]:
        target = target[int(token)] if isinstance(target, list) else target[token]

    last = tokens[-1]
    if isinstance(target, list):
        target[int(last)] = copy.deepcopy(value)
    else:
        target[last] = copy.deepcopy(value)


def _json_pointer_tokens(pointer: str) -> list[str]:
    assert pointer.startswith("/")
    return [
        token.replace("~1", "/").replace("~0", "~")
        for token in pointer[1:].split("/")
    ]


def _resolve_mesh_paths(value: Any, source_dir: Path) -> None:
    if isinstance(value, list):
        for item in value:
            _resolve_mesh_paths(item, source_dir)
    elif isinstance(value, dict):
        for key in ("mesh", "linear_map"):
            path_value = value.get(key)
            if isinstance(path_value, str):
                value[key] = str((source_dir / path_value).resolve())
        for child in value.values():
            _resolve_mesh_paths(child, source_dir)


def _normalize_solver_fallbacks(config: dict[str, Any]) -> None:
    solver = config.get("solver")
    if not isinstance(solver, dict):
        return
    linear = solver.get("linear")
    if not isinstance(linear, dict):
        return
    solver_name = linear.get("solver")
    if isinstance(solver_name, list):
        linear["solver"] = solver_name[0]


def _normalize_generated_mesh_payloads(config: dict[str, Any]) -> None:
    geometry = config.get("geometry")
    if not isinstance(geometry, list):
        return

    for item in geometry:
        if not isinstance(item, dict) or "mesh" not in item:
            continue
        item.setdefault("type", "mesh")

        transformation = item.get("transformation")
        if not isinstance(transformation, dict):
            continue
        for field in ("rotation", "scale"):
            if isinstance(transformation.get(field), (int, float)):
                transformation[field] = [float(transformation[field])]


@pytest.mark.parametrize("case", EXAMPLE_CASES, ids=_case_id)
def test_classic_generated_example_compiles(case: tuple[Path, Path]):
    example_path, _source_path = case
    py_compile.compile(str(example_path), doraise=True)


@pytest.mark.parametrize("case", EXAMPLE_CASES, ids=_case_id)
def test_classic_generated_example_declares_polyfem_data_source(case: tuple[Path, Path]):
    example_path, source_path = case
    example = _import_example(example_path)

    assert Path(example.SOURCE_JSON).resolve() == source_path.resolve()
    assert POLYFEM_DATA_CONTACT in Path(example.SOURCE_JSON).resolve().parents


def test_classic_common_helpers_support_polyfem_data_root_env(monkeypatch, tmp_path):
    data_root = tmp_path / "polyfem-data-alt"
    monkeypatch.setenv("POLYFEM_DATA_ROOT", str(data_root))

    common_2d = _import_module(
        CLASSIC_EXAMPLES / "2D" / "_contact_2d_common.py",
        "contact_2d_common_env_test",
    )
    common_3d = _import_module(
        CLASSIC_EXAMPLES / "3D" / "_contact_3d_common.py",
        "contact_3d_common_env_test",
    )

    assert common_2d.POLYFEM_DATA_ROOT == data_root.resolve()
    assert common_2d.POLYFEM_DATA_CONTACT == data_root.resolve() / "contact"
    assert common_2d.CONTACT_EXAMPLES_DIR == data_root.resolve() / "contact" / "examples"
    assert common_2d.MESHES_DIR == data_root.resolve() / "contact" / "meshes"
    assert common_3d.POLYFEM_DATA_ROOT == data_root.resolve()
    assert common_3d.POLYFEM_DATA_CONTACT == data_root.resolve() / "contact"


@pytest.mark.parametrize("case", EXAMPLE_CASES, ids=_case_id)
def test_classic_generated_example_matches_polyfem_data_source(case: tuple[Path, Path]):
    example_path, source_path = case
    example = _import_example(example_path)

    assert example.polyfem_config.as_dict() == _expected_config_from_source(source_path)


def test_all_classic_generated_examples_match_polyfem_data_sources():
    mismatches = []
    for example_path, source_path in _all_example_cases():
        example = _import_example(example_path)
        if example.polyfem_config.as_dict() != _expected_config_from_source(source_path):
            mismatches.append(example_path.relative_to(ROOT).as_posix())

    assert mismatches == []


@pytest.mark.parametrize("case", DOLPHIN_MATERIAL_SHAPE_CASES, ids=_case_id)
def test_dolphin_generated_backend_payload_keeps_source_materials_shape(
    case: tuple[Path, Path],
):
    example_path, source_path = case
    source_config = _load_source_config(source_path)
    example = _import_example(example_path)

    canonical = prepare_canonical_solve_input(
        vertices=None,
        cells=None,
        cfg=example.polyfem_config,
        dtype=None,
    )

    assert isinstance(source_config["materials"], dict)
    assert canonical.backend_settings["materials"] == source_config["materials"]


def test_classic_generated_api_fallbacks_are_limited_to_known_schema_gaps():
    actual = set()
    for _example_path, source_path in _all_example_cases():
        example = _import_example(_example_path)
        fallback_reason = getattr(example.polyfem_config, "fallback_reason", None)
        if fallback_reason:
            actual.add(source_path.relative_to(POLYFEM_DATA_EXAMPLES).as_posix())

    assert actual == EXPECTED_GENERATED_API_FALLBACK_SOURCES


@pytest.mark.parametrize("case", EXAMPLE_CASES, ids=_case_id)
def test_classic_generated_example_uses_generated_solve_path(case: tuple[Path, Path]):
    example_path, _source_path = case
    example = _import_example(example_path)

    canonical = prepare_canonical_solve_input(
        vertices=None,
        cells=None,
        cfg=example.polyfem_config,
        dtype=None,
    )

    assert canonical.metadata == {"mesh_source": "json", "config_source": "generated"}
    assert canonical.mesh_source.mode == "json"
    assert "geometry" in canonical.backend_settings


def test_all_classic_generated_examples_use_generated_solve_path():
    failures = []
    for example_path, _source_path in _all_example_cases():
        example = _import_example(example_path)
        canonical = prepare_canonical_solve_input(
            vertices=None,
            cells=None,
            cfg=example.polyfem_config,
            dtype=None,
        )
        if canonical.metadata != {"mesh_source": "json", "config_source": "generated"}:
            failures.append(example_path.relative_to(ROOT).as_posix())

    assert failures == []
