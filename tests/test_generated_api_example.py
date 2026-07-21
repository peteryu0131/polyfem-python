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


def _case_id(case: tuple[Path, Path]) -> str:
    example_path, _source_path = case
    return example_path.stem


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
            merged[key] = _deep_merge(merged[key], value) if key in merged else copy.deepcopy(value)
        return merged
    return copy.deepcopy(override)


def _load_source_config(source_path: Path) -> dict[str, Any]:
    with source_path.open(encoding="utf-8") as f:
        source = json.load(f)

    common_ref = source.pop("common", None)
    source.pop("tests", None)

    if common_ref is None:
        merged = source
    else:
        common_path = (source_path.parent / common_ref).resolve()
        with common_path.open(encoding="utf-8") as f:
            common = json.load(f)
        merged = _deep_merge(common, source)

    _resolve_mesh_paths(merged, source_path.parent)
    _normalize_solver_fallbacks(merged)
    _normalize_generated_mesh_payloads(merged)
    return merged


def _resolve_mesh_paths(value: Any, source_dir: Path) -> None:
    if isinstance(value, list):
        for item in value:
            _resolve_mesh_paths(item, source_dir)
    elif isinstance(value, dict):
        mesh = value.get("mesh")
        if isinstance(mesh, str):
            value["mesh"] = str((source_dir / mesh).resolve())
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

    assert example.polyfem_config.as_dict() == _load_source_config(source_path)


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
