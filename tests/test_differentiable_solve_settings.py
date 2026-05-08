from __future__ import annotations

from polyfempy.differentiable._solve_settings import (
    _apply_internal_differentiable_runtime_patches,
    _console_log_level_from_settings,
    _geometry_uses_only_absolute_mesh_paths,
)


def test_console_log_level_from_settings_defaults_and_named_levels():
    assert _console_log_level_from_settings({}) == 2
    assert _console_log_level_from_settings({"output": {"log": {"level": "warn"}}}) == 3
    assert _console_log_level_from_settings({"output": {"log": {"level": "off"}}}) == 6


def test_geometry_absolute_mesh_path_detection():
    assert _geometry_uses_only_absolute_mesh_paths(
        {"geometry": [{"mesh": "/tmp/body.msh"}]}
    )
    assert not _geometry_uses_only_absolute_mesh_paths(
        {"geometry": [{"mesh": "relative/body.msh"}]}
    )
    assert not _geometry_uses_only_absolute_mesh_paths(
        {"geometry": [{"type": "ground", "height": 0.0}]}
    )


def test_runtime_patch_sets_constant_differentiable_barrier_stiffness():
    settings = {"solver": {"contact": {"barrier_stiffness": "adaptive"}}}

    _apply_internal_differentiable_runtime_patches(config=None, settings=settings)

    assert settings["solver"]["contact"]["barrier_stiffness"] == 1e3
